# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CriticAgent — VLM-backed result verifier. EXPERIMENTAL in v1.

Ported from services/agent/src/vss_agents/agents/critic_agent.py:208-307 with
two adaptations:

  - The VLM caller comes from an injected VLMAnalyzer protocol (DESIGN.md §16.1
    option (b)) instead of NAT's `builder.get_function("video_understanding")`.
    Callers must supply a vlm_analyzer; the library cannot construct a default.

  - time_format="offset" path uses VSTSnapshot.resolve_stream_id and
    get_timeline to convert ISO timestamps to seconds-since-stream-start
    (matches agents/critic_agent.py:248-264). Both VST surfaces are wired up
    (clients/vst.py delegates to the existing tools.vst.* helpers).

Stable v1 contract requires resolving DESIGN.md §16.1. Until then, this
module's constructor signature, the VLMAnalyzer protocol, and the wire format
of CriticAgentOutput may all change.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING

from .._internal.time_convert import datetime_to_iso8601
from ..models.critic import CriticAgentInput
from ..models.critic import CriticAgentOutput
from ..models.critic import CriticAgentResult
from ..models.critic import TimeFormat
from ..models.critic import VideoResult

if TYPE_CHECKING:
    from ..clients.protocols import VLMAnalyzer
    from ..clients.protocols import VSTSnapshot
    from ..models.common import VideoInfo
    from ..runtime import SearchRuntime

logger = logging.getLogger(__name__)


# Mirrors agents/critic_agent.py:CRITIC_AGENT_PROMPT (substitute when needed).
DEFAULT_CRITIC_PROMPT = (
    "You are a video verification assistant. Given the user query and a short clip, "
    "answer whether the clip confirms or rejects the query, and list which named "
    "criteria are met. Reply with JSON: "
    '{{"result": "confirmed"|"rejected"|"unverified", "criteria_met": {{...}}}}. '
    "User prompt: {user_prompt}"
)


def _parse_iso(s: str | datetime) -> datetime:
    """Parse an ISO-8601 string into a datetime (passthrough if already datetime)."""
    if isinstance(s, datetime):
        return s
    normalized = s.rstrip("Z")
    return datetime.fromisoformat(normalized)


def _to_offset(ts: datetime | str, clip_start: datetime) -> float:
    """Convert an absolute timestamp to seconds since clip_start."""
    if isinstance(ts, str):
        ts = _parse_iso(ts)
    # Both datetimes need matching tz-awareness; if either is naive, normalize.
    if ts.tzinfo is None and clip_start.tzinfo is not None:
        ts = ts.replace(tzinfo=clip_start.tzinfo)
    elif ts.tzinfo is not None and clip_start.tzinfo is None:
        clip_start = clip_start.replace(tzinfo=ts.tzinfo)
    return (ts - clip_start).total_seconds()


def _extract_json(text: str) -> str:
    """Strip ```json fences from VLM output if present.

    Mirrors agents/critic_agent.py:get_json_from_string.
    """
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    return text


def _parse_criteria(vlm_text: str) -> tuple[CriticAgentResult, dict[str, bool]]:
    """Parse the VLM's JSON response into (verdict, criteria_met).

    Mirrors agents/critic_agent.py:281-291. On parse failure, returns
    (UNVERIFIED, {}). On any criterion = False, verdict is REJECTED.
    Otherwise CONFIRMED.
    """
    try:
        payload = json.loads(_extract_json(vlm_text))
        if not isinstance(payload, dict):
            raise TypeError(f"expected JSON object, got {type(payload).__name__}")

        explicit_result = payload.get("result")
        if isinstance(payload.get("criteria_met"), dict):
            raw_criteria = payload["criteria_met"]
        else:
            # No explicit criteria_met: treat the remaining top-level keys as
            # criteria, but drop the reserved 'result' verdict key so it isn't
            # bool()'d into a (possibly failing) criterion — e.g. {"result": ""}
            # would otherwise flip an otherwise-confirmable clip to REJECTED.
            raw_criteria = {k: v for k, v in payload.items() if k != "result"}
        criteria = {str(k): bool(v) for k, v in raw_criteria.items()}

        if isinstance(explicit_result, str):
            normalized = explicit_result.strip().lower()
            if normalized == CriticAgentResult.UNVERIFIED.value:
                return CriticAgentResult.UNVERIFIED, criteria
            if normalized == CriticAgentResult.REJECTED.value:
                return CriticAgentResult.REJECTED, criteria

        verdict = CriticAgentResult.CONFIRMED
        for v in criteria.values():
            if not v:
                verdict = CriticAgentResult.REJECTED
                break
        return verdict, criteria
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.error(f"Error parsing VLM response: {e}")
        return CriticAgentResult.UNVERIFIED, {}


class CriticAgent:
    """VLM-backed verification of search results."""

    def __init__(
        self,
        *,
        vlm_analyzer: VLMAnalyzer,
        vst: VSTSnapshot,
        prompt: str = DEFAULT_CRITIC_PROMPT,
        max_concurrent_verifications: int = 5,
        time_format: TimeFormat = "iso",
        num_videos_to_evaluate: int | None = None,
    ) -> None:
        self._vlm = vlm_analyzer
        self._vst = vst
        self._prompt = prompt
        self._max_concurrent = max_concurrent_verifications
        self._time_format = time_format
        self._default_eval_count = num_videos_to_evaluate

    async def run(self, inp: CriticAgentInput) -> CriticAgentOutput:
        """Verify each input video with the VLM; return per-video verdicts."""
        # Cap the eval count: explicit input takes precedence over the
        # constructor default; both are capped by the actual list length.
        video_count = min(
            inp.evaluation_count or self._default_eval_count or len(inp.videos),
            len(inp.videos),
        )
        semaphore = asyncio.Semaphore(self._max_concurrent)

        # Skip entries without a sensor_id (matches original at line 296).
        candidates = [v for v in inp.videos[:video_count] if v.sensor_id]

        tasks = [self._evaluate_video(semaphore, v, inp.query) for v in candidates]
        results = await asyncio.gather(*tasks)

        confirmed = sum(1 for r in results if r.result == CriticAgentResult.CONFIRMED)
        rejected = sum(1 for r in results if r.result == CriticAgentResult.REJECTED)
        logger.info(f"Critic agent: {confirmed} confirmed, {rejected} rejected, {len(results)} total")
        return CriticAgentOutput(video_results=results)

    async def _evaluate_video(
        self,
        semaphore: asyncio.Semaphore,
        video: VideoInfo,
        query: str,
    ) -> VideoResult:
        """Evaluate a single video against the user query via the VLM."""
        async with semaphore:
            formatted_prompt = self._prompt.format(user_prompt=query)
            logger.debug(f"Formatted prompt: {formatted_prompt}")

            try:
                if self._time_format == "iso":
                    # Emit the VSS-canonical 'Z'-suffixed ISO form (what the
                    # rest of the system and the legacy critic used), not
                    # datetime.isoformat()'s '+00:00' form, so a downstream
                    # video-analysis tool doing exact-string handling matches.
                    vlm_response = await self._vlm.analyze(
                        sensor_id=video.sensor_id,
                        start_timestamp=datetime_to_iso8601(video.start_timestamp),
                        end_timestamp=datetime_to_iso8601(video.end_timestamp),
                        prompt=formatted_prompt,
                        time_format="iso",
                    )
                else:
                    # offset-time: convert ISO timestamps to seconds-since-stream-start
                    # using VST's timeline endpoint. Mirrors agents/critic_agent.py:248-264.
                    stream_id = await self._vst.resolve_stream_id(video.sensor_id)
                    if stream_id is None:
                        raise ValueError(f"VST stream_id resolution failed for sensor {video.sensor_id}")
                    clip_start_iso, clip_end_iso = await self._vst.get_timeline(stream_id)
                    clip_start_dt = _parse_iso(clip_start_iso)
                    start_offset = _to_offset(video.start_timestamp, clip_start_dt)
                    end_offset = _to_offset(video.end_timestamp, clip_start_dt)
                    # Clamp end_offset to the clip's actual end if the caller asked
                    # for more than is available (matches original L259-260).
                    clip_end_offset = _to_offset(_parse_iso(clip_end_iso), clip_start_dt)
                    if end_offset > clip_end_offset:
                        end_offset = clip_end_offset
                    vlm_response = await self._vlm.analyze(
                        sensor_id=video.sensor_id,
                        start_timestamp=str(start_offset),
                        end_timestamp=str(end_offset),
                        prompt=formatted_prompt,
                        time_format="offset",
                    )
            except Exception as e:
                logger.error(f"Error calling VLM analyzer: {e}")
                return VideoResult(
                    video_info=video,
                    result=CriticAgentResult.UNVERIFIED,
                    criteria_met={},
                )

            logger.info(f"VLM response for {video.sensor_id}: {vlm_response}")
            verdict, criteria = _parse_criteria(vlm_response)
            logger.debug(f"Video {video.sensor_id} verdict={verdict.value} criteria={criteria}")
            return VideoResult(video_info=video, result=verdict, criteria_met=criteria)

    @classmethod
    def from_runtime(
        cls,
        rt: SearchRuntime,
        *,
        vlm_analyzer: VLMAnalyzer,
        vst: VSTSnapshot | None = None,
        time_format: TimeFormat = "iso",
        prompt: str | None = None,
        num_videos_to_evaluate: int | None = None,
    ) -> CriticAgent:
        """Construct from a SearchRuntime.

        `vlm_analyzer` is REQUIRED — there is no library-provided default. The
        NAT adapter passes one wrapping the existing `video_understanding` tool;
        host facades require explicit injection. See DESIGN.md §6.4 and §16.1.

        ``prompt`` and ``num_videos_to_evaluate`` let callers override the
        constructor defaults without reaching into private attributes after
        construction.
        """
        from ..clients.vst import VSTClient  # local — avoid cycle

        return cls(
            vlm_analyzer=vlm_analyzer,
            vst=vst or VSTClient.from_runtime(rt),
            prompt=prompt if prompt is not None else DEFAULT_CRITIC_PROMPT,
            max_concurrent_verifications=rt.max_concurrent_verifications,
            time_format=time_format,
            num_videos_to_evaluate=num_videos_to_evaluate,
        )

    async def aclose(self) -> None:
        return None
