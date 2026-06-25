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
"""VSTClient — VST surface used by primitives.

Includes the VST helpers (get_name_to_stream_id_map, get_stream_id, get_timeline)
ported from services/agent/src/vss_agents/tools/vst/{utils,timeline}.py with
two adjustments: no env reads (callers must pass internal URL explicitly),
and the retry exception tuple is widened to `Exception` to match the originals'
intent (they wrap `RuntimeError`/`VSTError` which aren't aiohttp types).

build_screenshot_url stays a free function for callers that don't need the
OO wrapper.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import aiohttp

from .._internal.retry import create_retry_strategy
from .._internal.time_convert import iso8601_to_datetime

if TYPE_CHECKING:
    from ..runtime import SearchRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- types


class VSTError(Exception):
    """Base exception for VST API errors. Mirrors tools/vst/utils.py:64."""


# ----------------------------------------------------------------- free helpers


def build_screenshot_url(vst_external_url: str, stream_id: str, timestamp: str) -> str:
    """Build a client-facing screenshot URL.

    Mirrors tools/vst/snapshot.py:49. Pure string composition.
    """
    vst_external_url = vst_external_url.rstrip("/")
    return f"{vst_external_url}/vst/api/v1/replay/stream/{stream_id}/picture?startTime={timestamp}"


async def get_name_to_stream_id_map(vst_internal_url: str) -> dict[str, str]:
    """Fetch `/api/v1/sensor/streams` and return `{sensor_name: stream_id}`.

    Mirrors tools/vst/utils.py:70-97 with the env-fallback removed.
    """
    url = f"{vst_internal_url.rstrip('/')}/vst/api/v1/sensor/streams"
    async with aiohttp.ClientSession() as session:
        async for retry in create_retry_strategy(retries=3, exceptions=(Exception,)):
            with retry:
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            raise RuntimeError(f"VST streams API returned status {response.status}")
                        text = await response.text()
                        payload = json.loads(text)
                        mapping: dict[str, str] = {}
                        for file in payload:
                            stream_id = next(iter(file))
                            if isinstance(file[stream_id], list) and len(file[stream_id]) > 0:
                                name = file[stream_id][0]["name"]
                                mapping[name] = stream_id
                            else:
                                logger.warning(f"Stream ID {stream_id} is empty, skipping")
                        return mapping
                except Exception as e:
                    logger.error(f"Error getting name to stream ID map: {e}")
                    raise
    return {}  # unreachable; satisfies mypy


async def get_streams_info(vst_internal_url: str) -> dict[str, dict[str, str]]:
    """Return `{stream_id: {"name": name, "url": rtsp_url}}` from VST.

    Mirrors tools/vst/utils.py:420-453. Used by the Search orchestrator to
    resolve video_sources by name when source_type='rtsp'.
    """
    url = f"{vst_internal_url.rstrip('/')}/vst/api/v1/sensor/streams"
    async with aiohttp.ClientSession() as session:
        async for retry in create_retry_strategy(retries=3, exceptions=(Exception,)):
            with retry:
                try:
                    async with session.get(url) as response:
                        if response.status != 200:
                            raise VSTError(f"VST streams API returned status {response.status}")
                        text = await response.text()
                        payload = json.loads(text)
                        result: dict[str, dict[str, str]] = {}
                        for entry in payload:
                            stream_id = next(iter(entry))
                            stream_list = entry[stream_id]
                            if stream_list and len(stream_list) > 0:
                                result[stream_id] = {
                                    "name": stream_list[0].get("name", ""),
                                    "url": stream_list[0].get("url", ""),
                                }
                        return result
                except Exception as e:
                    logger.error(f"Error getting streams info: {e}")
                    raise
    return {}  # unreachable; satisfies mypy


async def get_stream_id(sensor_id: str, vst_internal_url: str) -> str:
    """Resolve sensor_id → stream_id via VST. Mirrors tools/vst/utils.py:99-117.

    ``sensor_id`` may already be a stream_id (UUID); the function tolerates that.
    """
    stream_id_map = await get_name_to_stream_id_map(vst_internal_url)
    stream_id = stream_id_map.get(sensor_id)
    if not stream_id:
        if sensor_id in stream_id_map.values():
            stream_id = sensor_id
        else:
            raise VSTError(
                f"streamId not found for '{sensor_id}'. Available: {sorted(stream_id_map.keys())}"
                if stream_id_map
                else "streamId not found"
            )
    return stream_id


async def get_sensor_id_from_stream_id(stream_id: str, vst_internal_url: str) -> str:
    """Reverse lookup: stream_id (UUID) → sensor_id (camera name).

    Mirrors tools/vst/utils.py:119-153. If ``stream_id`` is already a sensor
    name (and present in the VST map), returns it as-is. Raises VSTError on miss.
    """
    name_to_stream_id_map = await get_name_to_stream_id_map(vst_internal_url)
    stream_id_to_name_map = {sid: name for name, sid in name_to_stream_id_map.items()}
    sensor_id = stream_id_to_name_map.get(stream_id)
    if not sensor_id:
        if stream_id in name_to_stream_id_map:
            sensor_id = stream_id
        else:
            raise VSTError(
                f"sensorId not found for stream_id '{stream_id}'. "
                f"Available stream_ids: {sorted(stream_id_to_name_map.keys())[:10]}..."
                if stream_id_to_name_map
                else "sensorId not found"
            )
    return sensor_id


async def get_timeline(stream_id: str, vst_internal_url: str) -> tuple[str, str]:
    """Return (start_iso, end_iso) for a stream's replay timeline.

    Mirrors tools/vst/timeline.py:69-125. Tolerates being given a sensor name
    instead of a stream_id (re-resolves via get_stream_id if the first lookup
    misses). Raises VSTError if the timeline is missing or shorter than 1s.
    """
    # Defensive: drop a trailing /vst if some caller already added it. Strip
    # trailing slashes FIRST so '<url>/vst/' is handled too — otherwise the
    # suffix check misses and the path doubles to '<url>/vst/vst/api/...'.
    base = vst_internal_url.rstrip("/")
    if base.endswith("/vst"):
        base = base[:-4]
    timelines_url = f"{base}/vst/api/v1/storage/timelines"

    async with aiohttp.ClientSession() as session:
        async for retry in create_retry_strategy(retries=3, exceptions=(Exception,)):
            with retry:
                try:
                    async with session.get(timelines_url) as response:
                        if response.status != 200:
                            raise RuntimeError(f"VST timelines API returned status {response.status}")
                        text = await response.text()
                        timelines_data = json.loads(text)
                        timeline_list = timelines_data.get(stream_id, [])
                        if not timeline_list:
                            logger.info("no timeline for input; trying to resolve as sensor name")
                            stream_id = await get_stream_id(stream_id, vst_internal_url)
                            timeline_list = timelines_data.get(stream_id, [])
                            if not timeline_list:
                                raise VSTError(f"No timeline found for stream {stream_id}")
                        logger.info("Timeline for stream %s: %s", stream_id, timeline_list)
                        start = timeline_list[0].get("startTime")
                        end = timeline_list[0].get("endTime")
                        start_dt = iso8601_to_datetime(start)
                        end_dt = iso8601_to_datetime(end)
                        if (end_dt - start_dt).total_seconds() < 1:
                            raise VSTError(f"Timeline duration is too short for stream {stream_id}")
                        return start, end
                except Exception as e:
                    raise VSTError(f"Error getting timeline for stream {stream_id}: {e}") from e
    return "", ""  # unreachable; satisfies mypy


# ---------------------------------------------------------------------- client


class VSTClient:
    """Implements the VSTSnapshot protocol.

    All methods accept the URL via constructor (typically from SearchRuntime);
    no env reads. resolve_stream_id and get_timeline forward to the free
    helpers above.
    """

    def __init__(self, *, internal_url: str, external_url: str) -> None:
        self._internal_url = internal_url
        self._external_url = external_url

    @classmethod
    def from_runtime(cls, rt: SearchRuntime) -> VSTClient:
        return cls(internal_url=rt.vst_internal_url, external_url=rt.vst_external_url)

    def build_screenshot_url(
        self,
        *,
        sensor_id: str,
        timestamp: str,
        internal: bool = False,
    ) -> str:
        """Build a screenshot URL. By default uses the external URL (client-facing);
        pass internal=True for in-cluster URLs.

        Today sensor_id and stream_id are treated as interchangeable
        (FIXME at tools/search.py:1638). We pass sensor_id straight through.
        """
        base = self._internal_url if internal else self._external_url
        return build_screenshot_url(base, sensor_id, timestamp)

    async def resolve_stream_id(self, sensor_id: str) -> str:
        """Resolve sensor_id → stream_id via the VST API. Raises VSTError on miss."""
        return await get_stream_id(sensor_id, self._internal_url)

    async def get_timeline(self, sensor_id: str) -> tuple[str, str]:
        """Return (start_iso, end_iso) for a sensor/stream's replay range."""
        # The free helper handles sensor-name → stream_id fallback internally.
        return await get_timeline(sensor_id, self._internal_url)

    async def aclose(self) -> None:
        return None
