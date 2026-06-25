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
"""vss-cli — entrypoint for the exec transport.

PHASE 1 STATUS: Dispatcher + arg parsing done; primitive calls raise
NotImplementedError until their respective phases land. Wired into
pyproject.toml's [project.scripts] in Phase 7.

Invocation contract (DESIGN.md §10):

    vss-cli <primitive> [--config <path>] [--json '<payload>'] [--stream]
       <primitive> ∈ embed_search | attribute_search | search
       --config:  path to a NAT-style config file
                  (default: $VSS_AGENT_CONFIG_FILE, set by the pod env).
                  Required for `search` parity with the deployed profile.
       --json:    payload as a JSON object matching the input model.
       --stream:  only valid for `search`; emits SearchEvent JSON lines.
       stdin:     alternative payload source; mutually exclusive with --json.

    Query decomposition is NAT-owned. `vss-cli search` accepts only
    agent_mode=false payloads; call the NAT search/search_agent functions for
    agent-mode decomposition.

    Exit codes:
       0   success (one final output produced)
       1   any other unexpected error
       2   invalid input / Pydantic validation error
       3   backend unreachable
       4   configuration error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from .errors import BackendUnreachableError
from .errors import ConfigurationError
from .errors import InvalidInputError
from .host import VSSSearch

logger = logging.getLogger(__name__)

PRIMITIVES = ("embed_search", "attribute_search", "search")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vss-cli",
        description="Invoke VSS search primitives directly (exec-transport entrypoint).",
    )
    p.add_argument("primitive", choices=PRIMITIVES, help="Which primitive to invoke.")
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a NAT-style config file. Defaults to $VSS_AGENT_CONFIG_FILE. "
            "Required for `search` parity with the deployed profile."
        ),
    )
    p.add_argument(
        "--json",
        dest="json_payload",
        default=None,
        help="Payload as a JSON object matching the primitive's input model.",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        help="Emit SearchEvent JSON lines instead of a single output JSON. Only valid for `search`.",
    )
    return p.parse_args(argv)


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Read payload from --json or stdin; mutually exclusive."""
    if args.json_payload is not None:
        if not sys.stdin.isatty() and sys.stdin.readable() and not sys.stdin.closed:
            # Best-effort detection of "stdin was redirected" without blocking.
            # We can't reliably tell at this point, so the test below favors --json.
            pass
        try:
            parsed: dict[str, Any] = json.loads(args.json_payload)
            return parsed
        except json.JSONDecodeError as e:
            raise InvalidInputError(f"--json is not valid JSON: {e}") from e

    if not sys.stdin.isatty():
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            return parsed
        except json.JSONDecodeError as e:
            raise InvalidInputError(f"stdin is not valid JSON: {e}") from e

    return {}


def _build_facade(args: argparse.Namespace) -> VSSSearch:
    """Pick the right builder based on --config / $VSS_AGENT_CONFIG_FILE.

    VSSSearch.from_config_file pulls SearchRuntime AND SearchOptions
    (use_attribute_search) from the same config — guaranteeing parity with
    the deployed profile.

    Error semantics:
      - --config explicitly given but path missing  →  ConfigurationError
        (don't silently fall through to env — user asked for a specific file)
      - --config not given AND $VSS_AGENT_CONFIG_FILE not set  →  fall back to
        env-only; warn for `search` since profile-level settings can't be
        recovered from env alone
      - --config not given but $VSS_AGENT_CONFIG_FILE set and file exists  →
        use the env-pointed config (the deployed-pod common case)
    """
    if args.config is not None:
        # Explicit path: must exist.
        if not Path(args.config).exists():
            raise ConfigurationError(f"--config path does not exist: {args.config!r}")
        return VSSSearch.from_config_file(args.config)

    env_path = os.environ.get("VSS_AGENT_CONFIG_FILE")
    if env_path and Path(env_path).exists():
        return VSSSearch.from_config_file(env_path)

    if args.primitive == "search":
        logger.warning(
            "vss-cli: VSS_AGENT_CONFIG_FILE not set and --config not given. "
            "Search behavior may diverge from the deployed profile "
            "(use_attribute_search, fusion weights, embed_confidence_threshold all fall back to defaults)."
        )
    return VSSSearch.from_env()


async def _run(args: argparse.Namespace) -> int:
    if args.stream and args.primitive != "search":
        raise InvalidInputError("--stream is only valid for the `search` primitive")

    payload = _load_payload(args)
    if args.primitive == "search" and payload.get("agent_mode") is True:
        raise InvalidInputError(
            "vss-cli search does not perform NAT query decomposition; set agent_mode=false "
            "or call the NAT search/search_agent function"
        )

    async with _build_facade(args) as vss:
        if args.primitive == "search" and args.stream:
            async for event in vss.search_stream(**payload):
                sys.stdout.write(event.model_dump_json() + "\n")
                sys.stdout.flush()
            return 0

        out = await getattr(vss, args.primitive)(**payload)
        sys.stdout.write(out.model_dump_json() + "\n")
        sys.stdout.flush()
        return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint installed via pyproject.toml's [project.scripts]."""
    logging.basicConfig(level=os.environ.get("VSS_CLI_LOG_LEVEL", "WARNING"))
    try:
        args = _parse_args(argv)
        return asyncio.run(_run(args))
    except InvalidInputError as e:
        sys.stderr.write(f"[vss-cli] invalid input: {e}\n")
        return 2
    except BackendUnreachableError as e:
        sys.stderr.write(f"[vss-cli] backend unreachable: {e}\n")
        return 3
    except ConfigurationError as e:
        sys.stderr.write(f"[vss-cli] configuration error: {e}\n")
        return 4
    except NotImplementedError as e:
        sys.stderr.write(f"[vss-cli] not yet implemented: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"[vss-cli] unexpected error: {e!r}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
