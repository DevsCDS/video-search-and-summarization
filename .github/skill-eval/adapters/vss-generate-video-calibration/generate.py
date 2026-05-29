#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill drives AutoMagicCalib (AMC):
deploying the `vss-auto-calibration` microservice + UI, and running
calibration over local MP4s, RTSP streams, or the bundled sample
dataset via the MS REST API. The current spec
([`skills/vss-generate-video-calibration/evals/auto-calibration.json`])
**omits the `profile` field by design** — each query carries its own
context (some ask the agent to deploy AMC, some assume the MS is
already running, some test failure handling). Per
`.github/skill-eval/AGENTS.md` § 2, an absent `profile` is the
supported signal that NO `/vss-deploy-profile` prerequisite should be
prepended; the trial runs directly on a bare Brev instance and the
skill itself handles any deploy via
`references/deploy-auto-calibration-service.md` (pre-authorized in CI
via the PREAMBLE below).

This mirrors the structure of the vss-manage-video-io-storage adapter
(`adapters/vss-manage-video-io-storage/generate.py`): a profile-less,
multi-query spec where each entry in `spec['expects']` becomes one
Harbor task (a `step-<k>/` subdir). The queries are **independent**
scenarios, not an ordered chain — but Harbor still dispatches them as
distinct tasks and the coordinator runs them one at a time per box.

## Platform

The spec declares a single platform — `RTXPRO6000BW` × 1 GPU. AMC's
`vss-auto-calibration` MS needs a GPU (VGGT refinement, calibration
solver); the UI is CPU-only. We honour exactly what the spec's
`resources.platforms` enumerates and do not fan out.

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/auto-calibration/<platform>/
        step-1/ ... step-N/        (one per spec['expects'] entry)
            task.toml
            instruction.md
            tests/test.sh
            tests/generic_judge.py
            tests/auto-calibration.json   (copied from skill)
            solution/solve.sh
            skills/vss-generate-video-calibration/   (full skill copy)
            skills/vss-deploy-profile/               (for agent debug)
            environment/Dockerfile     (FROM scratch; BrevEnvironment takes over)

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir /tmp/skill-eval/.../datasets/vss-generate-video-calibration/auto-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --deploy-skill-dir skills/vss-deploy-profile \\
        --spec skills/vss-generate-video-calibration/evals/auto-calibration.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — mirrors the vss-deploy-profile / vss-manage-video-io-storage adapters
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":          {"short_name": "h100",          "gpu_type": "H100",         "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S":          {"short_name": "l40s",          "gpu_type": "L40S",         "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW":  {"short_name": "rtxpro6000bw",  "gpu_type": "RTX PRO 6000", "min_vram_per_gpu": 96, "brev_search": "RTX PRO"},
    "DGX-SPARK":     {"short_name": "spark",         "gpu_type": "GB10",         "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR":      {"short_name": "thor",          "gpu_type": "Thor",         "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

# The spec enumerates exactly which platforms to run on via
# resources.platforms. Default to that set; do NOT silently fan out
# across every PLATFORMS entry.
DEFAULT_PLATFORM = "RTXPRO6000BW"

# Prepended to every instruction.md so the skill's own HITL bypass
# clause fires. Skills default to "ask the user" before /vss-deploy-profile
# (and before any AMC deploy); in CI there's no user, so without this
# preamble the agent either stalls or falls through to a localhost
# default. This is the verbatim PREAMBLE mandated by AGENTS.md § 3.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)


# ---------------------------------------------------------------------------
# Spec rendering — mirrors adapters/vss-deploy-profile/generate.py so
# `{{platform}}` / `{{repo_root}}` placeholders in the spec's query/env/check
# strings are resolved at generation time. The calibration spec uses
# `{{platform}}` in several query and env strings; leaving it unsubstituted
# would surface the literal token to both the agent (instruction.md) and the
# judge (tests/<spec>.json checks).
#
# In addition, this spec's checks reference the bare shell variable
# `$REPO_ROOT` (e.g. `grep ^VSS_AUTO_CALIBRATION_PORT
# $REPO_ROOT/deploy/docker/industry-profiles/warehouse-operations/.env`).
# `$REPO_ROOT` is NOT forwarded into the judge's environment by
# `envs/brev_env.py` (only PR_HEAD_SHA / PR_REPO / GITHUB_RUN_ID are), so
# when the generic_judge agent runs those check snippets via Bash the
# variable expands to empty and the path resolves to `/deploy/docker/...`
# — a guaranteed false negative on every port/HOST_IP grep check. The
# harness clones the repo to `$HOME/video-search-and-summarization` (see
# brev_env.py: `REPO="$HOME/video-search-and-summarization"`), so we
# rewrite `$REPO_ROOT` / `${REPO_ROOT}` to that portable path during
# rendering — the same destination `{{repo_root}}` resolves to. This keeps
# the spec author's `$REPO_ROOT/...` intent working without requiring the
# harness to export REPO_ROOT.
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
# Match $REPO_ROOT and ${REPO_ROOT} (incl. ${REPO_ROOT:-default}); only the
# bare variable reference is rewritten to the literal portable path so the
# judge's Bash resolves the real on-box clone.
_REPO_ROOT_VAR = re.compile(r"\$\{REPO_ROOT(?::-[^}]*)?\}|\$REPO_ROOT\b")
_LEGACY_REPO = "/home/ubuntu/video-search-and-summarization"
_PORTABLE_REPO = "$HOME/video-search-and-summarization"


def _render_eval_spec(spec: dict, platform: str) -> dict:
    """Substitute `{{platform}}` / `{{repo_root}}` and the bare `$REPO_ROOT`
    shell variable into every string field of the spec.

    This spec is profile-less, but `{{profile}}` is kept in the substitution
    map (resolving to "" if it ever appears) so a future profile-bound
    revision doesn't silently leak the token.

    `{{repo_root}}` and `$REPO_ROOT` both resolve to
    `$HOME/video-search-and-summarization`, a shell-expansion that matches
    whichever default user the Brev provider assigns (Crusoe → ubuntu,
    Massed Compute → shadeform, etc.). The legacy hardcoded
    `/home/ubuntu/...` path is rewritten to the same portable form.
    """
    substitutions = {
        "platform": platform,
        "profile": spec.get("profile", ""),
        "repo_root": _PORTABLE_REPO,
    }

    def _sub(value):
        if isinstance(value, str):
            rendered = _PLACEHOLDER.sub(
                lambda m: str(substitutions.get(m.group(1), m.group(0))),
                value,
            )
            # Rewrite the bare $REPO_ROOT shell var (the spec's checks use
            # it literally) to the on-box clone path the judge's Bash can
            # actually resolve. Do this BEFORE the legacy-path rewrite so
            # both converge on the same portable destination.
            rendered = _REPO_ROOT_VAR.sub(_PORTABLE_REPO, rendered)
            return rendered.replace(_LEGACY_REPO, _PORTABLE_REPO)
        if isinstance(value, list):
            return [_sub(v) for v in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    return _sub(spec)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier for a
    single step's checks. Harbor reads /logs/verifier/reward.txt."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}): delegates to the\n"
        "# generic LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
        "set -uo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "python3 -m pip install --quiet 'anthropic>=0.40.0' >/dev/null 2>&1 || true\n"
        "\n"
        'python3 "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step}\n'
        "exit 0\n"
    )


def generate_solve_script(platform: str) -> str:
    """Gold solution — the verifier drives the AMC probes / API contract
    checks directly, so the oracle just asserts the MS is reachable when
    one is expected and defers. For deploy-only queries the MS may not be
    up at solve time, so a failed probe is non-fatal (exit 0)."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier drives the AMC queries / probes directly — the\n"
        "# solution script only sanity-checks reachability and defers.\n"
        "set -uo pipefail\n"
        "\n"
        "PORT=8010\n"
        'ENV_FILE="$HOME/video-search-and-summarization/deploy/docker/industry-profiles/warehouse-operations/.env"\n'
        'if [ -f "$ENV_FILE" ]; then\n'
        '    P=$(grep ^VSS_AUTO_CALIBRATION_PORT "$ENV_FILE" 2>/dev/null | cut -d= -f2)\n'
        '    [ -n "$P" ] && PORT=$P\n'
        "fi\n"
        'if curl -sf --max-time 10 "http://localhost:${PORT}/v1/ready" >/dev/null 2>&1; then\n'
        '    echo "AMC microservice live on port ${PORT} — verifier will drive the queries."\n'
        "else\n"
        '    echo "AMC MS not (yet) reachable on port ${PORT}; deploy-mode query or"\n'
        '    echo "failure-handling scenario — verifier evaluates against the trajectory."\n'
        "fi\n"
    )


GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


def generate_task(platform: str, spec: dict, output_root: Path,
                  skill_dir: Path, deploy_skill_dir: Path | None) -> None:
    """Emit one Harbor task directory per entry in spec['expects'] — i.e.
    step-<k>/ subdirs under `auto-calibration/<platform>/` per AGENTS.md
    § 4. Single-step specs collapse to a flat `<platform>/`."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    spec_name = Path(spec.get("_source_path", "spec.json")).name or "spec.json"

    # Resolve {{platform}} / {{repo_root}} / $REPO_ROOT for THIS platform
    # before rendering instruction.md or shipping the spec to tests/. The
    # rendered spec is the single source of truth the judge reads.
    rendered_spec = _render_eval_spec(spec, platform)
    expects = rendered_spec.get("expects") or []

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / platform_short
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # instruction.md — ONE query + environment notes ONLY.
        # Never leak the verifier's `checks[]` into the instruction the
        # agent sees — they live in the spec, are copied into tests/, and
        # the verifier evaluates them independently.
        #
        # NOTE: unlike the vss-manage-video-io-storage adapter, the wrapper
        # does NOT assert "the VSS profile is already running" — calibration
        # queries vary: some ask the agent to deploy AMC, some assume the MS
        # is up, some deliberately probe a failure path (VIOS down, NGC key
        # missing). The per-query text carries that context; the wrapper
        # stays neutral and lets the query drive routing.
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host to "
            "handle the request below. Route to the correct mode "
            "(deploy / videos / rtsp / sample-dataset) per the skill's "
            "Input Routing table, probing service readiness before issuing "
            "API calls and standing up AMC yourself when the query requires "
            "it.",
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            expect.get("query", ""),
            "",
            "## Environment notes",
            "",
            rendered_spec.get("env", ""),
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(lines) + "\n")

        # task.toml
        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-generate-video-calibration-auto-calibration-{platform_short}{step_suffix}"',
            f'description = "AMC calibration query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "auto-calibration", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            # ANTHROPIC_MODEL gives the verifier's judge model cascade
            # (JUDGE_MODEL → ANTHROPIC_MODEL → literal) a working fallback
            # when JUDGE_MODEL is unset. Forwarding a literal default for
            # JUDGE_MODEL would bake it in and short-circuit the cascade —
            # the proxy 401s the literal default outright.
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # JUDGE_MAX_TURNS bumped from the generic_judge.py default of
            # 25, matching the vss-manage-video-io-storage adapter rationale:
            # AMC calibration trajectories run long (deploy + project
            # create + multi-file upload + poll-to-COMPLETED + result
            # stats), and several checks must resolve a placeholder
            # (<project_id>, <session_id>) from deep in the trajectory
            # before issuing a live probe. 50 turns gives the per-check
            # judge headroom without changing other skills' defaults.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # `profile` is emitted ONLY when the spec declares one. The
            # current auto-calibration.json omits `profile` by design — the
            # trial runs without a /vss-deploy-profile prerequisite (per
            # AGENTS.md § 2) and the agent stands AMC up itself via the
            # skill's deploy reference. Defaulting to a profile here would
            # resurrect the wrong prerequisite-deploy behaviour silently.
            *([f'profile = "{spec["profile"]}"'] if spec.get("profile") else []),
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            # requires_deployed_vss tracks whether the trial assumes a
            # pre-deployed VSS stack. With the profile-less spec the agent
            # is responsible for the deploy, so this is false.
            f"requires_deployed_vss = {'true' if spec.get('profile') else 'false'}",
            # prerequisite_deploy_mode is alerts-only; emit only if the spec
            # declares it (it does not here).
            *([f'prerequisite_deploy_mode = "{spec["prerequisite_deploy_mode"]}"'] if spec.get("prerequisite_deploy_mode") else []),
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # environment/
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # tests/ — wrapper + generic judge + spec
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Ship the RENDERED spec (placeholders + $REPO_ROOT resolved) —
        # never the raw source — so the judge evaluates checks with
        # {{platform}} substituted and $REPO_ROOT pointing at the real
        # on-box clone. Strip the adapter-internal _source_path key.
        spec_out = {k: v for k, v in rendered_spec.items() if k != "_source_path"}
        (tests_dir / spec_name).write_text(json.dumps(spec_out, indent=2))

        # solution/
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # skills/ — include vss-generate-video-calibration + deploy (so the
        # agent can diagnose / redeploy if the MS isn't live).
        for src, name in ((skill_dir, "vss-generate-video-calibration"),
                          (deploy_skill_dir, "vss-deploy-profile")):
            if src and src.exists():
                dst = step_dir / "skills" / name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True,
                        help="Dataset output root (e.g. .../datasets/vss-generate-video-calibration/auto-calibration)")
    parser.add_argument("--skill-dir", required=True,
                        help="Path to skills/vss-generate-video-calibration")
    parser.add_argument("--deploy-skill-dir", default=None,
                        help="Path to skills/vss-deploy-profile (optional — included for agent debug)")
    parser.add_argument("--spec", default=None,
                        help="Path to auto-calibration.json "
                             "(default: <skill-dir>/evals/auto-calibration.json)")
    parser.add_argument("--platform", default=None,
                        choices=list(PLATFORMS.keys()),
                        help="Generate for this platform only (default: the "
                             "platforms declared in the spec's "
                             "resources.platforms, falling back to "
                             f"{DEFAULT_PLATFORM})")
    parser.add_argument("--all-platforms", action="store_true",
                        help="Fan out across every platform in PLATFORMS — "
                             "the spec does NOT ask for this; honour "
                             "resources.platforms instead.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    deploy_skill_dir = Path(args.deploy_skill_dir) if args.deploy_skill_dir else None
    spec_path = Path(args.spec) if args.spec else (skill_dir / "evals" / "auto-calibration.json")

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)
    spec = json.loads(spec_path.read_text())
    spec["_source_path"] = str(spec_path)

    # Resolve the platform set: explicit flag > spec.resources.platforms >
    # DEFAULT_PLATFORM. Map any spec platform key not in PLATFORMS to an
    # error so a typo'd platform fails loudly rather than silently skipping.
    if args.platform:
        platforms = [args.platform]
    elif args.all_platforms:
        platforms = list(PLATFORMS.keys())
    else:
        spec_platforms = list((spec.get("resources") or {}).get("platforms", {}).keys())
        if spec_platforms:
            unknown = [p for p in spec_platforms if p not in PLATFORMS]
            if unknown:
                print(f"spec declares unknown platform(s): {unknown}; "
                      f"known: {list(PLATFORMS.keys())}", file=sys.stderr)
                sys.exit(1)
            platforms = spec_platforms
        else:
            platforms = [DEFAULT_PLATFORM]

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  platforms    : {platforms}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()
    for platform in platforms:
        task_id = PLATFORMS[platform]["short_name"]
        print(f"  GEN  vss-generate-video-calibration/auto-calibration/{task_id}")
        generate_task(platform, spec, output_root, skill_dir, deploy_skill_dir)
    print()
    print(f"Generated {len(platforms)} platform(s) under {output_root}/")
    print()
    print("Note: this spec OMITS `profile`. The trial runs on a bare Brev")
    print("instance — no /vss-deploy-profile prerequisite is injected. Each")
    print("query carries its own context; deploy-mode queries expect the")
    print("agent to stand AMC up via references/deploy-auto-calibration-service.md.")


if __name__ == "__main__":
    main()
