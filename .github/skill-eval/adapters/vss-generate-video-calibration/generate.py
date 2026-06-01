#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill exercises the AutoMagicCalib (AMC)
microservice — deploying it, then running calibration over local MP4s,
RTSP streams, or the bundled sample dataset. The current spec
([`skills/vss-generate-video-calibration/evals/auto-calibration.json`])
**omits the `profile` field by design** — the agent is expected to
deploy AMC standalone via the skill's bundled
`references/deploy-auto-calibration-service.md` runbook before
exercising the calibration API. Per `.github/skill-eval/AGENTS.md` § 2,
an absent `profile` is the supported signal to the harness that no
`/vss-deploy-profile` prerequisite should be prepended; the trial runs
directly on a bare Brev instance.

The spec declares a single platform: RTXPRO6000BW (1 GPU). The adapter
respects `resources.platforms` from the spec and generates one task
directory per platform × step.

## Directory layout

    $OUTPUT_DIR/<spec_stem>/<platform_short>/step-<N>/
        task.toml
        instruction.md
        tests/test.sh
        tests/generic_judge.py
        tests/auto-calibration.json         (spec copy)
        solution/solve.sh
        skills/vss-generate-video-calibration/   (full skill copy)
        skills/vss-deploy-profile/               (optional; for agent debug)
        environment/Dockerfile

One step-dir per query in `expects[]`. All share the same verifier
(generic LLM-as-judge). Multi-step dispatch is serialized by the
skills-eval agent per AGENTS.md § Harbor invocation.

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir /tmp/skill-eval/<run>/datasets/vss-generate-video-calibration/auto-calibration \\
        --skill-dir skills/vss-generate-video-calibration \\
        --deploy-skill-dir skills/vss-deploy-profile
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — keyed by the canonical platform name from the spec's
# resources.platforms. The short_name is used in task dirs and
# harbor --include-task-name globs.
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":          {"short_name": "h100",          "gpu_type": "H100",              "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S":          {"short_name": "l40s",          "gpu_type": "L40S",              "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW":  {"short_name": "rtxpro6000bw",  "gpu_type": "RTX PRO 6000",      "min_vram_per_gpu": 96, "brev_search": "RTX PRO"},
    "DGX-SPARK":     {"short_name": "spark",         "gpu_type": "GB10",              "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR":      {"short_name": "thor",          "gpu_type": "Thor",              "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

# Prepended to every instruction.md so the skill's own HITL bypass
# clause fires. Skills default to "ask the user" before /vss-deploy-profile; in CI
# there's no user, so without this preamble the agent either stalls or
# falls through to a localhost default.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier for a
    single step's checks. Harbor reads /logs/verifier/reward.txt."""
    return (
        "#!/bin/bash\n"
        f"# vss-generate-video-calibration verifier (step {step}): delegates to the generic\n"
        "# LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
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
    """Gold solution — assumes AMC is already deployed; the oracle just
    re-runs the verifier (there's no separate 'solve' action for a
    probe-style task since the agent's job is driving the API, which
    the verifier does independently)."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier drives the AMC queries directly — the solution\n"
        "# script simply asserts AMC is live, then defers to the verifier.\n"
        "set -euo pipefail\n"
        "\n"
        "AMC_PORT=${VSS_AUTO_CALIBRATION_PORT:-8010}\n"
        'curl -sf --connect-timeout 5 '
        '"http://localhost:${AMC_PORT}/v1/ready" '
        ">/dev/null || {\n"
        "    echo 'AMC is not deployed — cannot solve vss-generate-video-calibration task'\n"
        "    exit 1\n"
        "}\n"
        "echo 'AMC is live — verifier will drive the queries.'\n"
    )


GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


def generate_task(platform: str, spec: dict, output_root: Path,
                  skill_dir: Path, deploy_skill_dir: Path | None) -> None:
    """Emit one Harbor task directory per entry in spec['expects'] — i.e.
    step-<k>/ subdirs under `<spec_stem>/<platform>/` per AGENTS.md § 4.
    Single-step specs collapse to a flat `<spec_stem>/<platform>/`."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = Path(spec.get("_source_path", "auto-calibration.json")).name or "auto-calibration.json"

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / platform_short
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # instruction.md — ONE step's query + environment notes ONLY.
        # Never leak the verifier's `checks[]` into the instruction the agent
        # sees — they live in the spec, are copied into tests/, and the
        # verifier evaluates them independently.
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host. "
            "The skill handles deployment of the AMC microservice "
            "(`references/deploy-auto-calibration-service.md`) and calibration "
            "(`references/videos.md`, `references/rtsp.md`, `references/sample-dataset.md`).",
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            expect.get("query", ""),
            "",
            "## Environment notes",
            "",
            spec.get("env", ""),
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
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # AMC workflows are multi-step (deploy+calibrate) and checks may
            # need to resolve project_ids from deep in the trajectory.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # `profile` is emitted ONLY when the spec declares one.
            # The current auto-calibration.json omits `profile` by design —
            # the trial runs without a /vss-deploy-profile prerequisite and
            # the agent stands AMC up standalone via the skill's deploy reference.
            *([f'profile = "{spec["profile"]}"'] if spec.get("profile") else []),
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            f"requires_deployed_vss = {'true' if spec.get('profile') else 'false'}",
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
        spec_src = skill_dir / "evals" / spec_name
        if spec_src.exists():
            shutil.copy(spec_src, tests_dir / spec_name)
        else:
            # write a copy of the spec even if the source file path differs
            (tests_dir / "auto-calibration.json").write_text(
                json.dumps(spec, indent=2)
            )

        # solution/
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # skills/ — include vss-generate-video-calibration + deploy (so agent
        # can diagnose if VSS isn't live).
        for src, name in ((skill_dir, "vss-generate-video-calibration"), (deploy_skill_dir, "vss-deploy-profile")):
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
                        help="Dataset output root (e.g. $SCRATCH/datasets/vss-generate-video-calibration/auto-calibration)")
    parser.add_argument("--skill-dir", required=True,
                        help="Path to skills/vss-generate-video-calibration")
    parser.add_argument("--deploy-skill-dir", default=None,
                        help="Path to skills/vss-deploy-profile (optional — included for agent debug)")
    parser.add_argument("--spec", default=None,
                        help="Path to auto-calibration.json "
                             "(default: <skill-dir>/evals/auto-calibration.json)")
    parser.add_argument("--platform", default=None,
                        choices=list(PLATFORMS.keys()),
                        help="Generate for this platform only "
                             "(default: uses platforms from spec)")
    parser.add_argument("--all-platforms", action="store_true",
                        help="Fan out across every platform in PLATFORMS")
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

    # Determine platforms from the spec's resources.platforms unless overridden.
    if args.platform:
        platforms = [args.platform]
    elif args.all_platforms:
        platforms = list(PLATFORMS.keys())
    else:
        spec_platforms = list((spec.get("resources") or {}).get("platforms", {}).keys())
        platforms = [p for p in spec_platforms if p in PLATFORMS] or ["RTXPRO6000BW"]

    print(f"=== Inputs ===")
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
    if spec.get("profile"):
        print("Note: this spec declares a `profile` — the coordinator will inject")
        print("a matching /vss-deploy-profile task ahead of each trial.")
    else:
        print("Note: this spec OMITS `profile`. The trial runs on a bare Brev")
        print("instance — no /vss-deploy-profile prerequisite is injected. The agent is")
        print("expected to deploy AMC standalone via the skill's bundled")
        print("references/deploy-auto-calibration-service.md runbook before")
        print("exercising the calibration API.")


if __name__ == "__main__":
    main()
