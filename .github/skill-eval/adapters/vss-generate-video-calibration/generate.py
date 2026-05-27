#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-generate-video-calibration skill.

The vss-generate-video-calibration skill drives the AutoMagicCalib (AMC) microservice
across four workflows: deploy, calibrate-from-videos, calibrate-from-RTSP, and
calibrate-from-sample-dataset.  The current spec
([``skills/vss-generate-video-calibration/eval/auto-calibration.json``]) **omits
the ``profile`` field by design** — AMC is a standalone Docker service, not a VSS
profile stack.  The agent deploys AMC via the skill's
``references/deploy-auto-calibration-service.md`` runbook (part of the spec
checks); no ``/vss-deploy-profile`` prerequisite is injected.

The spec runs on RTXPRO6000BW (the single pool member whose Docker daemon is
pre-configured with NGC credentials and the ``nvcr.io/nvstaging`` registry
mirror that pulls the AMC containers).

## Harbor chaining / dependencies

Each of the 11 expects is an independent trial: no state from step N is
assumed in step N+1 (the AMC service is presumed to be either already
running or freshly deployed within the trial, as appropriate).  The multi-step
dispatch loop (see AGENTS.md § Harbor invocation) serialises the steps so a
prior-fail is surfaced before unnecessary steps run.

## Directory layout

    .github/skill-eval/datasets/vss-generate-video-calibration/auto-calibration/<platform>/
        step-1/
            task.toml, instruction.md, tests/, solution/, skills/, environment/
        step-2/
            ...
        ...
        step-11/
            ...

One task per (step × platform).  With a single platform (RTXPRO6000BW) and
11 steps this produces 11 task directories.

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-generate-video-calibration/generate.py \\
        --output-dir .github/skill-eval/datasets/vss-generate-video-calibration \\
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
# Platform table — mirrors the other adapters.
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100":          {"short_name": "h100",          "gpu_type": "H100",         "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S":          {"short_name": "l40s",          "gpu_type": "L40S",         "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW":  {"short_name": "rtxpro6000bw",  "gpu_type": "RTX PRO 6000", "min_vram_per_gpu": 96, "brev_search": "RTX PRO"},
    "DGX-SPARK":     {"short_name": "spark",         "gpu_type": "GB10",         "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR":      {"short_name": "thor",          "gpu_type": "Thor",         "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

DEFAULT_PLATFORM = "RTXPRO6000BW"

# Prepended to every instruction.md so the skill's own HITL bypass clause fires.
# Skills default to "ask the user" before running autonomous operations; in CI
# there is no user, so without this preamble the agent stalls or falls through
# to a localhost default.
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper that invokes the generic LLM-as-judge verifier for
    a single step's checks.  Harbor reads /logs/verifier/reward.txt."""
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
    """Gold solution stub — AMC calibration tasks are oracle-style:
    the verifier probes the live service.  The solve script verifies
    that AMC is running (and starts it if not), then defers to the
    verifier."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-generate-video-calibration on {platform}\n"
        "# The verifier drives the AMC API queries directly.  This script\n"
        "# ensures the AMC containers are running before the verifier fires.\n"
        "set -euo pipefail\n"
        "\n"
        'REPO="${REPO_ROOT:-$HOME/video-search-and-summarization}"\n'
        "\n"
        "# Check if AMC is already running\n"
        "if docker ps --format '{{.Names}}' | grep -qx vss-auto-calibration; then\n"
        "    echo 'AMC already running — proceeding to verify.'\n"
        "    exit 0\n"
        "fi\n"
        "\n"
        "echo 'AMC not running — use the skill to deploy it.'\n"
        "exit 1\n"
    )


def _platforms_from_spec(spec: dict) -> list[str]:
    """Extract the platform list from spec.resources.platforms."""
    plats = (spec.get("resources") or {}).get("platforms") or {}
    return [p for p in plats if p in PLATFORMS] or [DEFAULT_PLATFORM]


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_task(
    platform: str,
    spec: dict,
    output_root: Path,
    skill_dir: Path,
    deploy_skill_dir: Path | None,
) -> None:
    """Emit one Harbor task directory per entry in spec['expects'] — i.e.
    step-<k>/ subdirs under ``auto-calibration/<platform>/``."""
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = Path(spec.get("_source_path", "spec.json")).name or "spec.json"
    spec_stem = spec_name.removesuffix(".json")   # e.g. "auto-calibration"

    for idx, expect in enumerate(expects, 1):
        # Multi-step: always use step subdirs (11 steps here)
        step_dir = output_root / spec_stem / platform_short / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # -- instruction.md --
        # ONE step's query + environment notes.  Checks are intentionally
        # NOT included — the verifier evaluates them independently from the
        # spec JSON in tests/.
        step_suffix = f"-step-{idx}"
        lines = [
            PREAMBLE,
            "",
            f"Use the `/vss-generate-video-calibration` skill on this `{platform}` host.",
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

        # -- task.toml --
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-generate-video-calibration-{spec_stem}-{platform_short}{step_suffix}"',
            f'description = "AMC calibration query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-generate-video-calibration", "amc", "{spec_stem}", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            # ANTHROPIC_MODEL gives the verifier's judge model cascade
            # (JUDGE_MODEL → ANTHROPIC_MODEL → literal) a working fallback
            # when JUDGE_MODEL is unset.  Forwarding a literal default for
            # JUDGE_MODEL would bake it in and short-circuit the cascade.
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # JUDGE_MAX_TURNS bumped from the generic_judge.py default of 25
            # because AMC calibration step trajectories can be long
            # (deploy + calibrate + poll + evaluate), and several checks
            # need to resolve API-call context before issuing live curl probes.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-generate-video-calibration"',
            # `profile` is intentionally absent — spec omits it by design.
            # The trial runs on a bare Brev instance; no /vss-deploy-profile
            # prerequisite is injected.  Emitting profile here would cause
            # BrevEnvironment._ensure_prerequisite_deployed to fire an
            # unwanted sub-claude /vss-deploy-profile before the trial agent.
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            # AMC runs as standalone Docker containers — no VSS profile required.
            "requires_deployed_vss = false",
            # prerequisite_deploy_mode is alerts-only and does not apply here.
            # (no spec.get("prerequisite_deploy_mode") — omit the field entirely)
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # -- environment/ placeholder (BrevEnvironment takes over) --
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # -- tests/: wrapper + generic judge + spec copy --
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        spec_src = skill_dir / "eval" / spec_name
        if spec_src.exists():
            shutil.copy(spec_src, tests_dir / spec_name)
        else:
            # Fallback: write in-memory spec so tests/ is always complete
            (tests_dir / spec_name).write_text(json.dumps(spec, indent=2))

        # -- solution/ --
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform))

        # -- skills/ — vss-generate-video-calibration + vss-deploy-profile (for diagnosis) --
        copies = [
            (skill_dir,        "vss-generate-video-calibration"),
            (deploy_skill_dir, "vss-deploy-profile"),
        ]
        for src, name in copies:
            if src and src.exists():
                dst = step_dir / "skills" / name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Dataset output root "
             "(e.g. .github/skill-eval/datasets/vss-generate-video-calibration)",
    )
    parser.add_argument(
        "--skill-dir", required=True,
        help="Path to skills/vss-generate-video-calibration",
    )
    parser.add_argument(
        "--deploy-skill-dir", default=None,
        help="Path to skills/vss-deploy-profile (optional — included for agent diagnosis)",
    )
    parser.add_argument(
        "--spec", default=None,
        help="Path to spec JSON "
             "(default: <skill-dir>/eval/auto-calibration.json)",
    )
    parser.add_argument(
        "--platform", default=None,
        choices=list(PLATFORMS.keys()),
        help=f"Generate for one platform only "
             f"(default: {DEFAULT_PLATFORM}; overrides spec.resources.platforms)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    deploy_skill_dir = Path(args.deploy_skill_dir) if args.deploy_skill_dir else None
    spec_path = (
        Path(args.spec)
        if args.spec
        else (skill_dir / "eval" / "auto-calibration.json")
    )

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)
    spec = json.loads(spec_path.read_text())
    spec["_source_path"] = str(spec_path)

    platforms = [args.platform] if args.platform else _platforms_from_spec(spec)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  platforms    : {platforms}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()

    spec_stem = spec_path.stem  # "auto-calibration"
    for platform in platforms:
        task_id = PLATFORMS[platform]["short_name"]
        print(f"  GEN  vss-generate-video-calibration/{spec_stem}/{task_id} "
              f"({len(spec.get('expects', []))} steps)")
        generate_task(platform, spec, output_root, skill_dir, deploy_skill_dir)

    print()
    print(f"Generated {len(platforms)} platform(s) under {output_root}/{spec_stem}/")
    print()
    print("Note: this spec omits `profile` — the trial runs on a bare Brev instance.")
    print("AMC is a standalone Docker service; no /vss-deploy-profile prerequisite is")
    print("injected.  The agent deploys AMC itself via the skill's")
    print("references/deploy-auto-calibration-service.md runbook.")


if __name__ == "__main__":
    main()
