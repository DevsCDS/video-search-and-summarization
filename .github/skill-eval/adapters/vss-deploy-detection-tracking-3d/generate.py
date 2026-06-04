#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-deploy-detection-tracking-3d skill.

The vss-deploy-detection-tracking-3d skill deploys and operates the RTVI-CV-3D
(MV3DT) stack — per-camera DeepStream perception plus BEV Fusion over multiple
calibrated cameras. It has four eval specs:

  - deploy.json        — deploy/verify/teardown multi-step chain (3 steps)
  - calibration-chain.json — end-to-end custom-data calibration chain (2 steps)
  - routing.json       — CPU-only routing-coverage (4 queries, single step)
  - evals.json         — informational routing Q&A (6 Q&A items, no deploy)

Specs declare their platforms in `resources.platforms`. The adapter reads the
spec's platform matrix and generates one task directory per (spec, platform).
Multi-query specs produce ordered step-N/ subdirs.

This adapter does NOT use `/vss-deploy-profile` — the skill drives the
warehouse-operations compose tree directly via `deploy/docker/compose.yml`
with `--env-file industry-profiles/warehouse-operations/.env`. The spec's
first query contains the full environment/prerequisite description for the
trial agent.

Directory layout:
    <output-dir>/<spec_stem>/<platform_short>/          (single-step)
    <output-dir>/<spec_stem>/<platform_short>/step-N/   (multi-step)
        instruction.md, task.toml, tests/, solution/, skills/, environment/

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-deploy-detection-tracking-3d/generate.py \\
        --output-dir /tmp/skill-eval/datasets/<leg-slug>/<run_id> \\
        --skill-dir skills/vss-deploy-detection-tracking-3d \\
        --spec skills/vss-deploy-detection-tracking-3d/evals/deploy.json \\
        --platform RTXPRO6000BW
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — mirrors the fleet topology
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100": {
        "short_name": "h100",
        "gpu_type": "H100",
        "min_vram_per_gpu": 80,
        "brev_search": "H100",
    },
    "L40S": {
        "short_name": "l40s",
        "gpu_type": "L40S",
        "min_vram_per_gpu": 48,
        "brev_search": "L40S",
    },
    "RTXPRO6000BW": {
        "short_name": "rtxpro6000bw",
        "gpu_type": "RTX PRO 6000",
        "min_vram_per_gpu": 96,
        "brev_search": "RTX PRO",
    },
    "DGX-SPARK": {
        "short_name": "spark",
        "gpu_type": "GB10",
        "min_vram_per_gpu": 96,
        "brev_search": "GB10",
    },
    "IGX-THOR": {
        "short_name": "thor",
        "gpu_type": "Thor",
        "min_vram_per_gpu": 64,
        "brev_search": "Thor",
    },
}

# Prepended to every instruction.md so the skill's own HITL bypass
# clause fires. Skills default to "ask the user" before deploy actions;
# in CI there's no user, so without this preamble the agent stalls.
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
    """Shell wrapper that invokes the generic LLM-as-judge verifier for a
    single step's checks."""
    return (
        "#!/bin/bash\n"
        f"# vss-deploy-detection-tracking-3d verifier (step {step}): delegates to the generic\n"
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


def generate_solve_script(platform: str, step: int, total_steps: int) -> str:
    """Gold solution placeholder — the verifier drives checks independently."""
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-deploy-detection-tracking-3d on {platform} (step {step}/{total_steps})\n"
        "# The verifier drives the checks directly via LLM-as-judge.\n"
        "set -euo pipefail\n"
        "\n"
        "echo 'Solution placeholder — verifier evaluates the agent trajectory.'\n"
    )


def _render_spec(spec: dict, platform: str) -> dict:
    """Substitute {{platform}} placeholders in the spec."""
    import re as _re
    pattern = _re.compile(r"\{\{\s*(\w+)\s*\}\}")
    substitutions = {
        "platform": platform,
        "repo_root": "$HOME/video-search-and-summarization",
    }

    def _sub(value):
        if isinstance(value, str):
            return pattern.sub(
                lambda m: str(substitutions.get(m.group(1), m.group(0))),
                value,
            )
        if isinstance(value, list):
            return [_sub(v) for v in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    return _sub(spec)


def generate_task(
    platform: str,
    spec: dict,
    spec_stem: str,
    output_root: Path,
    skill_dir: Path,
    calibration_skill_dir: Path | None,
    deploy_skill_dir: Path | None,
) -> None:
    """Emit Harbor task directories for each step in the spec's expects list.

    Single-step specs produce a flat <spec_stem>/<platform_short>/ dir.
    Multi-step specs produce <spec_stem>/<platform_short>/step-N/ subdirs.
    """
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = f"{spec_stem}.json"

    # Determine gpu_count from the spec's resources.platforms
    resources = (spec.get("resources") or {}).get("platforms") or {}
    platform_res = resources.get(platform) or {}
    gpu_count = platform_res.get("gpu_count", 1)

    for idx, expect in enumerate(expects, 1):
        step_dir = output_root / spec_stem / platform_short
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # -- instruction.md --
        # Contains the preamble + the query for this step. Never leaks checks.
        # Substitute {{platform}} placeholders in the query text.
        query_text = expect.get("query", "")
        query_text = query_text.replace("{{platform}}", platform)
        lines = [
            PREAMBLE,
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            query_text,
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(lines) + "\n")

        # -- task.toml --
        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-deploy-detection-tracking-3d-{spec_stem}-{platform_short}{step_suffix}"',
            f'description = "MV3DT {spec_stem} query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-deploy-detection-tracking-3d", "{spec_stem}", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # MV3DT deploy specs can run long trajectories; bump judge turns.
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            f'skill = "vss-deploy-detection-tracking-3d"',
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'gpu_count = {gpu_count}',
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            f'brev_search = "{pspec["brev_search"]}"',
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # -- environment/ --
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # -- tests/ --
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        # Write the rendered spec (with platform substitutions applied)
        rendered_spec = _render_spec(spec, platform)
        (tests_dir / spec_name).write_text(json.dumps(rendered_spec, indent=2))

        # -- solution/ --
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(
            generate_solve_script(platform, idx, len(expects))
        )

        # -- skills/ --
        # Include the primary skill + optional chained skills for agent access
        skills_to_copy = [
            (skill_dir, "vss-deploy-detection-tracking-3d"),
        ]
        if calibration_skill_dir and calibration_skill_dir.exists():
            skills_to_copy.append(
                (calibration_skill_dir, "vss-generate-video-calibration")
            )
        if deploy_skill_dir and deploy_skill_dir.exists():
            skills_to_copy.append(
                (deploy_skill_dir, "vss-deploy-profile")
            )
        for src, name in skills_to_copy:
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
        help="Dataset output root",
    )
    parser.add_argument(
        "--skill-dir", required=True,
        help="Path to skills/vss-deploy-detection-tracking-3d",
    )
    parser.add_argument(
        "--calibration-skill-dir", default=None,
        help="Path to skills/vss-generate-video-calibration (optional — "
             "included for calibration-chain spec)",
    )
    parser.add_argument(
        "--deploy-skill-dir", default=None,
        help="Path to skills/vss-deploy-profile (optional — included for "
             "agent routing context)",
    )
    parser.add_argument(
        "--spec", default=None,
        help="Path to a specific eval spec JSON file "
             "(default: generate for ALL specs in <skill-dir>/evals/)",
    )
    parser.add_argument(
        "--platform", default=None,
        choices=list(PLATFORMS.keys()),
        help="Generate for this platform only (default: all platforms "
             "declared in the spec's resources.platforms)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    calibration_skill_dir = (
        Path(args.calibration_skill_dir) if args.calibration_skill_dir else None
    )
    deploy_skill_dir = (
        Path(args.deploy_skill_dir) if args.deploy_skill_dir else None
    )

    # Discover specs
    if args.spec:
        spec_paths = [Path(args.spec)]
    else:
        evals_dir = skill_dir / "evals"
        if not evals_dir.exists():
            evals_dir = skill_dir / "eval"
        if not evals_dir.exists():
            print(f"No evals/ directory found under {skill_dir}", file=sys.stderr)
            sys.exit(1)
        spec_paths = sorted(evals_dir.glob("*.json"))

    if not spec_paths:
        print("No eval spec JSON files found.", file=sys.stderr)
        sys.exit(1)

    print("=== Inputs ===")
    print(f"  output_dir            : {output_root}")
    print(f"  skill_dir             : {skill_dir}")
    print(f"  calibration_skill_dir : {calibration_skill_dir or '(none)'}")
    print(f"  deploy_skill_dir      : {deploy_skill_dir or '(none)'}")
    print(f"  spec(s)               : {[p.name for p in spec_paths]}")
    print(f"  platform filter       : {args.platform or '(all from spec)'}")
    print()

    total_generated = 0
    total_skipped = 0

    for spec_path in spec_paths:
        spec_stem = spec_path.stem
        if not spec_path.exists():
            print(f"  SKIP {spec_stem}: file not found at {spec_path}")
            total_skipped += 1
            continue

        raw = json.loads(spec_path.read_text())
        # Skip non-dict specs (e.g. evals.json is a list of Q&A items,
        # not an evaluable spec with resources.platforms / expects[])
        if not isinstance(raw, dict):
            print(f"  SKIP {spec_stem}: not a dict-type eval spec (got {type(raw).__name__})")
            total_skipped += 1
            continue
        spec = raw
        spec["_source_path"] = str(spec_path)

        # Determine platforms from the spec
        resources = (spec.get("resources") or {}).get("platforms")
        if not isinstance(resources, dict) or not resources:
            print(f"  SKIP {spec_stem}: no resources.platforms declared")
            total_skipped += 1
            continue

        expects = spec.get("expects") or []
        if not expects:
            print(f"  SKIP {spec_stem}: no expects[] entries")
            total_skipped += 1
            continue

        platforms_to_gen = []
        for p in resources.keys():
            if args.platform and p != args.platform:
                continue
            if p not in PLATFORMS:
                print(f"  SKIP {spec_stem}/{p}: unknown platform")
                total_skipped += 1
                continue
            platforms_to_gen.append(p)

        for platform in platforms_to_gen:
            task_id = PLATFORMS[platform]["short_name"]
            n_steps = len(expects)
            n_checks = sum(len(q.get("checks") or []) for q in expects)
            print(f"  GEN  {spec_stem}/{task_id}  "
                  f"steps={n_steps}  checks={n_checks}")
            generate_task(
                platform, spec, spec_stem, output_root,
                skill_dir, calibration_skill_dir, deploy_skill_dir,
            )
            total_generated += 1

    print()
    print(f"Summary: {total_generated} task(s) generated, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
