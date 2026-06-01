#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-deploy-detection-tracking-3d skill.

The vss-deploy-detection-tracking-3d skill deploys and manages the
Multi-View 3D Tracking (MV3DT / RTVI-CV-3D) stack — a compose-based
deployment of `vss-rtvi-cv-mv3dt`, `vss-rtvi-cv-bev-fusion`, Kafka/Redis,
mosquitto, NVStreamer, and optionally Elasticsearch for the extended
profile. It does NOT use `/vss-deploy-profile`; it drives
`deploy/docker/compose.yml` with `--env-file
industry-profiles/warehouse-operations/.env` gated on the
`bp_wh_kafka_mv3dt` compose profile.

Three eval specs ship with the skill:

  - evals/deploy.json          — Deploy/verify/teardown flow (3 steps,
                                 multi-step chain, GPU required)
  - evals/routing.json         — Routing-coverage eval: informational
                                 queries only (4 steps, CPU-only,
                                 gpu_count=0, no containers deployed)
  - evals/calibration-chain.json — End-to-end custom-data: AMC chain →
                                   MV3DT deploy → verify → teardown
                                   (2 steps, GPU required)

Each spec is multi-step: queries run in order with state preserved
between steps (skip-on-prior-fail). The adapter emits `step-<N>/`
subdirs so the harness dispatch loop drives them sequentially.

Usage from the repository root:
    # Generate dataset for a specific spec
    python3 .github/skill-eval/adapters/vss-deploy-detection-tracking-3d/generate.py \\
        --output-dir /tmp/skill-eval/datasets/<leg>/<run_id> \\
        --skill-dir skills/vss-deploy-detection-tracking-3d \\
        --spec skills/vss-deploy-detection-tracking-3d/evals/deploy.json

    # Generate for all specs
    python3 .github/skill-eval/adapters/vss-deploy-detection-tracking-3d/generate.py \\
        --output-dir /tmp/skill-eval/datasets/<leg>/<run_id> \\
        --skill-dir skills/vss-deploy-detection-tracking-3d \\
        --all-specs
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platforms — same registry as other adapters
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, dict] = {
    "H100": {"short_name": "h100", "gpu_type": "H100", "min_vram_per_gpu": 80, "brev_search": "H100"},
    "L40S": {"short_name": "l40s", "gpu_type": "L40S", "min_vram_per_gpu": 48, "brev_search": "L40S"},
    "RTXPRO6000BW": {
        "short_name": "rtxpro6000bw",
        "gpu_type": "RTX PRO 6000",
        "min_vram_per_gpu": 96,
        "brev_search": "RTX PRO",
    },
    "DGX-SPARK": {"short_name": "spark", "gpu_type": "GB10", "min_vram_per_gpu": 96, "brev_search": "GB10"},
    "IGX-THOR": {"short_name": "thor", "gpu_type": "Thor", "min_vram_per_gpu": 64, "brev_search": "Thor"},
}

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"

PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def _substitute_spec(spec: dict, platform: str) -> dict:
    """Substitute ``{{platform}}`` and ``{{repo_root}}`` in every string."""
    substitutions = {
        "platform": platform,
        "repo_root": "$HOME/video-search-and-summarization",
    }
    pattern = re.compile(r"\{\{\s*(\w+)\s*\}\}")

    def _sub(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: str(substitutions.get(m.group(1), m.group(0))), value)
        if isinstance(value, list):
            return [_sub(v) for v in value]
        if isinstance(value, dict):
            return {k: _sub(v) for k, v in value.items()}
        return value

    return _sub(spec)


def _spec_stem(spec_path: Path) -> str:
    """Extract the stem used for dataset grouping (e.g. 'deploy', 'routing')."""
    return spec_path.stem


def _platform_gpu_from_spec(spec: dict, platform_filter: str | None) -> list[tuple[str, int]]:
    """Return [(platform, gpu_count)] from the spec's resources.platforms."""
    declared = (spec.get("resources") or {}).get("platforms") or {}
    if not declared:
        return [(platform_filter or "RTXPRO6000BW", 1)]

    tasks: list[tuple[str, int]] = []
    for platform, cfg in declared.items():
        if platform_filter and platform != platform_filter:
            continue
        if platform not in PLATFORMS:
            print(f"WARN: unknown platform {platform!r} in spec — skipping", file=sys.stderr)
            continue
        gpu_count = (cfg or {}).get("gpu_count", 1)
        tasks.append((platform, gpu_count))
    return tasks


# ---------------------------------------------------------------------------
# Instruction generation
# ---------------------------------------------------------------------------

def _instruction_intro(spec_stem: str, platform: str) -> str:
    """Contextual intro for the instruction.md based on spec type."""
    if spec_stem == "routing":
        return (
            f"Use the `/vss-deploy-detection-tracking-3d` skill to answer "
            f"informational routing questions on this `{platform}` host. "
            "Do NOT deploy any containers — this is a CPU-only routing-coverage "
            "eval. Answer by loading the correct skill's `SKILL.md` and reasoning "
            "about routing, without invoking `docker run`, `docker compose up`, "
            "NGC pulls, or any compose tree."
        )
    if spec_stem == "calibration-chain":
        return (
            f"Use the `/vss-deploy-detection-tracking-3d` skill (and chain to "
            f"`/vss-generate-video-calibration` as needed) on this `{platform}` host "
            "to deploy the full MV3DT stack on custom video data. This involves "
            "the AMC calibration chain followed by the MV3DT compose deployment. "
            "Do not invoke `/vss-deploy-profile` or `scripts/dev-profile.sh` — "
            "this skill uses its own compose tree at "
            "`deploy/docker/compose.yml` with `--env-file "
            "industry-profiles/warehouse-operations/.env`."
        )
    # deploy spec (default)
    return (
        f"Use the `/vss-deploy-detection-tracking-3d` skill on this `{platform}` host "
        "to deploy the MV3DT (Multi-View 3D Tracking) stack. This skill uses its own "
        "compose tree at `deploy/docker/compose.yml` with `--env-file "
        "industry-profiles/warehouse-operations/.env` gated on the "
        "`bp_wh_kafka_mv3dt` compose profile. Do not invoke `/vss-deploy-profile` "
        "or `scripts/dev-profile.sh`."
    )


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def generate_test_script(step: int, spec_name: str) -> str:
    """Shell wrapper invoking the generic LLM-as-judge verifier."""
    return (
        "#!/bin/bash\n"
        f"# vss-deploy-detection-tracking-3d verifier (step {step}): delegates to the\n"
        "# generic LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
        "set -euo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "python3 -m pip install --quiet 'anthropic>=0.40.0' >/dev/null 2>&1 || true\n"
        "\n"
        'python3 "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step}\n'
    )


def generate_solve_script(platform: str, spec_stem: str) -> str:
    """Gold solution — minimal assertion that the expected state holds."""
    if spec_stem == "routing":
        return (
            "#!/bin/bash\n"
            f"# Gold solution: vss-deploy-detection-tracking-3d (routing) on {platform}\n"
            "# Routing eval is informational — no deployment. The verifier judges\n"
            "# the agent's text responses against the spec's checks.\n"
            "set -euo pipefail\n"
            "\n"
            "echo 'Routing eval — no deployment needed. Verifier drives assertions.'\n"
        )
    if spec_stem == "calibration-chain":
        return (
            "#!/bin/bash\n"
            f"# Gold solution: vss-deploy-detection-tracking-3d (calibration-chain) on {platform}\n"
            "# The verifier judges the agent's AMC + MV3DT deploy actions against\n"
            "# the spec's checks; the solver asserts MV3DT is reachable.\n"
            "set -euo pipefail\n"
            "\n"
            "docker inspect --format '{{.State.Health.Status}}' vss-rtvi-cv-bev-fusion 2>/dev/null \\\n"
            "    | grep -qx healthy \\\n"
            "    && echo 'MV3DT BEV Fusion healthy — calibration-chain deploy succeeded.' \\\n"
            "    || echo 'BEV Fusion not healthy — verifier will report the gap.'\n"
        )
    # deploy spec
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-deploy-detection-tracking-3d (deploy) on {platform}\n"
        "# The verifier judges the agent's deploy/verify/teardown actions;\n"
        "# the solver asserts the MV3DT core containers are running.\n"
        "set -euo pipefail\n"
        "\n"
        "docker ps --format '{{.Names}}' | grep -qx vss-rtvi-cv-mv3dt \\\n"
        "    && echo 'MV3DT core container running — deploy succeeded.' \\\n"
        "    || echo 'MV3DT not running — verifier will report the gap.'\n"
    )


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_task(
    platform: str,
    gpu_count: int,
    spec: dict,
    spec_path: Path,
    output_root: Path,
    skill_dir: Path,
    calibration_skill_dir: Path | None,
) -> None:
    """Emit one Harbor task directory set for (spec, platform).

    Multi-step specs get step-<N>/ subdirs; single-step (unlikely for
    this skill) get a flat directory.
    """
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = spec_path.name
    spec_stem = _spec_stem(spec_path)
    rendered_spec = _substitute_spec(spec, platform)
    dataset_group = spec_stem

    for idx, expect in enumerate(rendered_spec.get("expects") or [], 1):
        step_dir = output_root / dataset_group / platform_short
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # -- instruction.md --
        instruction = [
            PREAMBLE,
            "",
            _instruction_intro(spec_stem, platform),
            "",
            f"## Query {idx} of {len(expects)}",
            "",
            expect.get("query", ""),
            "",
            "Run autonomously without prompting for confirmation.",
            "",
        ]
        (step_dir / "instruction.md").write_text("\n".join(instruction) + "\n")

        # -- task.toml --
        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""
        meta_lines = [
            "[task]",
            f'name = "nvidia-vss/vss-deploy-detection-tracking-3d-{dataset_group}-{platform_short}{step_suffix}"',
            f'description = "MV3DT {spec_stem} query {idx}/{len(expects)} on {platform}"',
            f'keywords = ["vss-deploy-detection-tracking-3d", "mv3dt", "{dataset_group}", "{platform}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            # Bump judge turns: MV3DT deploy trajectories are large
            # (multi-compose + health checks + broker verification).
            'JUDGE_MAX_TURNS = "50"',
            "",
            "[metadata]",
            'skill = "vss-deploy-detection-tracking-3d"',
            f'spec = "{spec_stem}"',
            f'platform = "{platform}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f"gpu_count = {gpu_count}",
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            "min_root_disk_gb = 120",
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        # -- environment/ placeholder --
        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        # -- tests/ --
        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        (tests_dir / spec_name).write_text(json.dumps(rendered_spec, indent=2))

        # -- solution/ --
        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform, spec_stem))

        # -- skills/ — include the primary skill + calibration skill if needed --
        # Selective copy: SKILL.md + references/ for this skill (the agent
        # needs reference docs to reason about the deployment). Omit evals/
        # to avoid leaking check content into the agent's context.
        dst_primary = step_dir / "skills" / "vss-deploy-detection-tracking-3d"
        if dst_primary.exists():
            shutil.rmtree(dst_primary)
        if skill_dir.exists():
            dst_primary.mkdir(parents=True, exist_ok=True)
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                shutil.copy2(skill_md, dst_primary / "SKILL.md")
            refs_src = skill_dir / "references"
            if refs_src.exists():
                shutil.copytree(refs_src, dst_primary / "references")

        # For calibration-chain spec, also include vss-generate-video-calibration
        if spec_stem == "calibration-chain" and calibration_skill_dir and calibration_skill_dir.exists():
            dst_calib = step_dir / "skills" / "vss-generate-video-calibration"
            if dst_calib.exists():
                shutil.rmtree(dst_calib)
            dst_calib.mkdir(parents=True, exist_ok=True)
            calib_skill_md = calibration_skill_dir / "SKILL.md"
            if calib_skill_md.exists():
                shutil.copy2(calib_skill_md, dst_calib / "SKILL.md")
            calib_refs = calibration_skill_dir / "references"
            if calib_refs.exists():
                shutil.copytree(calib_refs, dst_calib / "references")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True,
                        help="Dataset output root")
    parser.add_argument("--skill-dir", required=True,
                        help="Path to skills/vss-deploy-detection-tracking-3d")
    parser.add_argument("--calibration-skill-dir", default=None,
                        help="Path to skills/vss-generate-video-calibration "
                             "(included for calibration-chain spec)")
    parser.add_argument("--spec", default=None,
                        help="Path to a specific spec file (default: first found)")
    parser.add_argument("--all-specs", action="store_true",
                        help="Generate datasets for ALL specs under evals/")
    parser.add_argument("--platform", default=None, choices=list(PLATFORMS.keys()),
                        help="Generate for this platform only")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    calibration_skill_dir = (
        Path(args.calibration_skill_dir) if args.calibration_skill_dir else None
    )
    # Auto-detect calibration skill if not explicitly passed
    if calibration_skill_dir is None:
        candidate = skill_dir.parent / "vss-generate-video-calibration"
        if candidate.exists():
            calibration_skill_dir = candidate

    # Resolve spec(s)
    if args.spec:
        spec_paths = [Path(args.spec)]
    elif args.all_specs:
        evals_dir = skill_dir / "evals"
        if not evals_dir.exists():
            evals_dir = skill_dir / "eval"
        spec_paths = sorted(evals_dir.glob("*.json")) if evals_dir.exists() else []
    else:
        # Default: first spec found
        evals_dir = skill_dir / "evals"
        if not evals_dir.exists():
            evals_dir = skill_dir / "eval"
        spec_paths = sorted(evals_dir.glob("*.json"))[:1] if evals_dir.exists() else []

    if not spec_paths:
        print("No spec files found", file=sys.stderr)
        sys.exit(1)

    print("=== Inputs ===")
    print(f"  output_dir           : {output_root}")
    print(f"  skill_dir            : {skill_dir}")
    print(f"  calibration_skill_dir: {calibration_skill_dir or '(auto-detect failed)'}")
    print(f"  specs                : {[str(p.name) for p in spec_paths]}")
    print(f"  filter platform      : {args.platform or '(all declared)'}")
    print()

    total_generated = 0
    for spec_path in spec_paths:
        if not spec_path.exists():
            print(f"  SKIP {spec_path.name} — not found", file=sys.stderr)
            continue

        spec = json.loads(spec_path.read_text())
        spec["_source_path"] = str(spec_path)
        spec_stem = _spec_stem(spec_path)
        tasks = _platform_gpu_from_spec(spec, args.platform)

        print(f"  === Spec: {spec_path.name} (stem={spec_stem}) ===")
        print(f"      queries     : {len(spec.get('expects', []))}")
        print(f"      total checks: {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
        print(f"      platforms   : {tasks}")
        print()

        for platform, gpu_count in tasks:
            task_id = PLATFORMS[platform]["short_name"]
            print(f"      GEN  {spec_stem}/{task_id}  gpu_count={gpu_count}")
            generate_task(
                platform, gpu_count, spec, spec_path,
                output_root, skill_dir, calibration_skill_dir,
            )
            total_generated += 1

    print()
    print(f"Summary: {total_generated} task set(s) generated across {len(spec_paths)} spec(s).")


if __name__ == "__main__":
    main()
