#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-deploy-detection-tracking-3d skill.

The vss-deploy-detection-tracking-3d skill (MV3DT / RTVI-CV-3D /
Multi-View 3D Tracking) deploys the warehouse-blueprint perception stack
— per-camera DeepStream perception plus BEV Fusion — over multiple
calibrated cameras. Like its 2D sibling it is **standalone**: there is no
`/vss-deploy-profile` prerequisite. The skill drives the warehouse
compose tree at `deploy/docker/compose.yml` directly, gated on the
`bp_wh_kafka_mv3dt` compose profile via
`--env-file industry-profiles/warehouse-operations/.env`.

Three specs ship with the skill (all under `skills/.../evals/`):

  - evals/deploy.json            — DEPLOY → VERIFY → TEARDOWN on the
                                   ship-with-repo sample dataset.
                                   RTXPRO6000BW, gpu_count=1. Multi-step,
                                   state preserved between steps.
  - evals/calibration-chain.json — custom-data end-to-end: AMC →
                                   calibration → MV3DT → verify →
                                   teardown. Chains to
                                   `vss-generate-video-calibration`.
                                   RTXPRO6000BW, gpu_count=1.
  - evals/routing.json           — CPU-only routing-coverage probe.
                                   gpu_count=0 (no deploy). Each query is
                                   informational; the framework rejects
                                   trials that actually start containers.

Each spec's `expects` list contains multiple ordered steps; the adapter
emits a `step-<N>/` subdir per step so Harbor's dispatch loop runs them
in declared order with skip-on-prior-fail. State is preserved between
steps (the environment must NOT reset Docker between them).

Datasets are laid out as
    <output-dir>/<spec_stem>/<platform_short>-<mode>/step-<N>/
so they match the AGENTS.md convention
`datasets/<skill>/<spec_stem>/<platform>` (the coordinator passes
`-p <output-dir>/<spec_stem>` to `uvx harbor run`).

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-deploy-detection-tracking-3d/generate.py \\
        --output-dir /tmp/skill-eval/datasets/vss-deploy-detection-tracking-3d \\
        --skill-dir skills/vss-deploy-detection-tracking-3d \\
        --spec skills/vss-deploy-detection-tracking-3d/evals/deploy.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

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

DEFAULT_PLATFORM = "RTXPRO6000BW"
DEFAULT_MODE = "standalone"
DEFAULT_SPEC = "deploy.json"

SKILL_NAME = "vss-deploy-detection-tracking-3d"

GENERIC_JUDGE = Path(__file__).resolve().parents[2] / "verifiers" / "generic_judge.py"

PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "You are pre-authorized to deploy prerequisites autonomously — "
    "do not pause to ask for confirmation on `/vss-deploy-profile` or any other "
    "setup action the trial requires."
)


def _substitute_spec(spec: dict, platform: str, mode: str) -> dict:
    substitutions = {
        "platform": platform,
        "mode": mode,
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


def _platform_cfgs_from_spec(spec: dict, platform_filter: str | None) -> list[tuple[str, str, dict]]:
    """Return (platform, mode, platform_cfg) tuples declared by the spec.

    The platform_cfg dict is carried through so per-task gpu_count is read
    from the spec rather than hardcoded — routing.json declares
    gpu_count=0 (CPU-only routing probe) and the harness's
    _check_instance_matches skips the GPU-type check only when task.toml
    reports gpu_count=0.
    """
    declared = ((spec.get("resources") or {}).get("platforms") or {})
    if not declared:
        declared = {DEFAULT_PLATFORM: {"modes": [DEFAULT_MODE]}}

    tasks: list[tuple[str, str, dict]] = []
    for platform, cfg in declared.items():
        if platform_filter and platform != platform_filter:
            continue
        if platform not in PLATFORMS:
            continue
        cfg = cfg or {}
        for mode in cfg.get("modes") or [DEFAULT_MODE]:
            tasks.append((platform, mode, cfg))
    return tasks


def _spec_kind(spec_path: Path, spec: dict) -> str:
    """Classify the spec for instruction-text selection.

    - routing : CPU-only, gpu_count==0 on every declared platform, no deploy.
    - calib   : the custom-data AMC chain (spec lists the calibration skill).
    - deploy  : the sample-data DEPLOY/VERIFY/TEARDOWN flow (default).
    """
    declared = ((spec.get("resources") or {}).get("platforms") or {})
    gpu_counts = [int((cfg or {}).get("gpu_count", 1) or 0) for cfg in declared.values()]
    if declared and all(c == 0 for c in gpu_counts):
        return "routing"
    if "vss-generate-video-calibration" in (spec.get("skills") or []):
        return "calib"
    return "deploy"


def _instruction_intro(kind: str, platform: str) -> str:
    if kind == "routing":
        return (
            "This is a CPU-only routing-coverage probe. Answer the query by "
            "loading the correct skill's `SKILL.md` and reasoning about routing. "
            "**Do NOT deploy or modify any containers** — no `docker run`, "
            "`docker compose up`, `docker pull`, or NGC pulls. The harness "
            "rejects any trial where the container count changes. Each query is "
            "purely informational; explain, do not execute."
        )
    if kind == "calib":
        return (
            f"Use the `/{SKILL_NAME}` skill on this `{platform}` host to deploy "
            "MV3DT (RTVI-CV-3D) on a custom 4-camera video dataset that has no "
            "calibration yet. The calibration is missing, so you must chain to "
            "`/vss-generate-video-calibration` (the AMC skill) FIRST to produce "
            "`calibration.json` + per-camera `camInfo/`, then configure and bring "
            "up the warehouse-operations compose tree. Drive the warehouse "
            "compose tree at `deploy/docker/compose.yml` with "
            "`--env-file industry-profiles/warehouse-operations/.env` gated on the "
            "`bp_wh_kafka_mv3dt` compose profile. Do NOT invoke `/vss-deploy-profile` "
            "or `scripts/dev-profile.sh`."
        )
    return (
        f"Use the `/{SKILL_NAME}` skill on this `{platform}` host to deploy, "
        "verify, or tear down the MV3DT (RTVI-CV-3D) stack on the ship-with-repo "
        "sample dataset. Drive the warehouse-blueprint compose tree at "
        "`deploy/docker/compose.yml` with "
        "`--env-file industry-profiles/warehouse-operations/.env` gated on the "
        "`bp_wh_kafka_mv3dt` compose profile. Do NOT invoke `/vss-deploy-profile` "
        "or `scripts/dev-profile.sh` — this skill is standalone."
    )


def generate_test_script(step: int, spec_name: str) -> str:
    # The script's exit code MUST reflect whether the judge itself ran
    # cleanly. Harbor reads the per-check reward from
    # /logs/verifier/reward.txt (which the judge writes even on partial
    # pass/fail), but a non-zero exit signals a verifier-side failure
    # the harness should report distinctly from low-reward outcomes.
    # `set -e` plus no trailing `exit 0` propagates the judge's code.
    return (
        "#!/bin/bash\n"
        f"# {SKILL_NAME} verifier (step {step}): delegates to the\n"
        "# generic LLM-as-judge (.github/skill-eval/verifiers/generic_judge.py).\n"
        "set -euo pipefail\n"
        "\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        "python3 -m pip install --quiet 'anthropic>=0.40.0' >/dev/null 2>&1 || true\n"
        "\n"
        'python3 "$TEST_DIR/generic_judge.py" \\\n'
        f'    --spec "$TEST_DIR/{spec_name}" --step {step}\n'
    )


def generate_solve_script(platform: str, kind: str) -> str:
    if kind == "routing":
        return (
            "#!/bin/bash\n"
            f"# Gold solution: {SKILL_NAME} (routing) on {platform}\n"
            "# Routing probe is informational only — no deploy. The verifier\n"
            "# judges the agent's routing answer against the spec's checks.\n"
            "set -euo pipefail\n"
            "echo 'Routing probe — no deploy expected; verifier judges the answer.'\n"
        )
    return (
        "#!/bin/bash\n"
        f"# Gold solution: {SKILL_NAME} ({kind}) on {platform}\n"
        "# The verifier judges the agent's deploy/teardown actions against the\n"
        "# spec's checks; the solver simply reports the current MV3DT state.\n"
        "set -euo pipefail\n"
        "\n"
        "if docker ps --format '{{.Names}}' | grep -qx vss-rtvi-cv-bev-fusion; then\n"
        "    echo 'MV3DT BEV Fusion container is running.'\n"
        "else\n"
        "    echo 'MV3DT not running — verifier will report the gap (expected after teardown).'\n"
        "fi\n"
    )


def _copy_skill(skill_src: Path, dst_root: Path) -> None:
    """Copy a skill's SKILL.md + references/ (+ scripts/ if present).

    The 3D skill drives the compose tree via its `references/` docs (it
    has no `scripts/`), and several spec checks require the agent to read
    `references/deploy-rtvi-cv-3d-stack.md`, `configure-cameras.md`,
    `calibration-workflow.md`, etc. — so references/ is copied here
    (unlike the 2D adapter, which omits them). The combined gzipped
    payload (3D SKILL.md+references ~38 KB, +calibration skill ~67 KB)
    stays well under brev exec's 128 KB MAX_ARG_STRLEN upload limit.
    """
    if not skill_src.exists():
        return
    dst = dst_root / skill_src.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    skill_md = skill_src / "SKILL.md"
    if skill_md.exists():
        shutil.copy2(skill_md, dst / "SKILL.md")
    for sub in ("references", "scripts"):
        sub_src = skill_src / sub
        if sub_src.exists():
            shutil.copytree(sub_src, dst / sub)


def generate_task(
    platform: str,
    mode: str,
    platform_cfg: dict,
    spec: dict,
    spec_path: Path,
    output_root: Path,
    skill_dir: Path,
) -> None:
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = spec_path.name
    spec_stem = spec_path.stem
    kind = _spec_kind(spec_path, spec)
    rendered_spec = _substitute_spec(spec, platform, mode)
    gpu_count = int(platform_cfg.get("gpu_count", 1) or 0)

    # Disk hint: calibration chain needs ~100 GB (VGGT + AMC state); the
    # plain sample deploy needs ~50 GB; routing is CPU-only and tiny.
    if kind == "calib":
        min_disk = 100
    elif kind == "routing":
        min_disk = 20
    else:
        min_disk = 60

    skills_root = skill_dir.parent
    spec_skills = spec.get("skills") or [SKILL_NAME]

    for idx, expect in enumerate(rendered_spec.get("expects") or [], 1):
        step_dir = output_root / spec_stem / f"{platform_short}-{mode}"
        if len(expects) > 1:
            step_dir = step_dir / f"step-{idx}"
        step_dir.mkdir(parents=True, exist_ok=True)

        instruction = [
            PREAMBLE,
            "",
            _instruction_intro(kind, platform),
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
        (step_dir / "instruction.md").write_text("\n".join(instruction) + "\n")

        step_suffix = f"-step-{idx}" if len(expects) > 1 else ""
        task_name = (
            f"nvidia-vss/{SKILL_NAME}-{spec_stem}-{platform_short}-{mode}{step_suffix}"
        )
        meta_lines = [
            "[task]",
            f'name = "{task_name}"',
            f'description = "MV3DT {kind} query {idx}/{len(expects)} on {platform}/{mode}"',
            f'keywords = ["{SKILL_NAME}", "mv3dt", "rtvi-cv-3d", "{spec_stem}", "{platform}", "{mode}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[verifier.env]",
            'ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"',
            'ANTHROPIC_BASE_URL = "${ANTHROPIC_BASE_URL}"',
            'ANTHROPIC_MODEL = "${ANTHROPIC_MODEL}"',
            "",
            "[metadata]",
            f'skill = "{SKILL_NAME}"',
            f'spec = "{spec_stem}"',
            f'deployment = "{kind}"',
            f'platform = "{platform}"',
            f'mode = "{mode}"',
            f'gpu_type = "{pspec["gpu_type"]}"',
            f'brev_search = "{pspec["brev_search"]}"',
            f"gpu_count = {gpu_count}",
            f'min_vram_gb_per_gpu = {pspec["min_vram_per_gpu"]}',
            f"min_root_disk_gb = {min_disk}",
            f"step_index = {idx}",
            f"step_count = {len(expects)}",
            f"check_count = {len(expect.get('checks') or [])}",
            "",
        ]
        (step_dir / "task.toml").write_text("\n".join(meta_lines))

        env_dir = step_dir / "environment"
        env_dir.mkdir(exist_ok=True)
        (env_dir / "Dockerfile").write_text("FROM scratch\n")

        tests_dir = step_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        (tests_dir / "test.sh").write_text(generate_test_script(idx, spec_name))
        if GENERIC_JUDGE.exists():
            shutil.copy(GENERIC_JUDGE, tests_dir / "generic_judge.py")
        (tests_dir / spec_name).write_text(json.dumps(rendered_spec, indent=2))

        solution_dir = step_dir / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(generate_solve_script(platform, kind))

        # Copy every skill the spec declares (primary 3D skill always; the
        # calibration skill too for calibration-chain). Resolve siblings
        # relative to the primary skill's parent directory.
        skills_dst = step_dir / "skills"
        if skills_dst.exists():
            shutil.rmtree(skills_dst)
        skills_dst.mkdir(parents=True, exist_ok=True)
        for sk in spec_skills:
            _copy_skill(skills_root / sk, skills_dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument(
        "--spec",
        default=None,
        help=f"Path to spec file (default: <skill-dir>/evals/{DEFAULT_SPEC})",
    )
    parser.add_argument("--platform", default=None, choices=list(PLATFORMS.keys()))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    skill_dir = Path(args.skill_dir)
    spec_path = Path(args.spec) if args.spec else (skill_dir / "evals" / DEFAULT_SPEC)

    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec = json.loads(spec_path.read_text())
    spec["_source_path"] = str(spec_path)

    declared_platforms = ((spec.get("resources") or {}).get("platforms") or {})
    if not declared_platforms:
        print(
            f"spec {spec_path} has no resources.platforms — refusing to "
            "synthesize a default. Add a platforms matrix to the spec.",
            file=sys.stderr,
        )
        sys.exit(2)

    tasks = _platform_cfgs_from_spec(spec, args.platform)
    if not tasks:
        print(
            f"spec {spec_path} declares no platform supported by this adapter "
            f"(declared={list(declared_platforms)}, known={list(PLATFORMS)}).",
            file=sys.stderr,
        )
        sys.exit(2)
    kind = _spec_kind(spec_path, spec)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  spec stem    : {spec_path.stem}")
    print(f"  spec kind    : {kind}")
    print(f"  skills       : {spec.get('skills')}")
    print(f"  tasks        : {[(p, m, int((c or {}).get('gpu_count', 1) or 0)) for p, m, c in tasks]}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()

    for platform, mode, cfg in tasks:
        print(
            f"  GEN  {SKILL_NAME}/{spec_path.stem}/"
            f"{PLATFORMS[platform]['short_name']}-{mode} (gpu_count={int((cfg or {}).get('gpu_count', 1) or 0)})"
        )
        generate_task(platform, mode, cfg, spec, spec_path, output_root, skill_dir)

    print()
    print(f"Generated {len(tasks)} task(s) under {output_root}/{spec_path.stem}/")


if __name__ == "__main__":
    main()
