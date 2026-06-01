#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the vss-deploy-detection-tracking-3d skill.

The vss-deploy-detection-tracking-3d skill deploys Multi-View 3D Tracking
(MV3DT / RTVI-CV-3D) on the VSS warehouse-blueprint compose tree at
`deploy/docker/compose.yml`, gated on the `bp_wh_kafka_mv3dt` compose
profile with `--env-file industry-profiles/warehouse-operations/.env`.

It is **standalone** — there is no `/vss-deploy-profile` prerequisite. The
skill drives the warehouse-operations compose tree directly; the eval
harness runs each trial on a bare RTXPRO6000BW host and the agent
deploys/verifies/tears-down via the skill's own references.

Three specs ship with the skill (all target `RTXPRO6000BW`, mode
`standalone`):
  - evals/evals.json             — DEPLOY -> VERIFY -> TEARDOWN on the
                                    ship-with-repo sample dataset (GPU).
  - evals/calibration-chain.json — custom-data AMC -> MV3DT chain ->
                                    TEARDOWN (GPU; chains the
                                    vss-generate-video-calibration skill).
  - evals/routing.json           — CPU-only routing-coverage probe
                                    (gpu_count = 0; the agent must NOT
                                    deploy any container).

Each spec's `expects` list contains multiple ordered steps; cases run
in declared order with state preserved between them. The adapter emits
a `step-<N>/` subdir per step so Harbor's dispatch loop (driven by the
coordinator) runs them in order with skip-on-prior-fail. The framework
must NOT reset Docker / container state between steps within a spec.

Unlike the sibling vss-deploy-detection-tracking-2d adapter, this skill
is reference-driven (no `scripts/` dir): many spec checks assert that
the agent actually read `references/deploy-rtvi-cv-3d-stack.md`,
`references/calibration-workflow.md`, `references/configure-cameras.md`,
`references/verify-and-view.md`, or `references/teardown.md`. The
payload therefore copies SKILL.md + the full `references/` tree
(SKILL.md + references gzip+base64 to ~57 KB, well under brev exec's
128 KB MAX_ARG_STRLEN upload limit).

Usage from the repository root:
    python3 .github/skill-eval/adapters/vss-deploy-detection-tracking-3d/generate.py \\
        --output-dir /tmp/skill-eval/<run>/datasets/vss-deploy-detection-tracking-3d/evals \\
        --skill-dir skills/vss-deploy-detection-tracking-3d \\
        --spec skills/vss-deploy-detection-tracking-3d/evals/evals.json
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
DEFAULT_SPEC = "evals.json"

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


def _platform_modes_from_spec(spec: dict, platform_filter: str | None) -> list[tuple[str, str]]:
    declared = ((spec.get("resources") or {}).get("platforms") or {})
    if not declared:
        declared = {DEFAULT_PLATFORM: {"modes": [DEFAULT_MODE]}}

    tasks: list[tuple[str, str]] = []
    for platform, cfg in declared.items():
        if platform_filter and platform != platform_filter:
            continue
        if platform not in PLATFORMS:
            continue
        for mode in (cfg or {}).get("modes") or [DEFAULT_MODE]:
            tasks.append((platform, mode))
    return tasks or [(platform_filter or DEFAULT_PLATFORM, DEFAULT_MODE)]


def _spec_kind(spec_path: Path) -> str:
    """Classify the spec for instruction intro + dataset grouping.

    Reads the spec *filename* stem (free-form per AGENTS.md) and maps
    the known three to stable kinds; anything else falls back to its
    stem so a future spec still groups cleanly.
    """
    stem = spec_path.stem.lower()
    if "routing" in stem:
        return "routing"
    if "calibration" in stem or "chain" in stem:
        return "calibration"
    # evals.json is the deploy spec
    if "deploy" in stem or "evals" in stem:
        return "deploy"
    return stem


def _gpu_count_from_spec(spec: dict, platform: str) -> int:
    cfg = (((spec.get("resources") or {}).get("platforms") or {}).get(platform) or {})
    # routing.json declares gpu_count = 0 (CPU-only); deploy/calibration
    # declare 1. Honor whatever the spec says; default to 1 for GPU work.
    return int(cfg.get("gpu_count", 1))


def _min_disk_for_kind(kind: str) -> int:
    # calibration-chain pulls VGGT (~4.7 GB) + ~30 GB app-data + AMC
    # project state on top of the MV3DT image set: spec says >=100 GB.
    # deploy uses the sample dataset: spec says >=50 GB. routing is
    # CPU-only and touches no images.
    if kind == "calibration":
        return 100
    if kind == "routing":
        return 20
    return 60


def _instruction_intro(kind: str, platform: str) -> str:
    if kind == "routing":
        return (
            f"Answer the following routing/coverage question on this `{platform}` host. "
            "This is an **informational** query only — do NOT deploy or modify any "
            "containers, do NOT run `docker run` / `docker compose up` / `docker pull`, "
            "and do NOT invoke `/vss-deploy-profile` or the warehouse compose tree. "
            "Answer by loading the correct skill's `SKILL.md` and reasoning about "
            "routing. The harness rejects any trial where `docker ps -a` count changes."
        )
    if kind == "calibration":
        return (
            f"Use the `/vss-deploy-detection-tracking-3d` skill on this `{platform}` host "
            "to run the end-to-end custom-data calibration chain: drive the AMC "
            "(auto multi-camera calibration) flow via the `vss-generate-video-calibration` "
            "skill, then deploy Multi-View 3D Tracking (MV3DT) on the warehouse-operations "
            "compose tree at `deploy/docker/compose.yml` with "
            "`--env-file industry-profiles/warehouse-operations/.env`, gated on the "
            "`bp_wh_kafka_mv3dt` compose profile. Do NOT invoke `/vss-deploy-profile` or "
            "`scripts/dev-profile.sh`. Follow the skill's references "
            "(`calibration-workflow.md` -> `configure-cameras.md` -> "
            "`deploy-rtvi-cv-3d-stack.md` -> `verify-and-view.md` -> `teardown.md`)."
        )
    return (
        f"Use the `/vss-deploy-detection-tracking-3d` skill on this `{platform}` host "
        "to deploy, verify, or tear down Multi-View 3D Tracking (MV3DT / RTVI-CV-3D) on "
        "the VSS warehouse-blueprint compose tree at `deploy/docker/compose.yml` with "
        "`--env-file industry-profiles/warehouse-operations/.env`, gated on the "
        "`bp_wh_kafka_mv3dt` compose profile. Do NOT invoke `/vss-deploy-profile` or "
        "`scripts/dev-profile.sh`. Follow the skill's references "
        "(`deploy-rtvi-cv-3d-stack.md`, `verify-and-view.md`, `teardown.md`)."
    )


def generate_test_script(step: int, spec_name: str) -> str:
    # The script's exit code MUST reflect whether the judge itself ran
    # cleanly. Harbor reads the per-check reward from
    # /logs/verifier/reward.txt (which the judge writes even on partial
    # pass/fail), but a non-zero exit signals a verifier-side failure
    # the harness should report distinctly from low-reward outcomes.
    #
    # `set -e` plus no trailing `exit 0` ensures the judge's actual exit
    # code propagates: judge exit 0 -> script exits 0 and Harbor scores
    # reward.txt; judge non-zero (spec parse error, missing trajectory,
    # SDK import failure) -> Harbor reports a verifier failure rather
    # than silently scoring zero.
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


def generate_solve_script(platform: str, kind: str) -> str:
    if kind == "routing":
        return (
            "#!/bin/bash\n"
            f"# Gold solution: vss-deploy-detection-tracking-3d (routing) on {platform}\n"
            "# Routing is informational and CPU-only — there is nothing to deploy.\n"
            "# The verifier judges the agent's routing answer + no-deploy invariant.\n"
            "set -euo pipefail\n"
            "echo 'Routing query — verifier judges the agent response; nothing to solve.'\n"
        )
    if kind == "calibration":
        return (
            "#!/bin/bash\n"
            f"# Gold solution: vss-deploy-detection-tracking-3d (calibration chain) on {platform}\n"
            "# The verifier judges the AMC->MV3DT chain against the spec's checks;\n"
            "# the solver asserts BEV fusion is healthy after the agent runs.\n"
            "set -euo pipefail\n"
            "docker inspect --format '{{.State.Health.Status}}' vss-rtvi-cv-bev-fusion 2>/dev/null \\\n"
            "    | grep -qx healthy \\\n"
            "    && echo 'BEV Fusion healthy — calibration chain + deploy succeeded.' \\\n"
            "    || echo 'BEV Fusion not healthy — verifier will report the gap.'\n"
        )
    return (
        "#!/bin/bash\n"
        f"# Gold solution: vss-deploy-detection-tracking-3d (deploy) on {platform}\n"
        "# The verifier judges the agent's deploy/verify/teardown actions against the\n"
        "# spec's checks; the solver asserts BEV fusion health as a coarse reachability\n"
        "# signal (no-op for the teardown step, where the container is intentionally gone).\n"
        "set -euo pipefail\n"
        "docker inspect --format '{{.State.Health.Status}}' vss-rtvi-cv-bev-fusion 2>/dev/null \\\n"
        "    | grep -qx healthy \\\n"
        "    && echo 'BEV Fusion healthy.' \\\n"
        "    || echo 'BEV Fusion not healthy/absent — expected after teardown; verifier judges.'\n"
    )


def _copy_skill_payload(skill_dir: Path, dst: Path) -> None:
    """Copy SKILL.md + references/ into the per-step payload.

    This skill is reference-driven (no scripts/ dir): the spec checks
    require the agent to read references/*.md by name. SKILL.md +
    references gzip+base64 to ~57 KB, under brev exec's 128 KB
    MAX_ARG_STRLEN upload limit, so copying the full references tree is
    safe. eval/ is omitted (the agent must not see its own answer key).
    """
    if dst.exists():
        shutil.rmtree(dst)
    if not skill_dir.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        shutil.copy2(skill_md, dst / "SKILL.md")
    refs_src = skill_dir / "references"
    if refs_src.exists():
        shutil.copytree(refs_src, dst / "references")
    # skill-card / BENCHMARK are explanatory metadata — copy if present
    # but cheap; they help the agent self-describe without leaking the
    # eval answer key.
    for extra in ("skill-card.md", "BENCHMARK.md"):
        src = skill_dir / extra
        if src.exists():
            shutil.copy2(src, dst / extra)


def generate_task(
    platform: str,
    mode: str,
    spec: dict,
    spec_path: Path,
    output_root: Path,
    skill_dir: Path,
) -> None:
    pspec = PLATFORMS[platform]
    platform_short = pspec["short_name"]
    expects = spec.get("expects") or []
    spec_name = spec_path.name
    kind = _spec_kind(spec_path)
    rendered_spec = _substitute_spec(spec, platform, mode)
    gpu_count = _gpu_count_from_spec(spec, platform)
    min_disk = _min_disk_for_kind(kind)

    # Coordinator drives harbor with `-p <output_root>` and
    # `--include-task-name "<platform>[-step-<N>]"`, where <platform>
    # is the short name (e.g. rtxpro6000bw). Lay steps out as
    # <output_root>/<platform_short>[/step-<N>] so the suffix glob
    # matches the AGENTS.md harbor templates exactly.
    for idx, expect in enumerate(rendered_spec.get("expects") or [], 1):
        step_dir = output_root / platform_short
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
            f"nvidia-vss/vss-deploy-detection-tracking-3d-{kind}-{platform_short}{step_suffix}"
        )
        meta_lines = [
            "[task]",
            f'name = "{task_name}"',
            f'description = "MV3DT {kind} query {idx}/{len(expects)} on {platform}/{mode}"',
            f'keywords = ["vss-deploy-detection-tracking-3d", "mv3dt", "rtvi-cv-3d", "{kind}", "{platform}", "{mode}"]',
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
            'skill = "vss-deploy-detection-tracking-3d"',
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
        ]
        meta_lines.append("")
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

        _copy_skill_payload(skill_dir, step_dir / "skills" / "vss-deploy-detection-tracking-3d")

        # The calibration-chain spec chains to vss-generate-video-calibration;
        # several checks require the agent to read that skill's references.
        # Copy it alongside when present so the chain trial isn't blind.
        if kind == "calibration":
            sibling = skill_dir.parent / "vss-generate-video-calibration"
            if sibling.exists():
                _copy_skill_payload(
                    sibling, step_dir / "skills" / "vss-generate-video-calibration"
                )


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
    tasks = _platform_modes_from_spec(spec, args.platform)
    kind = _spec_kind(spec_path)

    print("=== Inputs ===")
    print(f"  output_dir   : {output_root}")
    print(f"  skill_dir    : {skill_dir}")
    print(f"  spec         : {spec_path}")
    print(f"  spec kind    : {kind}")
    print(f"  tasks        : {tasks}")
    print(f"  queries      : {len(spec.get('expects', []))}")
    print(f"  total checks : {sum(len(q.get('checks', [])) for q in spec.get('expects', []))}")
    print()

    for platform, mode in tasks:
        print(
            f"  GEN  vss-deploy-detection-tracking-3d/{kind}/"
            f"{PLATFORMS[platform]['short_name']}"
        )
        generate_task(platform, mode, spec, spec_path, output_root, skill_dir)

    print()
    print(f"Generated {len(tasks)} task(s) under {output_root}/")


if __name__ == "__main__":
    main()
