#!/usr/bin/env python
"""Recompute schema-v3 metrics for every completed experiment below a root."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("output/dt4lm-improvements/run")


def resolve_input(path):
    """Resolve a CLI input relative to the DT4LM project root."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def discover_runs(input_dir):
    """Find completed run directories deterministically by their resolved config."""

    return sorted(path.parent for path in Path(input_dir).rglob("config.resolved.yaml"))


def evaluation_command(run_dir, stage):
    """Build an isolated evaluator process so model memory is released per run."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "statistics" / "evaluate_improvements.py"),
        "--stage",
        stage,
        "--config",
        str(run_dir / "config.resolved.yaml"),
        "--results",
        str(run_dir / "results.jsonl"),
        "--manifest",
        str(run_dir / "sample_manifest.json"),
        "--output-dir",
        str(run_dir / "metrics"),
        "--status-file",
        str(run_dir / "status.json"),
    ]
    nli_profile = run_dir / "nli_profile.json"
    if nli_profile.exists():
        command.extend(["--nli-profile", str(nli_profile)])
    return command


def quality_status(run_dir):
    """Return the consolidated quality status after evaluator completion."""

    path = run_dir / "metrics" / "quality.json"
    if not path.exists():
        return "missing"
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("status", "missing")
    except (json.JSONDecodeError, OSError):
        return "invalid"


def main():
    parser = argparse.ArgumentParser(
        description="Recompute metrics without rerunning completed attacks."
    )
    parser.add_argument("--i", default=str(DEFAULT_INPUT), help="Run tree root.")
    parser.add_argument(
        "--stage", choices=("core", "quality", "all"), default="all"
    )
    args = parser.parse_args()

    input_dir = resolve_input(args.i)
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment input directory does not exist: {input_dir}"
        )
    run_dirs = discover_runs(input_dir)
    if not run_dirs:
        raise ValueError(f"No config.resolved.yaml files found below {input_dir}.")

    failures = []
    for position, run_dir in enumerate(run_dirs, start=1):
        if not (run_dir / "results.jsonl").is_file() or not (
            run_dir / "sample_manifest.json"
        ).is_file():
            failures.append((run_dir, "missing results.jsonl or sample_manifest.json"))
            print(f"[{position}/{len(run_dirs)}] SKIP {run_dir}")
            continue
        print(f"[{position}/{len(run_dirs)}] EVALUATE {run_dir}", flush=True)
        process = subprocess.run(
            evaluation_command(run_dir, args.stage),
            cwd=PROJECT_ROOT,
            check=False,
        )
        if process.returncode != 0:
            failures.append((run_dir, f"evaluator exit code {process.returncode}"))
        elif (
            args.stage in {"quality", "all"}
            and quality_status(run_dir) != "completed"
        ):
            failures.append((run_dir, f"quality status {quality_status(run_dir)}"))

    if failures:
        details = "\n".join(f"- {path}: {reason}" for path, reason in failures)
        raise RuntimeError(
            f"{len(failures)} run(s) were not fully evaluated:\n{details}"
        )
    print(f"Completed metrics for {len(run_dirs)} run(s) below {input_dir}.")


if __name__ == "__main__":
    main()
