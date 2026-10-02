"""Unified, code-only DeLM launcher; select the upstream protocol with --n 2 or 4."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from runtime.config import DEFAULT_COUNT, DEFAULT_MODELS, EFFORT

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LaunchPlan:
    task: str
    series: str
    repeat: int
    n: int
    harness: str
    model: str
    effort: str
    service_cpus: float
    service_memory_mb: int
    record: str
    credential_file: str
    auth_mode: str = "api-key"


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("task", help="Fetched Terminal-Bench 4.0 task name")
    result.add_argument("--n", type=int, choices=(2, 4), default=DEFAULT_COUNT,
                        help=f"Number of DeLM workers / protocol variant (default: {DEFAULT_COUNT})")
    result.add_argument("--harness", choices=("codex", "claude-code"), default="codex")
    result.add_argument("--series", required=True, help="Immutable experiment settings; use a new name for changes")
    result.add_argument("--repeat", type=int, default=1)
    result.add_argument("--model", help="Defaults: " + "; ".join(f"{h}: {m}" for h, m in DEFAULT_MODELS.items()))
    result.add_argument("--effort", choices=(EFFORT,), default=EFFORT)
    result.add_argument("--auth-mode", choices=("api-key", "coding-plan"), default="api-key",
                        help="API billing or the selected harness's subscription (default: api-key)")
    credentials = result.add_mutually_exclusive_group(required=True)
    credentials.add_argument("--api-key-file", type=Path, help="Private raw API-key file for the selected provider")
    credentials.add_argument("--auth-file", type=Path, help="Codex auth.json or a raw Claude setup-token file")
    result.add_argument("--service-cpus", type=float, default=0,
                        help="Additional service CPU reservation per worker")
    result.add_argument("--service-memory-mb", type=int, default=0,
                        help="Additional service RAM reservation per worker")
    result.add_argument("--dry-run", action="store_true",
                        help="Print the resolved plan; no files, Docker, network or model calls")
    return result


def build_plan(args):
    """Validate the launch once; the runner relies on these checks."""
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
           for name in (args.task, args.series)):
        raise ValueError("Use simple task and series names, not filesystem paths")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if (not math.isfinite(args.service_cpus) or args.service_cpus < 0
            or args.service_memory_mb < 0):
        raise ValueError("Service reservations must be finite and nonnegative")
    model = args.model or DEFAULT_MODELS[args.harness]
    effort = args.effort
    if not model.strip() or effort != EFFORT:
        raise ValueError("An explicit model and xhigh effort are required")
    if args.auth_mode == "coding-plan":
        if not args.auth_file:
            raise ValueError("--auth-mode coding-plan requires --auth-file")
    elif not args.api_key_file:
        raise ValueError("--auth-mode api-key requires --api-key-file")
    credential = (args.auth_file or args.api_key_file).expanduser().resolve()
    folder = ROOT / ".local/runs" / args.series / args.task / f"repeat-{args.repeat}"
    return LaunchPlan(args.task, args.series, args.repeat, args.n, args.harness, model, effort,
                      args.service_cpus, args.service_memory_mb,
                      str(folder / "run.json"), str(credential), args.auth_mode)


def execute(plan):
    if sys.version_info < (3, 13):
        raise ValueError("Live runs require Python 3.13 or newer; --dry-run is available without launching")
    task = ROOT / "tasks/terminal-bench-4.0" / plan.task
    if not (task / "task.toml").is_file():
        raise ValueError("Task not fetched; run fetch_task.py first")
    if Path(plan.record).parent.exists():
        raise ValueError("Repeat folder already exists; inspect it rather than overwriting or retrying it")
    from runtime.runner import run
    metadata = run(task, plan.model, plan.effort, plan.series, plan.repeat, Path(plan.credential_file),
                   plan.service_cpus, plan.service_memory_mb, plan.n, plan.harness, plan.auth_mode)
    if metadata["status"] != "finished":
        raise ValueError("Run is incomplete; records preserved, no automatic retry")
    return 0


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(asdict(plan), indent=2))
            return 0
        return execute(plan)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        arguments.exit(1, f"delm: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
