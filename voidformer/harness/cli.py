"""CLI Dispatcher Entry Point for Harness Package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from harness.ab_benchmark import run_ab_benchmark
from harness.ablation_sweep import run_ablation_sweep
from harness.scaling_profile import profile_scaling


def main():
    parser = argparse.ArgumentParser(description="VoidFormer Evaluation Harness CLI")
    parser.add_argument("mode", choices=["benchmark", "sweep", "profile", "all"], help="Execution mode")
    parser.add_argument("--steps", type=int, default=10, help="Training steps")
    args = parser.parse_args()

    if args.mode in ["benchmark", "all"]:
        run_ab_benchmark(steps=args.steps)
    if args.mode in ["sweep", "all"]:
        run_ablation_sweep(steps=args.steps)
    if args.mode in ["profile", "all"]:
        profile_scaling()


if __name__ == "__main__":
    main()
