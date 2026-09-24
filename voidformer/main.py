"""Main Dispatcher CLI for VoidFormer.

Routes subcommands:
    python -m voidformer.main train ...
    python -m voidformer.main infer ...
    python -m voidformer.main demo ...
    python -m voidformer.main sanity ...
"""

from __future__ import annotations

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="VoidFormer CLI Dispatcher")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to run")

    train_parser = subparsers.add_parser("train", help="Run model training")
    infer_parser = subparsers.add_parser("infer", help="Run model inference")
    demo_parser = subparsers.add_parser("demo", help="Run quantum simulation demo")
    sanity_parser = subparsers.add_parser("sanity", help="Run sanity check experiment")

    args, remaining_args = parser.parse_known_args()

    if args.subcommand == "train":
        from voidformer.train import main as train_main
        sys.argv = [sys.argv[0]] + remaining_args
        train_main()
    elif args.subcommand == "infer":
        from voidformer.infer import main as infer_main
        sys.argv = [sys.argv[0]] + remaining_args
        infer_main()
    elif args.subcommand == "demo":
        from voidformer.demo_quantum import main as demo_main
        sys.argv = [sys.argv[0]] + remaining_args
        demo_main()
    elif args.subcommand == "sanity":
        from voidformer.experiments.sanity_run import main as sanity_main
        sys.argv = [sys.argv[0]] + remaining_args
        sanity_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
