"""Dispatcher CLI entry point for VoidFormer.

Usage:
    python -m voidformer.main train [args...]
    python -m voidformer.main infer [args...]
    python -m voidformer.main demo [args...]
    python -m voidformer.main test [args...]
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m voidformer.main <command> [args...]")
        print("Commands:")
        print("  train    - Train VoidFormer or QuantumVoidFormer")
        print("  infer    - Run text generation inference")
        print("  demo     - Run quantum core demo")
        print("  test     - Run quick integration test")
        sys.exit(1)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if cmd == "train":
        from voidformer.train import main as train_main
        train_main()
    elif cmd == "infer":
        from voidformer.infer import main as infer_main
        infer_main()
    elif cmd == "demo":
        from voidformer.demo_quantum import run_demo
        run_demo()
    elif cmd == "test":
        from voidformer.test_quantum_simple import run_test
        run_test()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
