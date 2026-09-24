"""Quick Integration & Sanity Verification Script for Quantum VoidFormer.

Usage:
    python -m voidformer.test_quantum_simple
"""

from __future__ import annotations

import torch
from voidformer.quantum_init import initialize_quantum_processor
from voidformer.models import QuantumVoidFormer


def main():
    print("Running quick quantum integration tests...")

    # 1. Test Quantum Processor
    proc = initialize_quantum_processor(d_model=32, n_qubits_per_token=2)
    x = torch.randn(2, 4, 32)
    out, diag = proc(x)
    assert out.shape == (2, 4, 32), f"Expected (2, 4, 32), got {out.shape}"
    print("  ✓ VirtualQuantumProcessor forward pass passed.")

    # 2. Test QuantumVoidFormer Model
    model = QuantumVoidFormer(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_qubits_per_token=2,
        max_seq_len=8,
    )
    tokens = torch.randint(0, 100, (2, 4))
    res = model(tokens, return_diagnostics=True)
    assert res.logits.shape == (2, 4, 100), f"Expected (2, 4, 100), got {res.logits.shape}"
    print("  ✓ QuantumVoidFormer end-to-end forward pass passed.")

    print("ALL SIMPLE QUANTUM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
