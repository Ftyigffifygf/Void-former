"""Quick Integration Test for Quantum VoidFormer.

Usage:
    python -m voidformer.test_quantum_simple
"""

from __future__ import annotations

import torch

from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.quantum_init import initialize_quantum_processor


def run_test() -> None:
    print("=" * 65)
    print("           QUANTUM VOIDFORMER QUICK INTEGRATION TEST             ")
    print("=" * 65)

    vocab_size = 100
    d_model = 64
    seq_len = 8
    batch_size = 2

    # 1. Test Processor Initialization
    print("\n1. Initializing Virtual Quantum Processor...")
    processor = initialize_quantum_processor(
        d_model=d_model,
        n_qubits_per_token=3,
        max_seq_len=16,
    )
    print(f"   Processor initialized successfully.")

    # 2. Test QuantumVoidFormer Model
    print("\n2. Initializing QuantumVoidFormer Model...")
    model = QuantumVoidFormer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        n_qubits_per_token=3,
        max_seq_len=16,
    )
    print(f"   Model parameters: {model.num_params():,}")

    # 3. Test Forward Pass
    print("\n3. Running Forward Pass...")
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = model(tokens, return_diagnostics=True)

    assert out.logits.shape == (batch_size, seq_len, vocab_size)
    assert torch.isfinite(out.logits).all()
    print(f"   Forward pass output logits shape: {tuple(out.logits.shape)}")

    # 4. Test Text Generation
    print("\n4. Running Quantum Text Generation...")
    gen_tokens = model.generate(tokens[:, :2], max_new_tokens=4, use_quantum=True)
    assert gen_tokens.shape == (batch_size, 6)
    print(f"   Generated sequence shape: {tuple(gen_tokens.shape)}")

    print("\n" + "=" * 65)
    print("                 INTEGRATION TEST PASSED                         ")
    print("=" * 65)


if __name__ == "__main__":
    run_test()
