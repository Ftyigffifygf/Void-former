"""Comprehensive Quantum Core Demo.

Demonstrates quantum state vector management, quantum gate execution,
superposition initialization, cross-token entanglement, measurement collapse,
and quantum algorithm execution (e.g. Grover's search / QFT).

Usage:
    python -m voidformer.demo_quantum
"""

from __future__ import annotations

import torch

from voidformer.quantum_init import initialize_quantum_processor
from voidformer.quantum import (
    QuantumStateVector,
    QubitStateManager,
    QuantumGateRegistry,
    EntanglementManager,
    MeasurementLayer,
    CollapseProtocol,
    QuantumAlgorithm,
)


def run_demo() -> None:
    print("=" * 65)
    print("          VOIDFORMER VIRTUAL QUANTUM CORE DEMO                   ")
    print("=" * 65)

    batch_size = 2
    seq_len = 4
    n_qubits = 3
    d_model = 64

    # 1. State Vector & Superposition Initialization
    print("\n1. Quantum Superposition State Initialization...")
    manager = QubitStateManager(n_qubits_per_token=n_qubits)
    uniform_state = manager.initialize_uniform_superposition(batch_size, seq_len)
    print(f"   State vector dimension: 2^{n_qubits} = {uniform_state.amplitudes.shape[-1]}")
    print(f"   Normalized check: {uniform_state.is_normalized}")
    print(f"   Initial von Neumann Entropy: {uniform_state.measure_entropy().mean().item():.4f} bits")

    # 2. Quantum Gate Application
    print("\n2. Executing Quantum Gates (Hadamard, Pauli-X, CNOT)...")
    registry = QuantumGateRegistry()
    H = registry.get_gate("H")
    CNOT = registry.get_gate("CNOT")

    # Start in basis state |000>
    basis_state = manager.initialize_computational_basis(batch_size, seq_len, basis_state=0)
    state_h = H.apply(basis_state, target_qubits=[0])
    state_bell = CNOT.apply(state_h, target_qubits=[0, 1])

    print(f"   After H(0) and CNOT(0,1):")
    print(f"   |000> amplitude: {state_bell.amplitudes[0, 0, 0].item()}")
    print(f"   |011> amplitude: {state_bell.amplitudes[0, 0, 3].item()}")

    # 3. Cross-Token Quantum Entanglement
    print("\n3. Applying Cross-Token Entanglement...")
    entangler = EntanglementManager(n_qubits_per_token=n_qubits, max_seq_len=16)
    entangled_state, shared_reg = entangler.apply_cross_token_entanglement(uniform_state)
    is_entangled, entropy = entangler.verify_entanglement(entangled_state)
    print(f"   State entangled: {is_entangled.item()}")
    print(f"   Bipartite Entanglement Entropy: {entropy.mean().item():.4f} bits")

    # 4. Born Rule Measurement Collapse
    print("\n4. Measurement Collapse Protocols...")
    layer_hard = MeasurementLayer(n_qubits=n_qubits, d_output=d_model, collapse_protocol=CollapseProtocol.HARD)
    layer_entropy = MeasurementLayer(n_qubits=n_qubits, d_output=d_model, collapse_protocol=CollapseProtocol.ENTROPY_GATED)

    out_hard, col_hard, diag_hard = layer_hard(entangled_state, return_collapsed_state=True)
    out_ent, col_ent, diag_ent = layer_entropy(entangled_state, return_collapsed_state=True)

    print(f"   Hard Collapse Output Shape: {tuple(out_hard.shape)}")
    print(f"   Entropy-Gated Collapse Effective Dim: {diag_ent['effective_dimension'].mean().item():.2f}")

    # 5. Virtual Quantum Processor & Algorithm Execution
    print("\n5. Virtual Quantum Processor Execution...")
    processor = initialize_quantum_processor(
        d_model=d_model,
        n_qubits_per_token=n_qubits,
        max_seq_len=16,
        collapse_protocol=CollapseProtocol.ENTROPY_GATED,
        enable_entanglement=True,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    out_proc, proc_diag = processor(x)
    print(f"   Processor output shape: {tuple(out_proc.shape)}")
    print(f"   Gates executed: {proc_diag['gates_applied']}")

    # Quantum Fourier Transform execution
    out_qft, qft_diag = processor.execute_algorithm(QuantumAlgorithm.QUANTUM_FOURIER_TRANSFORM, x)
    print(f"   QFT algorithm execution completed: output shape {tuple(out_qft.shape)}")

    print("\n" + "=" * 65)
    print("                  DEMO COMPLETED SUCCESSFULLY                  ")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
