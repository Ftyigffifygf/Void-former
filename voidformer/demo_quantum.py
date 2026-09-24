"""Comprehensive Quantum Processor & Model Demo.

Usage:
    python -m voidformer.demo_quantum
"""

from __future__ import annotations

import torch
from voidformer.quantum_init import initialize_quantum_processor
from voidformer.quantum import (
    QubitStateManager,
    QuantumGateRegistry,
    EntanglementManager,
    MeasurementLayer,
    CollapseProtocol,
)
from voidformer.models import QuantumVoidFormer


def main():
    print("=" * 70)
    print("      🔬 VOIDFORMER VIRTUAL QUANTUM SIMULATOR DEMO 🔬      ")
    print("=" * 70)

    device = torch.device("cpu")
    batch_size = 2
    seq_len = 4
    d_model = 64
    n_qubits = 3

    print("\n1. Quantum State Vector Initialization")
    manager = QubitStateManager(n_qubits_per_token=n_qubits, device=device)
    basis_state = manager.initialize_computational_basis(batch_size, seq_len, basis_state=0)
    super_state = manager.initialize_uniform_superposition(batch_size, seq_len)

    norm_status = basis_state.is_normalized() if callable(basis_state.is_normalized) else basis_state.is_normalized
    print(f"  Initialized Basis State |0⟩ Shape: {basis_state.amplitudes.shape}")
    print(f"  Is Normalized: {norm_status}")
    print(f"  Uniform Superposition Entropy: {super_state.measure_entropy()[0, 0].item():.4f} bits")

    print("\n2. Quantum Gates (Unitary Evolution)")
    gate_reg = QuantumGateRegistry(device=device)
    H = gate_reg.get_gate("H")
    CNOT = gate_reg.get_gate("CNOT")

    # Apply H then CNOT on 2-qubit state
    state2 = QubitStateManager(2, device=device).initialize_computational_basis(1, 1, 0)
    s_h = H.apply(state2, target_qubits=[0])
    bell_state = CNOT.apply(s_h, target_qubits=[0, 1])

    print(f"  Bell State |Φ⁺⟩ Amplitudes: {bell_state.amplitudes[0, 0].tolist()}")

    print("\n3. Entanglement & Concurrence")
    ent_mgr = EntanglementManager(n_qubits_per_token=2, max_seq_len=16, device=device)
    concurrence = ent_mgr.compute_concurrence(bell_state)
    print(f"  Calculated Concurrence: {concurrence[0, 0].item():.4f} (1.0 = Maximally Entangled)")

    print("\n4. Measurement Collapse Protocols")
    m_layer_hard = MeasurementLayer(n_qubits=2, d_output=d_model, collapse_protocol=CollapseProtocol.HARD)
    m_layer_soft = MeasurementLayer(n_qubits=2, d_output=d_model, collapse_protocol=CollapseProtocol.SOFT)

    out_hard, col_hard, diag_hard = m_layer_hard(bell_state, return_collapsed_state=True)
    out_soft, col_soft, diag_soft = m_layer_soft(bell_state, return_collapsed_state=True)

    print(f"  Hard Collapse Output Shape: {out_hard.shape}")
    print(f"  Hard Collapsed Entropy: {col_hard.measure_entropy()[0, 0].item():.4f}")
    print(f"  Soft Collapse Output Shape: {out_soft.shape}")

    print("\n5. End-to-End Quantum VoidFormer Forward Pass")
    processor = initialize_quantum_processor(
        d_model=d_model,
        n_qubits_per_token=n_qubits,
        collapse_protocol="entropy_gated",
        enable_entanglement=True,
    )
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    output, diagnostics = processor(dummy_input)

    entropy_val = diagnostics.get("final_quantum_entropy", diagnostics.get("initial_quantum_entropy", 0.0))
    if isinstance(entropy_val, torch.Tensor):
        entropy_val = entropy_val.mean().item()

    print(f"  Quantum Processor Output Shape: {output.shape}")
    print(f"  Gates Applied: {diagnostics['gates_applied']}")
    print(f"  Initial Quantum Entropy: {diagnostics['initial_quantum_entropy']:.4f}")
    print(f"  Final Quantum Entropy  : {entropy_val:.4f}")

    print("\n" + "=" * 70)
    print("                  DEMO COMPLETED SUCCESSFULLY!                     ")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
