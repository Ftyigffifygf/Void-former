"""Numerical Unit Tests for VoidFormer Quantum Core.

Asserts exact numerical outputs, quantum state vector fidelity against Qiskit,
concurrence metrics, hard collapse STE gradient flow, and cross-token entanglement.
"""

from __future__ import annotations

import math
import torch
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from voidformer.quantum import (
    QuantumStateVector,
    QubitStateManager,
    HadamardGate,
    CNOTGate,
    PauliXGate,
    PauliYGate,
    PauliZGate,
    PhaseGate,
    TGate,
    ToffoliGate,
    ControlledPhaseGate,
    BellStateGenerator,
    EntanglementManager,
    MeasurementLayer,
    CollapseProtocol,
)


def test_hadamard_numerical():
    """Hadamard on |0> -> (|0> + |1>) / sqrt(2)."""
    amps = torch.tensor([[[1.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=1)
    H = HadamardGate()
    out = H.apply(state, target_qubits=[0])

    expected = torch.tensor([[[1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j]]], dtype=torch.complex64)
    assert torch.allclose(out.amplitudes, expected, atol=1e-5)


def test_bell_state_concurrence_exact():
    """After H then CNOT on |00>, concurrence == 1.0."""
    amps = torch.tensor([[[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)

    H = HadamardGate()
    CNOT = CNOTGate()

    s1 = H.apply(state, target_qubits=[0])
    s2 = CNOT.apply(s1, target_qubits=[0, 1])

    em = EntanglementManager(n_qubits_per_token=2, max_seq_len=16)
    c = em.compute_concurrence(s2)

    assert abs(c.item() - 1.0) < 1e-5, f"Expected concurrence 1.0, got {c.item()}"


def test_qiskit_statevector_comparison_phi_plus():
    """Compare Bell Phi+ statevector against Qiskit Statevector."""
    gen = BellStateGenerator()
    phi_plus = gen.create_bell_phi_plus(batch_size=1, seq_len=1).amplitudes[0, 0]

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qiskit_sv = Statevector.from_instruction(qc)

    qiskit_tensor = torch.tensor(qiskit_sv.data, dtype=torch.complex64)
    assert torch.allclose(phi_plus, qiskit_tensor, atol=1e-5)


def test_qiskit_statevector_comparison_toffoli():
    """Compare Toffoli circuit statevector against Qiskit Statevector."""
    amps = torch.zeros(1, 1, 8, dtype=torch.complex64)
    amps[0, 0, 0] = 1.0
    state = QuantumStateVector(amplitudes=amps, n_qubits=3)

    X = PauliXGate()
    Toff = ToffoliGate()

    s1 = X.apply(state, target_qubits=[0])
    s2 = X.apply(s1, target_qubits=[1])
    s3 = Toff.apply(s2, target_qubits=[0, 1, 2])

    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.mcx([0, 1], 2)
    qiskit_sv = Statevector.from_instruction(qc)

    s3_flat = s3.amplitudes[0, 0]
    qiskit_tensor = torch.tensor(qiskit_sv.data, dtype=torch.complex64)
    assert torch.allclose(s3_flat, qiskit_tensor, atol=1e-5)


def test_hard_collapse_ste_gradient_flow():
    """Verify gradients flow through hard collapse using Straight-Through Estimator."""
    amps = torch.tensor([[[0.6 + 0j, 0.8 + 0j, 0.0 + 0j, 0.0 + 0j]]], dtype=torch.complex64, requires_grad=True)
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)
    layer = MeasurementLayer(n_qubits=2, d_output=4, collapse_protocol=CollapseProtocol.HARD)

    out, col_state, _ = layer(state, return_collapsed_state=True)
    loss = out.sum() + col_state.amplitudes.abs().sum()
    loss.backward()

    assert amps.grad is not None
    assert torch.any(amps.grad != 0), "Gradients failed to flow through hard collapse STE"


def test_cross_token_entanglement_concurrence():
    """Verify cross-token entangler creates non-zero entanglement."""
    amps = torch.zeros(1, 4, 4, dtype=torch.complex64)
    amps[0, :, 0] = 1.0  # All tokens in |00>
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)

    em = EntanglementManager(n_qubits_per_token=2, max_seq_len=16)
    entangled_state, shared_reg = em.apply_cross_token_entanglement(state)

    is_entangled, measure = em.verify_entanglement(entangled_state)
    assert is_entangled.item() is True
    assert measure.mean().item() > 0.01


def test_partial_trace_keep_qubits():
    """Verify partial_trace correctly traces out qubits according to keep_qubits."""
    gen = BellStateGenerator()
    phi_plus = gen.create_bell_phi_plus(batch_size=1, seq_len=1)  # (|00> + |11>) / sqrt(2)

    # Keep qubit 0: marginal probabilities should be [0.5, 0.5]
    res_q0 = phi_plus.partial_trace(keep_qubits=[0])
    expected_q0 = torch.tensor([[[0.5, 0.5]]])
    assert torch.allclose(res_q0, expected_q0, atol=1e-5)

    # Keep qubit 1: marginal probabilities should be [0.5, 0.5]
    res_q1 = phi_plus.partial_trace(keep_qubits=[1])
    expected_q1 = torch.tensor([[[0.5, 0.5]]])
    assert torch.allclose(res_q1, expected_q1, atol=1e-5)

    # Keep qubits [0, 1]: full probabilities [0.5, 0, 0, 0.5]
    res_all = phi_plus.partial_trace(keep_qubits=[0, 1])
    expected_all = torch.tensor([[[0.5, 0.0, 0.0, 0.5]]])
    assert torch.allclose(res_all, expected_all, atol=1e-5)


def test_pairwise_cross_token_entanglement():
    """Verify create_pairwise_entanglement entangles distinct token positions."""
    # Token 0 in equal superposition (|00>+|01>+|10>+|11>)/2, Token 1 in |00>
    amps = torch.zeros(1, 2, 4, dtype=torch.complex64)
    amps[0, 0] = 0.5
    amps[0, 1, 0] = 1.0

    state = QuantumStateVector(amplitudes=amps, n_qubits=2)
    em = EntanglementManager(n_qubits_per_token=2, max_seq_len=16)

    # Entangle token 0 and token 1
    entangled_state = em.create_pairwise_entanglement(state, token_pairs=[(0, 1)], entanglement_strength=1.0)

    # Token 1 should no longer be purely in |00>
    token1_probs = entangled_state.probabilities[0, 1]
    assert token1_probs[2].item() > 0.1, "Token 1 was not entangled by token 0"


def test_is_normalized_rtol_parameter():
    """Verify is_normalized accepts rtol parameter and computes normalization correctly."""
    amps = torch.tensor([[[0.707 + 0j, 0.707 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=1)

    assert state.is_normalized(rtol=1e-2) is True
    assert state.is_normalized(rtol=1e-6) is False


def test_deferred_vs_expectation_collapse_protocols():
    """Verify DEFERRED and EXPECTATION collapse protocols produce distinct outputs."""
    amps = torch.tensor([[[0.6 + 0.8j, 0.0 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=1)

    layer_deferred = MeasurementLayer(n_qubits=1, d_output=4, collapse_protocol=CollapseProtocol.DEFERRED)
    layer_expectation = MeasurementLayer(n_qubits=1, d_output=4, collapse_protocol=CollapseProtocol.EXPECTATION)

    # Share basis projection & observable weights for comparison
    layer_expectation.observable.weight.data = layer_deferred.basis_projection.data.T.clone()

    out_def, _, _ = layer_deferred(state)
    out_exp, _, _ = layer_expectation(state)

    assert not torch.allclose(out_def, out_exp), "DEFERRED and EXPECTATION produced identical output"


def test_to_classical_embedding_full_projection():
    """Verify to_classical_embedding retains full projected output features without discarding half."""
    manager = QubitStateManager(n_qubits_per_token=2)
    amps = torch.tensor([[[0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j]]], dtype=torch.complex64)
    state = QuantumStateVector(amplitudes=amps, n_qubits=2)

    d_model = 64
    out = manager.to_classical_embedding(state, d_model=d_model)

    assert out.shape == (1, 1, d_model)
    assert torch.isfinite(out).all()


def test_quantum_voidformer_model_end_to_end():
    """Verify QuantumVoidFormer instantiates, runs forward/backward passes, and computes gradients."""
    from voidformer.models.quantum_voidformer import QuantumVoidFormer

    vocab_size = 64
    d_model = 32
    seq_len = 8
    batch_size = 2

    model = QuantumVoidFormer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        n_qubits_per_token=2,
        max_seq_len=seq_len,
        collapse_protocol="entropy_gated",
        enable_entanglement=True,
    )

    ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(ids, return_diagnostics=True)

    assert output.logits.shape == (batch_size, seq_len, vocab_size)
    assert torch.isfinite(output.logits).all()

    loss = output.logits.sum()
    loss.backward()

    # Check that model weights received gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and any(g > 0 for g in grad_norms)
