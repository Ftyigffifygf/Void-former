"""Unit tests for IBMQuantumBackend in voidformer/quantum/ibm_backend.py."""

import pytest
import numpy as np
import torch

from voidformer.quantum.ibm_backend import IBMBackend, build_circuit
from voidformer.quantum_init import get_backend


def test_build_circuit():
    ops = [("h", 0), ("cx", 0, 1), ("rx", 0, 0.5)]
    qc = build_circuit(n_qubits=2, ops=ops, measure=True)
    assert qc.num_qubits == 2
    assert qc.num_clbits == 2


def test_ibm_backend_simulator_counts():
    backend = IBMBackend(use_simulator=True, shots=500)
    bell_ops = [("h", 0), ("cx", 0, 1)]
    counts = backend.run(n_qubits=2, ops=bell_ops)
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 500
    assert "00" in counts or "11" in counts


def test_ibm_backend_probabilities():
    backend = IBMBackend(use_simulator=True, shots=1000)
    bell_ops = [("h", 0), ("cx", 0, 1)]
    probs = backend.probabilities(n_qubits=2, ops=bell_ops)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 4
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0


def test_get_backend_factory():
    sim_backend = get_backend("simulator")
    assert sim_backend is None

    aer_backend = get_backend("ibm_aer")
    assert isinstance(aer_backend, IBMBackend)
    assert aer_backend.use_simulator is True
