"""Unit tests for IBMBackend local simulator bridge."""

from __future__ import annotations

import pytest
from voidformer.quantum.ibm_backend import IBMBackend, build_circuit


def test_ibm_backend_simulator_bell_circuit():
    bell_ops = [("h", 0), ("cx", 0, 1)]
    backend = IBMBackend(use_simulator=True, shots=1000)

    counts = backend.run(2, bell_ops)
    assert isinstance(counts, dict)
    assert len(counts) > 0

    probs = backend.probabilities(2, bell_ops)
    assert probs.shape == (4,)
    # Probabilities for |00> and |11> should be around 0.5
    assert abs(probs[0] - 0.5) < 0.15
    assert abs(probs[3] - 0.5) < 0.15
