"""IBM Quantum bridge for VoidFormer.

Converts a simple gate list into a Qiskit circuit and runs it either on a local simulator
(no account needed) or on real IBM Quantum hardware.

Gate list format (qubit 0 first):
    ("h", 0)
    ("x", 1)
    ("cx", 0, 1)
    ("ccx", 0, 1, 2)
    ("rx", qubit, theta)
    ("ry", qubit, theta)
    ("rz", qubit, theta)

Bitstring convention:
    Qiskit puts qubit 0 as the RIGHTMOST bit, so index i of the returned
    probability vector has qubit 0 as its least significant bit.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

_ONE_QUBIT = {"h", "x", "y", "z", "s", "t"}
_ROTATIONS = {"rx", "ry", "rz", "p"}


def save_account(
    token: str,
    instance: Optional[str] = None,
    channel: str = "ibm_quantum_platform",
    overwrite: bool = True,
):
    """Save IBM Quantum account credentials locally.

    Args:
        token: 44-character API token from IBM Quantum Platform
        instance: Optional CRN or hub/group/project instance string
        channel: Authentication channel, default 'ibm_quantum_platform'
        overwrite: Whether to overwrite existing saved account
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {"token": token, "channel": channel, "overwrite": overwrite}
    if instance:
        kwargs["instance"] = instance

    QiskitRuntimeService.save_account(**kwargs)
    print("Successfully saved IBM Quantum account credentials.")


def build_circuit(n_qubits: int, ops: Sequence[tuple], measure: bool = True):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for name, *args in ops:
        name = name.lower()
        if name in _ONE_QUBIT:
            qc_method = getattr(qc, name)
            qc_method(args[0])
        elif name in _ROTATIONS:
            if len(args) == 2:
                # Handle both (qubit, theta) and (theta, qubit)
                if isinstance(args[0], (int, np.integer)) and not isinstance(args[1], (int, np.integer)):
                    qubit, theta = args[0], args[1]
                elif isinstance(args[1], (int, np.integer)) and not isinstance(args[0], (int, np.integer)):
                    theta, qubit = args[0], args[1]
                else:
                    qubit, theta = args[0], args[1]
                qc_method = getattr(qc, name)
                qc_method(float(theta), int(qubit))
        elif name in ("cx", "cnot"):
            qc.cx(args[0], args[1])
        elif name == "cz":
            qc.cz(args[0], args[1])
        elif name in ("ccx", "toffoli"):
            qc.ccx(args[0], args[1], args[2])
        else:
            raise ValueError(f"Unsupported gate: {name}")
    if measure:
        qc.measure_all()
    return qc


class IBMBackend:
    def __init__(
        self,
        use_simulator: bool = True,
        backend_name: Optional[str] = None,
        token: Optional[str] = None,
        shots: int = 1024,
    ):
        self.use_simulator = use_simulator
        self.shots = shots
        self.backend_name = backend_name
        self._token = token or os.environ.get("IBM_QUANTUM_TOKEN")
        self._backend = None
        self._service = None

    def _get_backend(self):
        if self._backend is not None:
            return self._backend
        if self.use_simulator:
            from qiskit_aer import AerSimulator

            self._backend = AerSimulator()
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService

            if self._token:
                self._service = QiskitRuntimeService(
                    channel="ibm_quantum_platform", token=self._token
                )
            else:
                self._service = QiskitRuntimeService()  # uses saved account
            if self.backend_name:
                self._backend = self._service.backend(self.backend_name)
            else:
                self._backend = self._service.least_busy(
                    operational=True, simulator=False
                )
        return self._backend

    def run(self, n_qubits: int, ops: Sequence[tuple]) -> dict:
        """Return measurement counts, e.g. {"00": 510, "11": 514}."""
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        backend = self._get_backend()
        qc = build_circuit(n_qubits, ops)
        isa = generate_preset_pass_manager(
            backend=backend, optimization_level=1
        ).run(qc)

        if self.use_simulator:
            return backend.run(isa, shots=self.shots).result().get_counts()

        from qiskit_ibm_runtime import SamplerV2 as Sampler

        job = Sampler(mode=backend).run([isa], shots=self.shots)
        return job.result()[0].data.meas.get_counts()

    def probabilities(self, n_qubits: int, ops: Sequence[tuple]) -> np.ndarray:
        """Estimated Born-rule probabilities, length 2**n_qubits."""
        counts = self.run(n_qubits, ops)
        probs = np.zeros(2 ** n_qubits)
        total = sum(counts.values())
        for bits, c in counts.items():
            probs[int(bits.replace(" ", ""), 2)] = c / total
        return probs


IBMQuantumBackend = IBMBackend


if __name__ == "__main__":
    bell = [("h", 0), ("cx", 0, 1)]
    sim = IBMBackend(use_simulator=True)
    print("Simulator Bell probabilities:", sim.probabilities(2, bell))
