"""Real Quantum Backend Integration Interface.

Provides conversion tools to translate VoidFormer QuantumCircuit objects into Qiskit QuantumCircuit
objects and interface with Qiskit Aer simulators or IBM Quantum real hardware backends.
"""

from __future__ import annotations

from typing import Optional, Any
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit_aer import AerSimulator

from .quantum_processor import QuantumCircuit


class BackendIntegration:
    """Interface for running VoidFormer quantum circuits on Qiskit Aer or real QPUs."""

    @staticmethod
    def to_qiskit_circuit(circuit: QuantumCircuit) -> QiskitCircuit:
        """Convert a VoidFormer QuantumCircuit into a native Qiskit QuantumCircuit.

        Args:
            circuit: VoidFormer QuantumCircuit object

        Returns:
            Qiskit QuantumCircuit object
        """
        qc = QiskitCircuit(circuit.n_qubits)

        for gate_name, targets, params in circuit.gates:
            if gate_name == "H":
                qc.h(targets[0])
            elif gate_name == "X":
                qc.x(targets[0])
            elif gate_name == "Y":
                qc.y(targets[0])
            elif gate_name == "Z":
                qc.z(targets[0])
            elif gate_name == "CNOT":
                qc.cx(targets[0], targets[1])
            elif gate_name == "Toffoli":
                qc.mcx([targets[0], targets[1]], targets[2])
            elif gate_name.startswith("Phase") or gate_name == "P":
                angle = params.get("angle", 0.0)
                qc.p(angle, targets[0])
            elif gate_name.startswith("CPhase") or gate_name == "CP":
                angle = params.get("angle", 0.0)
                qc.cp(angle, targets[0], targets[1])
            elif gate_name == "MEASURE":
                qc.measure_all()

        return qc

    @classmethod
    def execute_on_aer(
        cls,
        circuit: QuantumCircuit,
        shots: int = 1024,
    ) -> dict[str, int]:
        """Execute a VoidFormer circuit on Qiskit Aer simulator.

        Args:
            circuit: VoidFormer QuantumCircuit
            shots: Number of measurement shots

        Returns:
            Dictionary of outcome bitstring counts
        """
        qc = cls.to_qiskit_circuit(circuit)
        if not any(inst.operation.name == "measure" for inst in qc.data):
            qc.measure_all()

        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        return result.get_counts(qc)
