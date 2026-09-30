"""Real Quantum Backend Integration Interface.

Provides conversion tools to translate VoidFormer QuantumCircuit objects into Qiskit QuantumCircuit
objects and interface with Qiskit Aer simulators or IBM Quantum real hardware backends.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Any, Union
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit_aer import AerSimulator

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    HAS_IBM_RUNTIME = True
except ImportError:
    HAS_IBM_RUNTIME = False

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
            elif gate_name == "RX" or gate_name.startswith("RX"):
                angle = params.get("angle", 0.0)
                if hasattr(angle, "item"):
                    angle = angle.item()
                qc.rx(float(angle), targets[0])
            elif gate_name == "RY" or gate_name.startswith("RY"):
                angle = params.get("angle", 0.0)
                if hasattr(angle, "item"):
                    angle = angle.item()
                qc.ry(float(angle), targets[0])
            elif gate_name == "RZ" or gate_name.startswith("RZ"):
                angle = params.get("angle", 0.0)
                if hasattr(angle, "item"):
                    angle = angle.item()
                qc.rz(float(angle), targets[0])
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
        """Execute a VoidFormer circuit on Qiskit Aer simulator."""
        qc = cls.to_qiskit_circuit(circuit)
        if not any(inst.operation.name == "measure" for inst in qc.data):
            qc.measure_all()

        simulator = AerSimulator()
        job = simulator.run(qc, shots=shots)
        result = job.result()
        return result.get_counts(qc)

    @classmethod
    def execute_batch_on_aer(
        cls,
        circuits: list[Union[QuantumCircuit, QiskitCircuit]],
        shots: int = 1024,
    ) -> list[dict[str, int]]:
        """Execute a batch of quantum circuits on Qiskit Aer simulator in a single payload."""
        qc_list = []
        for circuit in circuits:
            if isinstance(circuit, QuantumCircuit):
                qc = cls.to_qiskit_circuit(circuit)
            else:
                qc = circuit
            if not any(inst.operation.name == "measure" for inst in qc.data):
                qc.measure_all()
            qc_list.append(qc)

        simulator = AerSimulator()
        job = simulator.run(qc_list, shots=shots)
        result = job.result()

        if len(qc_list) == 1:
            return [result.get_counts(qc_list[0])]
        else:
            counts_list = result.get_counts()
            return counts_list if isinstance(counts_list, list) else [counts_list]

    @classmethod
    def get_ibm_service(
        cls,
        token: Optional[str] = None,
        instance: Optional[str] = None,
        channel: str = "ibm_quantum",
    ) -> Any:
        """Authenticate and return QiskitRuntimeService instance."""
        if not HAS_IBM_RUNTIME:
            raise ImportError(
                "qiskit-ibm-runtime is not installed. Install via pip install qiskit-ibm-runtime"
            )

        token = token or os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("IBMQ_TOKEN")

        kwargs = {"channel": channel}
        if token:
            kwargs["token"] = token
        if instance:
            kwargs["instance"] = instance

        try:
            service = QiskitRuntimeService(**kwargs)
            return service
        except Exception as e:
            raise RuntimeError(f"Failed to authenticate with IBM Quantum backend: {e}") from e

    @classmethod
    def execute_on_ibm_hardware(
        cls,
        circuit: Union[QuantumCircuit, QiskitCircuit],
        backend_name: str = "ibm_brisbane",
        token: Optional[str] = None,
        instance: Optional[str] = None,
        shots: int = 1024,
    ) -> Any:
        """Submit a quantum circuit to real IBM Quantum hardware via qiskit-ibm-runtime."""
        if isinstance(circuit, QuantumCircuit):
            qc = cls.to_qiskit_circuit(circuit)
        else:
            qc = circuit

        if not any(inst.operation.name == "measure" for inst in qc.data):
            qc.measure_all()

        service = cls.get_ibm_service(token=token, instance=instance)
        backend = service.backend(backend_name)

        sampler = Sampler(mode=backend)
        job = sampler.run([qc], shots=shots)
        return job

    @classmethod
    def execute_batch_on_ibm_hardware(
        cls,
        circuits: list[Union[QuantumCircuit, QiskitCircuit]],
        backend_name: str = "ibm_brisbane",
        token: Optional[str] = None,
        instance: Optional[str] = None,
        shots: int = 1024,
        timeout_seconds: float = 300.0,
    ) -> list[dict[str, int]]:
        """Submit a batch of circuits representing (B, T) sequence items to real IBM Quantum hardware."""
        qc_list = []
        for circuit in circuits:
            if isinstance(circuit, QuantumCircuit):
                qc = cls.to_qiskit_circuit(circuit)
            else:
                qc = circuit
            if not any(inst.operation.name == "measure" for inst in qc.data):
                qc.measure_all()
            qc_list.append(qc)

        service = cls.get_ibm_service(token=token, instance=instance)
        backend = service.backend(backend_name)

        sampler = Sampler(mode=backend)
        job = sampler.run(qc_list, shots=shots)
        result = cls.poll_job(job, timeout_seconds=timeout_seconds)

        counts_list = []
        for pub_result in result:
            counts = pub_result.data.meas.get_counts()
            counts_list.append(counts)

        return counts_list

    @classmethod
    def poll_job(
        cls,
        job: Any,
        timeout_seconds: float = 300.0,
        poll_interval: float = 1.0,
    ) -> Any:
        """Poll job status asynchronously until completion or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                status = job.status()
                status_str = str(status).lower()
            except Exception:
                status_str = "unknown"

            if any(term in status_str for term in ("done", "completed")):
                return job.result()
            elif any(err in status_str for err in ("error", "failed", "cancelled")):
                raise RuntimeError(f"IBM Quantum Job failed with status: {status_str}")

            time.sleep(poll_interval)

        try:
            job.cancel()
        except Exception:
            pass
        raise TimeoutError(
            f"IBM Quantum Job polling timed out after {timeout_seconds} seconds"
        )
