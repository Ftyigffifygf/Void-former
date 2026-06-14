"""Virtual Quantum Processor and Circuit Execution Engine.

This module provides the top-level quantum computing simulator that orchestrates:
- Quantum circuit construction and execution
- Multi-gate quantum algorithms
- Quantum state management across processing layers
- Integration with neural network pipelines

The Virtual Quantum Processor serves as the "quantum CPU" for the Voidformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn

from .qubit_state import QuantumStateVector, QubitStateManager
from .quantum_gates import QuantumGate, QuantumGateRegistry
from .entanglement import EntanglementManager, BellStateGenerator
from .measurement import MeasurementLayer, CollapseProtocol


class QuantumAlgorithm(Enum):
    """Pre-defined quantum algorithms that can be executed."""
    GROVER_SEARCH = "grover"           # Amplitude amplification for search
    QUANTUM_FOURIER_TRANSFORM = "qft"  # QFT for frequency analysis
    VARIATIONAL_EIGENSOLVER = "vqe"    # VQE for optimization
    QUANTUM_PHASE_ESTIMATION = "qpe"   # Phase estimation
    CUSTOM = "custom"                   # User-defined circuit


@dataclass
class QuantumCircuit:
    """Representation of a quantum circuit as a sequence of gate operations.
    
    A quantum circuit defines a computational graph of unitary operations
    that transform an initial quantum state to a final state.
    
    Circuit = [Gate₁, Gate₂, ..., Gateₙ]
    |ψ_out⟩ = Gateₙ · ... · Gate₂ · Gate₁ · |ψ_in⟩
    """
    
    name: str
    n_qubits: int
    gates: List[Tuple[str, List[int], dict]] = field(default_factory=list)
    # Each gate tuple: (gate_name, target_qubits, gate_params)
    
    def add_gate(
        self,
        gate_name: str,
        target_qubits: List[int],
        params: Optional[dict] = None,
    ):
        """Add a gate operation to the circuit.
        
        Args:
            gate_name: Name of the gate (H, CNOT, X, etc.)
            target_qubits: Which qubits to apply gate to
            params: Optional gate parameters (e.g., rotation angles)
        """
        params = params or {}
        self.gates.append((gate_name, target_qubits, params))
    
    def add_hadamard(self, qubit: int):
        """Add Hadamard gate to create superposition."""
        self.add_gate("H", [qubit])
    
    def add_cnot(self, control: int, target: int):
        """Add CNOT gate for entanglement."""
        self.add_gate("CNOT", [control, target])
    
    def add_rotation(self, qubit: int, axis: str, angle: float):
        """Add rotation gate (parameterized)."""
        self.add_gate(f"R{axis.upper()}", [qubit], {"angle": angle})
    
    def add_measurement(self, qubits: Optional[List[int]] = None):
        """Add measurement operation (all qubits if not specified)."""
        qubits = qubits or list(range(self.n_qubits))
        self.add_gate("MEASURE", qubits)
    
    def depth(self) -> int:
        """Return circuit depth (number of sequential gate layers)."""
        return len(self.gates)
    
    def gate_count(self) -> dict:
        """Count gates by type."""
        counts = {}
        for gate_name, _, _ in self.gates:
            counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(name={self.name}, n_qubits={self.n_qubits}, depth={self.depth()})"


class VirtualQuantumProcessor(nn.Module):
    """Virtual Quantum Computing Processor.
    
    Central orchestration module that:
    1. Manages quantum state vectors for token sequences
    2. Executes quantum circuits and algorithms
    3. Handles entanglement across tokens/features
    4. Performs quantum measurements to collapse to classical outputs
    
    This is the "quantum CPU" that transforms classical neural network
    activations into quantum superpositions, processes them through
    quantum circuits, and measures back to classical space.
    """
    
    def __init__(
        self,
        d_model: int,
        n_qubits_per_token: int = 4,
        max_seq_len: int = 512,
        collapse_protocol: CollapseProtocol = CollapseProtocol.ENTROPY_GATED,
        enable_entanglement: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits_per_token
        self.state_dim = 2 ** n_qubits_per_token
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        self.enable_entanglement = enable_entanglement
        
        # Core quantum components
        self.qubit_manager = QubitStateManager(
            n_qubits_per_token=n_qubits_per_token,
            device=self.device,
        )
        
        self.gate_registry = QuantumGateRegistry(device=self.device)
        
        self.measurement_layer = MeasurementLayer(
            n_qubits=n_qubits_per_token,
            d_output=d_model,
            collapse_protocol=collapse_protocol,
        )
        
        if enable_entanglement:
            self.entanglement_manager = EntanglementManager(
                n_qubits_per_token=n_qubits_per_token,
                max_seq_len=max_seq_len,
                device=self.device,
            )
            self.bell_generator = BellStateGenerator(device=self.device)
        
        # Learnable classical ↔ quantum projections
        self.classical_to_quantum = nn.Parameter(
            torch.randn(d_model, self.state_dim, dtype=torch.complex64)
            / (d_model ** 0.5)
        )
        
        self.quantum_to_classical = nn.Parameter(
            torch.randn(self.state_dim, d_model, dtype=torch.complex64)
            / (self.state_dim ** 0.5)
        )
        
        # Quantum processing circuits (learnable)
        self.processing_circuit = self._build_default_circuit()
        
        # Statistics tracking
        self.register_buffer("total_quantum_ops", torch.tensor(0))
        self.register_buffer("total_measurements", torch.tensor(0))
    
    def _build_default_circuit(self) -> QuantumCircuit:
        """Build default quantum processing circuit.
        
        Circuit structure:
        1. Hadamard layer (create superposition)
        2. Entangling layer (CNOT cascade)
        3. Rotation layer (parameterized gates)
        4. Measurement
        """
        circuit = QuantumCircuit(
            name="default_processor",
            n_qubits=self.n_qubits,
        )
        
        # Layer 1: Hadamard on all qubits (superposition)
        for q in range(self.n_qubits):
            circuit.add_hadamard(q)
        
        # Layer 2: Entangling CNOTs
        for q in range(self.n_qubits - 1):
            circuit.add_cnot(control=q, target=q + 1)
        
        # Layer 3: Rotations (will be parameterized)
        for q in range(self.n_qubits):
            circuit.add_rotation(q, axis="Z", angle=0.0)
        
        return circuit
    
    def encode_classical_to_quantum(
        self,
        classical_tensor: torch.Tensor,  # (B, T, d_model)
    ) -> QuantumStateVector:
        """Phase 1: Encode classical embeddings into quantum superposition states.
        
        Maps: ℝ^d_model → ℂ^(2^n) (Hilbert space)
        
        Each classical token embedding becomes a quantum state vector
        existing in superposition across 2^n computational basis states.
        """
        quantum_state = self.qubit_manager.from_classical_embedding(
            classical_tensor,
            projection_matrix=self.classical_to_quantum,
        )
        
        return quantum_state
    
    def execute_quantum_circuit(
        self,
        quantum_state: QuantumStateVector,
        circuit: Optional[QuantumCircuit] = None,
    ) -> Tuple[QuantumStateVector, dict]:
        """Phase 2: Execute quantum circuit on the state.
        
        Applies sequence of unitary quantum gates to transform the state
        through quantum superposition and entanglement.
        
        Args:
            quantum_state: Input quantum state
            circuit: Quantum circuit to execute (uses default if None)
        
        Returns:
            Transformed quantum state, execution diagnostics
        """
        circuit = circuit or self.processing_circuit
        current_state = quantum_state
        
        diagnostics = {
            "gates_applied": 0,
            "entanglement_created": False,
            "max_entropy": 0.0,
        }
        
        for gate_name, target_qubits, params in circuit.gates:
            if gate_name == "MEASURE":
                continue  # Skip measurement in circuit execution
            
            # Retrieve gate from registry
            try:
                gate = self.gate_registry.get_gate(gate_name)
            except KeyError:
                continue  # Skip unknown gates
            
            # Apply gate
            current_state = gate.apply(current_state, target_qubits)
            diagnostics["gates_applied"] += 1
            
            # Track entropy evolution
            entropy = current_state.measure_entropy().max()
            diagnostics["max_entropy"] = max(diagnostics["max_entropy"], entropy.item())
            
            # Check if gate creates entanglement
            if gate_name in ["CNOT", "Toffoli", "CPhase"]:
                diagnostics["entanglement_created"] = True
        
        self.total_quantum_ops += diagnostics["gates_applied"]
        
        return current_state, diagnostics
    
    def apply_entanglement_layer(
        self,
        quantum_state: QuantumStateVector,
        temperature: float = 1.0,
    ) -> Tuple[QuantumStateVector, dict]:
        """Phase 3: Create entanglement between token positions.
        
        Applies learned entanglement patterns to create quantum correlations
        between tokens (similar to attention but quantum).
        
        This allows tokens to share quantum information non-locally.
        """
        if not self.enable_entanglement or not hasattr(self, 'entanglement_manager'):
            return quantum_state, {"entanglement_applied": False}
        
        entangled_state, entangle_attn = self.entanglement_manager.apply_learned_entanglement(
            quantum_state,
            temperature=temperature,
            top_k=min(5, quantum_state.amplitudes.shape[1]),
        )
        
        # Measure entanglement strength
        is_entangled, entangle_measure = self.entanglement_manager.verify_entanglement(
            entangled_state
        )
        
        diagnostics = {
            "entanglement_applied": True,
            "is_entangled": is_entangled,
            "entanglement_measure": entangle_measure.mean().item(),
            "entanglement_attention_entropy": -(
                entangle_attn.clamp(min=1e-10) * 
                torch.log2(entangle_attn.clamp(min=1e-10))
            ).sum(dim=-1).mean().item(),
        }
        
        return entangled_state, diagnostics
    
    def measure_quantum_to_classical(
        self,
        quantum_state: QuantumStateVector,
    ) -> Tuple[torch.Tensor, dict]:
        """Phase 4: Collapse quantum state to classical output.
        
        Performs quantum measurement following Born rule:
            P(outcome_i) = |⟨i|ψ⟩|² = |αᵢ|²
        
        This is the final stage where quantum superposition collapses
        to deterministic classical values for network output.
        
        Args:
            quantum_state: Quantum state in superposition
        
        Returns:
            Classical tensor (B, T, d_model), measurement diagnostics
        """
        classical_output, collapsed_state, measurement_diag = self.measurement_layer(
            quantum_state,
            return_collapsed_state=True,
        )
        
        self.total_measurements += 1
        
        # Calculate measurement back-action (how much did measurement disturb state)
        if collapsed_state is not None:
            back_action = self.measurement_layer.compute_measurement_back_action(
                quantum_state, collapsed_state
            )
            measurement_diag["measurement_back_action"] = back_action.mean().item()
        
        return classical_output, measurement_diag
    
    def forward(
        self,
        classical_input: torch.Tensor,  # (B, T, d_model)
        circuit: Optional[QuantumCircuit] = None,
        apply_entanglement: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Complete quantum processing pipeline.
        
        Pipeline:
        1. Classical → Quantum encoding (superposition)
        2. Quantum circuit execution (unitary gates)
        3. Entanglement layer (quantum correlations)
        4. Measurement → Classical output (collapse)
        
        Args:
            classical_input: Classical token embeddings (B, T, d_model)
            circuit: Optional custom quantum circuit
            apply_entanglement: Whether to apply inter-token entanglement
        
        Returns:
            Classical output (B, T, d_model), full diagnostics
        """
        diagnostics = {}
        
        # Phase 1: Encode
        quantum_state = self.encode_classical_to_quantum(classical_input)
        diagnostics["initial_quantum_entropy"] = quantum_state.measure_entropy().mean().item()
        
        # Phase 2: Execute quantum circuit
        quantum_state, circuit_diag = self.execute_quantum_circuit(quantum_state, circuit)
        diagnostics.update(circuit_diag)
        diagnostics["post_circuit_entropy"] = quantum_state.measure_entropy().mean().item()
        
        # Phase 3: Entanglement
        if apply_entanglement and self.enable_entanglement:
            quantum_state, entangle_diag = self.apply_entanglement_layer(quantum_state)
            diagnostics.update(entangle_diag)
            diagnostics["post_entanglement_entropy"] = quantum_state.measure_entropy().mean().item()
        
        # Phase 4: Measure
        classical_output, measure_diag = self.measure_quantum_to_classical(quantum_state)
        diagnostics.update(measure_diag)
        
        # Global statistics
        diagnostics["total_quantum_ops"] = self.total_quantum_ops.item()
        diagnostics["total_measurements"] = self.total_measurements.item()
        
        return classical_output, diagnostics
    
    def execute_algorithm(
        self,
        algorithm: QuantumAlgorithm,
        classical_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Execute a pre-defined quantum algorithm.
        
        Args:
            algorithm: Which quantum algorithm to run
            classical_input: Input data
            **kwargs: Algorithm-specific parameters
        
        Returns:
            Classical output, diagnostics
        """
        if algorithm == QuantumAlgorithm.GROVER_SEARCH:
            return self._grover_search(classical_input, **kwargs)
        elif algorithm == QuantumAlgorithm.QUANTUM_FOURIER_TRANSFORM:
            return self._quantum_fourier_transform(classical_input, **kwargs)
        elif algorithm == QuantumAlgorithm.VARIATIONAL_EIGENSOLVER:
            return self._variational_eigensolver(classical_input, **kwargs)
        else:
            return self.forward(classical_input)
    
    def _grover_search(
        self,
        classical_input: torch.Tensor,
        target_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Grover's algorithm: Amplitude amplification for search.
        
        Amplifies the amplitude of a target state through repeated
        inversion about average and oracle operations.
        
        Speedup: O(√N) vs O(N) classical search
        """
        # Encode to quantum
        quantum_state = self.encode_classical_to_quantum(classical_input)
        
        # Number of Grover iterations: ~π/4 * √(2^n)
        n_iterations = int((math.pi / 4) * (self.state_dim ** 0.5))
        
        for _ in range(n_iterations):
            # Oracle: flip phase of target state
            if target_index is not None:
                oracle_mask = torch.ones_like(quantum_state.amplitudes)
                oracle_mask[..., target_index] *= -1
                quantum_state.amplitudes = quantum_state.amplitudes * oracle_mask
            
            # Diffusion: inversion about average
            avg = quantum_state.amplitudes.mean(dim=-1, keepdim=True)
            quantum_state.amplitudes = 2 * avg - quantum_state.amplitudes
            quantum_state = quantum_state.normalize()
        
        # Measure
        classical_output, diagnostics = self.measure_quantum_to_classical(quantum_state)
        diagnostics["algorithm"] = "grover"
        diagnostics["iterations"] = n_iterations
        
        return classical_output, diagnostics
    
    def _quantum_fourier_transform(
        self,
        classical_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Quantum Fourier Transform: Maps computational basis to frequency basis.
        
        QFT|x⟩ = (1/√2^n) Σ_y e^(2πixy/2^n) |y⟩
        
        Used for phase estimation and quantum algorithms.
        """
        quantum_state = self.encode_classical_to_quantum(classical_input)
        
        # Simplified QFT using FFT on probability amplitudes
        # Full QFT requires cascade of controlled rotations
        fft_amplitudes = torch.fft.fft(quantum_state.amplitudes, dim=-1)
        fft_amplitudes = fft_amplitudes / (self.state_dim ** 0.5)
        
        quantum_state.amplitudes = fft_amplitudes
        quantum_state = quantum_state.normalize()
        
        classical_output, diagnostics = self.measure_quantum_to_classical(quantum_state)
        diagnostics["algorithm"] = "qft"
        
        return classical_output, diagnostics
    
    def _variational_eigensolver(
        self,
        classical_input: torch.Tensor,
        n_iterations: int = 5,
    ) -> Tuple[torch.Tensor, dict]:
        """Variational Quantum Eigensolver (VQE): Find ground state energy.
        
        Iteratively optimize parameterized quantum circuit to minimize
        expectation value ⟨ψ(θ)|H|ψ(θ)⟩.
        """
        quantum_state = self.encode_classical_to_quantum(classical_input)
        
        # Placeholder: would train circuit parameters to minimize energy
        # For now, apply default circuit multiple times
        for _ in range(n_iterations):
            quantum_state, _ = self.execute_quantum_circuit(quantum_state)
        
        classical_output, diagnostics = self.measure_quantum_to_classical(quantum_state)
        diagnostics["algorithm"] = "vqe"
        diagnostics["vqe_iterations"] = n_iterations
        
        return classical_output, diagnostics
    
    def get_quantum_state_diagnostics(
        self,
        classical_input: torch.Tensor,
    ) -> dict:
        """Get detailed quantum state information without measurement.
        
        Returns diagnostics about quantum state properties:
        - Entropy, purity, coherence
        - Entanglement measures
        - Probability distributions
        """
        quantum_state = self.encode_classical_to_quantum(classical_input)
        
        return {
            "entropy": quantum_state.measure_entropy().mean().item(),
            "max_probability": quantum_state.probabilities.max(dim=-1).values.mean().item(),
            "state_dimension": self.state_dim,
            "is_normalized": quantum_state.is_normalized,
            "global_phase": quantum_state.global_phase,
        }


import math  # For Grover iterations
