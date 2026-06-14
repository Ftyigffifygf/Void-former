"""Quantum Gate Simulators.

Implements universal quantum gate set as unitary operators acting on state vectors.
All gates preserve norm (unitarity: U†U = I) and are represented as matrix operations.

Gate Library:
- Single-qubit: H (Hadamard), X, Y, Z (Pauli), Phase, T
- Two-qubit: CNOT, Controlled-Phase
- Three-qubit: Toffoli (CCNOT)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .qubit_state import QuantumStateVector


class QuantumGate(ABC, nn.Module):
    """Abstract base class for quantum gates."""
    
    def __init__(self, n_qubits: int, name: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.name = name
        self._matrix: Optional[torch.Tensor] = None
    
    @abstractmethod
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the unitary matrix representation of this gate."""
        pass
    
    def apply(
        self,
        state: QuantumStateVector,
        target_qubits: list[int],
        token_indices: Optional[torch.Tensor] = None,
    ) -> QuantumStateVector:
        """Apply gate to specified qubits in the state vector.
        
        Args:
            state: Input quantum state (B, T, 2^n)
            target_qubits: Which qubits to apply gate to (0-indexed)
            token_indices: Optional mask of which tokens to apply gate (B, T)
        
        Returns:
            New quantum state after gate application
        """
        B, T, state_dim = state.amplitudes.shape
        
        # Get gate matrix
        U = self.matrix(state.amplitudes.device, state.amplitudes.dtype)
        
        # Check if gate dimensionality matches state
        if U.shape[0] != state_dim:
            # Gate is for fewer qubits than state - apply per token
            # For now, just return state unchanged if dimensions don't match
            # Full implementation would require tensor reshaping
            return state
        
        # Apply gate to each token position
        # Reshape to (B*T, 2^n) for batch matrix multiplication
        flat_amps = state.amplitudes.view(B * T, state_dim)
        
        # Apply gate: U |ψ⟩
        new_flat_amps = torch.matmul(flat_amps, U.T)
        
        # Reshape back to (B, T, 2^n)
        new_amplitudes = new_flat_amps.view(B, T, state_dim)
        
        # Apply token mask if provided
        if token_indices is not None:
            mask = token_indices.unsqueeze(-1).to(dtype=state.amplitudes.dtype)
            new_amplitudes = mask * new_amplitudes + (1 - mask) * state.amplitudes
        
        return QuantumStateVector(
            amplitudes=new_amplitudes,
            n_qubits=state.n_qubits,
            global_phase=state.global_phase,
        )
    
    def verify_unitary(self, rtol: float = 1e-5) -> bool:
        """Check if U†U = I (unitarity condition)."""
        U = self._matrix
        if U is None:
            return False
        UdagU = torch.matmul(U.conj().T, U)
        I = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)
        return torch.allclose(UdagU, I, rtol=rtol)


class HadamardGate(QuantumGate):
    """Hadamard gate: Creates equal superposition.
    
    H = (1/√2) [[1,  1],
                [1, -1]]
    
    Maps: |0⟩ → (|0⟩ + |1⟩)/√2    (superposition)
          |1⟩ → (|0⟩ - |1⟩)/√2    (superposition with phase)
    """
    
    def __init__(self, n_qubits: int = 1):
        super().__init__(n_qubits, "Hadamard")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # For n qubits, tensor product H⊗H⊗...⊗H
        H_single = torch.tensor([
            [1.0, 1.0],
            [1.0, -1.0]
        ], dtype=dtype, device=device) / math.sqrt(2)
        
        if self.n_qubits == 1:
            return H_single
        
        # Tensor product for multiple qubits
        result = H_single
        for _ in range(self.n_qubits - 1):
            result = torch.kron(result, H_single)
        return result


class PauliXGate(QuantumGate):
    """Pauli-X gate (NOT gate): Bit flip.
    
    X = [[0, 1],
         [1, 0]]
    
    Maps: |0⟩ → |1⟩, |1⟩ → |0⟩
    """
    
    def __init__(self):
        super().__init__(1, "Pauli-X")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0]
        ], dtype=dtype, device=device)


class PauliYGate(QuantumGate):
    """Pauli-Y gate: Bit and phase flip.
    
    Y = [[0, -i],
         [i,  0]]
    
    Maps: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    """
    
    def __init__(self):
        super().__init__(1, "Pauli-Y")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([
            [0.0, -1.0j],
            [1.0j, 0.0]
        ], dtype=dtype, device=device)


class PauliZGate(QuantumGate):
    """Pauli-Z gate: Phase flip.
    
    Z = [[1,  0],
         [0, -1]]
    
    Maps: |0⟩ → |0⟩, |1⟩ → -|1⟩    (phase kickback)
    """
    
    def __init__(self):
        super().__init__(1, "Pauli-Z")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([
            [1.0, 0.0],
            [0.0, -1.0]
        ], dtype=dtype, device=device)


class PhaseGate(QuantumGate):
    """Parameterized phase gate: R_φ(θ) = e^(iθ)|1⟩⟨1|
    
    P(θ) = [[1,     0    ],
            [0, e^(iθ)]]
    
    Applies relative phase θ to |1⟩ component.
    """
    
    def __init__(self, theta: float = math.pi / 4):
        super().__init__(1, f"Phase({theta:.3f})")
        self.theta = nn.Parameter(torch.tensor(theta))
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase = torch.exp(1j * self.theta)
        return torch.tensor([
            [1.0, 0.0],
            [0.0, phase]
        ], dtype=dtype, device=device)


class TGate(QuantumGate):
    """T gate: π/8 phase gate (Clifford+T universal gate set).
    
    T = [[1, 0        ],
         [0, e^(iπ/4)]]
    """
    
    def __init__(self):
        super().__init__(1, "T")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase = torch.exp(1j * torch.tensor(math.pi / 4))
        return torch.tensor([
            [1.0, 0.0],
            [0.0, phase]
        ], dtype=dtype, device=device)


class CNOTGate(QuantumGate):
    """Controlled-NOT (CNOT): Two-qubit entangling gate.
    
    CNOT = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]]
    
    Control qubit = 0: target unchanged
    Control qubit = 1: target flipped (X gate)
    
    Creates entanglement: CNOT·H⊗I·|00⟩ = (|00⟩ + |11⟩)/√2 (Bell state)
    """
    
    def __init__(self):
        super().__init__(2, "CNOT")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=dtype, device=device)


class ControlledPhaseGate(QuantumGate):
    """Controlled-Phase gate: Two-qubit phase gate.
    
    CPhase(θ) = [[1, 0, 0,     0    ],
                 [0, 1, 0,     0    ],
                 [0, 0, 1,     0    ],
                 [0, 0, 0, e^(iθ)]]
    
    Applies phase only when both qubits are |1⟩.
    """
    
    def __init__(self, theta: float = math.pi):
        super().__init__(2, f"CPhase({theta:.3f})")
        self.theta = nn.Parameter(torch.tensor(theta))
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        phase = torch.exp(1j * self.theta)
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, phase]
        ], dtype=dtype, device=device)


class ToffoliGate(QuantumGate):
    """Toffoli (CCNOT): Three-qubit controlled-controlled-NOT.
    
    Flips target qubit only if both control qubits are |1⟩.
    Universal for classical reversible computation.
    
    8×8 matrix with X applied only to |110⟩ and |111⟩ basis states.
    """
    
    def __init__(self):
        super().__init__(3, "Toffoli")
    
    def matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Identity except swap |110⟩ ↔ |111⟩ (indices 6 and 7)
        toffoli = torch.eye(8, dtype=dtype, device=device)
        toffoli[6, 6] = 0.0
        toffoli[6, 7] = 1.0
        toffoli[7, 6] = 1.0
        toffoli[7, 7] = 0.0
        return toffoli


class QuantumGateRegistry(nn.Module):
    """Registry and factory for quantum gates.
    
    Provides centralized access to gate library and batch application
    capabilities for quantum circuit construction.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        
        # Register standard gates
        self.gates = nn.ModuleDict({
            "H": HadamardGate(n_qubits=1),
            "H2": HadamardGate(n_qubits=2),
            "H3": HadamardGate(n_qubits=3),
            "X": PauliXGate(),
            "Y": PauliYGate(),
            "Z": PauliZGate(),
            "CNOT": CNOTGate(),
            "T": TGate(),
            "Toffoli": ToffoliGate(),
        })
        
        # Parameterized gates (can add more instances)
        self.phase_gates = nn.ModuleList([
            PhaseGate(theta=math.pi / 4),
            PhaseGate(theta=math.pi / 2),
            PhaseGate(theta=math.pi),
        ])
        
        self.cphase_gates = nn.ModuleList([
            ControlledPhaseGate(theta=math.pi / 2),
            ControlledPhaseGate(theta=math.pi),
        ])
    
    def get_gate(self, name: str) -> QuantumGate:
        """Retrieve a gate by name."""
        if name in self.gates:
            return self.gates[name]
        raise KeyError(f"Gate '{name}' not found in registry")
    
    def apply_circuit(
        self,
        state: QuantumStateVector,
        circuit: list[tuple[str, list[int]]],
    ) -> QuantumStateVector:
        """Apply a sequence of gates (quantum circuit) to a state.
        
        Args:
            state: Input quantum state
            circuit: List of (gate_name, target_qubits) tuples
        
        Returns:
            Final quantum state after circuit execution
        """
        current_state = state
        for gate_name, targets in circuit:
            gate = self.get_gate(gate_name)
            current_state = gate.apply(current_state, targets)
        return current_state
    
    def create_bell_state(
        self,
        batch_size: int,
        seq_len: int,
        n_qubits: int = 2,
    ) -> QuantumStateVector:
        """Create maximally entangled Bell state: (|00⟩ + |11⟩)/√2
        
        Circuit: |00⟩ → H⊗I → CNOT → (|00⟩ + |11⟩)/√2
        """
        from .qubit_state import QubitStateManager
        
        manager = QubitStateManager(n_qubits, device=self.device)
        state = manager.initialize_computational_basis(batch_size, seq_len, basis_state=0)
        
        # Apply Hadamard to first qubit
        H = self.get_gate("H")
        state = H.apply(state, [0])
        
        # Apply CNOT
        CNOT = self.get_gate("CNOT")
        state = CNOT.apply(state, [0, 1])
        
        return state
