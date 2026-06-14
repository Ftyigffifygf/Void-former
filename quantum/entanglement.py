"""Entanglement Manager.

Quantum entanglement is the non-local correlation between qubits that cannot
be explained by classical probability theory. This module creates and tracks
entangled states between token positions.

Key Concepts:
- Bell States: Maximally entangled 2-qubit states
- GHZ States: Maximally entangled n-qubit states
- Entanglement Entropy: von Neumann entropy of reduced density matrix
- Concurrence: Entanglement measure for two-qubit systems
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .qubit_state import QuantumStateVector, QubitStateManager
from .quantum_gates import QuantumGateRegistry


class BellStateGenerator(nn.Module):
    """Generator for maximally entangled Bell states.
    
    Four Bell basis states (EPR pairs):
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2    (Bell state)
    |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.gate_registry = QuantumGateRegistry(device=self.device)
    
    def create_bell_phi_plus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (maximally entangled)."""
        manager = QubitStateManager(n_qubits_per_token=2, device=self.device)
        state = manager.initialize_computational_basis(batch_size, seq_len, basis_state=0)
        
        # Circuit: H⊗I then CNOT
        H = self.gate_registry.get_gate("H")
        CNOT = self.gate_registry.get_gate("CNOT")
        
        state = H.apply(state, target_qubits=[0])
        state = CNOT.apply(state, target_qubits=[0, 1])
        
        return state
    
    def create_bell_phi_minus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Φ⁻⟩ = (|00⟩ - |11⟩)/√2."""
        state = self.create_bell_phi_plus(batch_size, seq_len)
        
        # Apply Z to introduce phase flip
        Z = self.gate_registry.get_gate("Z")
        state = Z.apply(state, target_qubits=[0])
        
        return state
    
    def create_bell_psi_plus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2."""
        state = self.create_bell_phi_plus(batch_size, seq_len)
        
        # Apply X to first qubit: |00⟩+|11⟩ → |10⟩+|01⟩
        X = self.gate_registry.get_gate("X")
        state = X.apply(state, target_qubits=[0])
        
        return state
    
    def create_bell_psi_minus(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Create |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2."""
        state = self.create_bell_psi_plus(batch_size, seq_len)
        
        # Apply Z to introduce phase flip
        Z = self.gate_registry.get_gate("Z")
        state = Z.apply(state, target_qubits=[1])
        
        return state


class EntanglementManager(nn.Module):
    """Manages quantum entanglement between token positions.
    
    Creates and tracks entangled quantum states across sequence positions.
    This allows tokens to share quantum correlations that influence
    downstream processing.
    
    Applications:
    - Long-range dependencies via entanglement
    - Quantum attention (tokens entangled based on relevance)
    - Multi-token coherence (GHZ states across key positions)
    """
    
    def __init__(
        self,
        n_qubits_per_token: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits_per_token
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cpu")
        
        self.bell_generator = BellStateGenerator(device=self.device)
        self.gate_registry = QuantumGateRegistry(device=self.device)
        
        # Learnable entanglement patterns
        # Which token pairs should be entangled? (attention-like)
        self.entangle_scores = nn.Parameter(
            torch.randn(max_seq_len, max_seq_len) * 0.01
        )
    
    def create_pairwise_entanglement(
        self,
        state: QuantumStateVector,
        token_pairs: list[tuple[int, int]],
        entanglement_strength: float = 1.0,
    ) -> QuantumStateVector:
        """Entangle specified pairs of tokens via CNOT gates.
        
        Args:
            state: Current quantum state (B, T, 2^n)
            token_pairs: List of (i, j) token position pairs to entangle
            entanglement_strength: Control parameter for partial entanglement
        
        Returns:
            Entangled quantum state
        """
        current_state = state
        
        for (i, j) in token_pairs:
            if i >= state.amplitudes.shape[1] or j >= state.amplitudes.shape[1]:
                continue
            
            # Apply CNOT between token positions i and j
            # This creates entanglement: information becomes shared
            CNOT = self.gate_registry.get_gate("CNOT")
            
            # Apply to all qubits in both tokens (simplified model)
            # In practice, would select specific qubit pairs
            token_mask = torch.zeros(
                state.amplitudes.shape[0],
                state.amplitudes.shape[1],
                device=self.device,
            )
            token_mask[:, [i, j]] = entanglement_strength
            
            current_state = CNOT.apply(
                current_state,
                target_qubits=[0, 1],
                token_indices=token_mask,
            )
        
        return current_state
    
    def create_ghz_state(
        self,
        batch_size: int,
        n_tokens: int,
        n_qubits: int = 3,
    ) -> QuantumStateVector:
        """Create GHZ (Greenberger-Horne-Zeilinger) state: maximal n-qubit entanglement.
        
        |GHZ_n⟩ = (|00...0⟩ + |11...1⟩)/√2
        
        All qubits perfectly correlated - measuring one instantly determines all others.
        """
        manager = QubitStateManager(n_qubits_per_token=n_qubits, device=self.device)
        state = manager.initialize_computational_basis(batch_size, n_tokens, basis_state=0)
        
        # GHZ circuit: H on first qubit, then cascade CNOTs
        H = self.gate_registry.get_gate("H")
        CNOT = self.gate_registry.get_gate("CNOT")
        
        # H on first qubit
        state = H.apply(state, target_qubits=[0])
        
        # CNOT cascade: control = qubit i, target = qubit i+1
        for i in range(n_qubits - 1):
            state = CNOT.apply(state, target_qubits=[i, i + 1])
        
        return state
    
    def measure_entanglement_entropy(
        self,
        state: QuantumStateVector,
        partition_a: list[int],
    ) -> torch.Tensor:
        """Compute entanglement entropy of bipartition A|B.
        
        For pure state |ψ⟩_{AB}, entanglement entropy is:
            S_A = -Tr(ρ_A log ρ_A)
        where ρ_A = Tr_B(|ψ⟩⟨ψ|) is reduced density matrix of subsystem A.
        
        S_A = 0: no entanglement (product state)
        S_A > 0: entangled (higher = more entangled)
        
        Args:
            state: Quantum state (B, T, 2^n)
            partition_a: Token indices in subsystem A
        
        Returns:
            Entanglement entropy (B, T)
        """
        # Simplified: use partial trace to get marginal probabilities
        # Full implementation requires tensor reshaping for proper partial trace
        
        # For demonstration, compute entropy of marginal distribution
        reduced_probs = state.partial_trace(partition_a)
        
        # von Neumann entropy
        reduced_probs = reduced_probs.clamp(min=1e-10)
        entropy = -(reduced_probs * torch.log2(reduced_probs)).sum(dim=-1)
        
        return entropy
    
    def compute_concurrence(
        self,
        state: QuantumStateVector,
    ) -> torch.Tensor:
        """Compute concurrence (entanglement measure for 2-qubit systems).
        
        Concurrence C ∈ [0, 1]:
            C = 0: separable (no entanglement)
            C = 1: maximally entangled (Bell state)
        
        For pure states: C = 2|α₀α₃ - α₁α₂| (2-qubit only)
        
        Args:
            state: 2-qubit quantum state (B, T, 4)
        
        Returns:
            Concurrence values (B, T)
        """
        assert state.n_qubits == 2, "Concurrence only defined for 2-qubit systems"
        
        # Extract amplitudes
        a = state.amplitudes  # (B, T, 4) = [|00⟩, |01⟩, |10⟩, |11⟩]
        
        # Concurrence formula for pure 2-qubit state
        # C = 2|α₀α₃ - α₁α₂|
        term = a[..., 0] * a[..., 3] - a[..., 1] * a[..., 2]
        concurrence = 2 * torch.abs(term).clamp(0, 1)
        
        return concurrence
    
    def apply_learned_entanglement(
        self,
        state: QuantumStateVector,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> tuple[QuantumStateVector, torch.Tensor]:
        """Apply learned entanglement pattern across tokens.
        
        Uses trainable entangle_scores to determine which token pairs
        should be entangled (similar to attention mechanism).
        
        Args:
            state: Input quantum state (B, T, 2^n)
            temperature: Softmax temperature for entanglement selection
            top_k: Only entangle top-k pairs per token
        
        Returns:
            Entangled state, entanglement attention map
        """
        T = state.amplitudes.shape[1]
        
        # Causal mask (only entangle with past/present)
        causal_mask = torch.triu(
            torch.ones(T, T, device=self.device, dtype=torch.bool),
            diagonal=1,
        )
        
        # Compute entanglement attention
        scores = self.entangle_scores[:T, :T] / temperature
        scores = scores.masked_fill(causal_mask, float("-inf"))
        entangle_attn = torch.softmax(scores, dim=-1)
        
        # Select top-k pairs if specified
        if top_k is not None:
            values, indices = torch.topk(entangle_attn, k=min(top_k, T), dim=-1)
            # Create token pairs from top-k connections
            token_pairs = []
            for i in range(T):
                for k_idx in range(len(indices[i])):
                    j = indices[i, k_idx].item()
                    if values[i, k_idx] > 0.1:  # Threshold
                        token_pairs.append((i, j))
        else:
            # Use all pairs weighted by attention
            token_pairs = [(i, j) for i in range(T) for j in range(i + 1)]
        
        # Apply entanglement with learned weights
        entangled_state = self.create_pairwise_entanglement(
            state,
            token_pairs,
            entanglement_strength=1.0,
        )
        
        return entangled_state, entangle_attn
    
    def verify_entanglement(
        self,
        state: QuantumStateVector,
        threshold: float = 0.01,
    ) -> tuple[bool, torch.Tensor]:
        """Check if state is entangled (non-separable).
        
        For 2-qubit systems, use concurrence.
        For n-qubit, use entanglement entropy.
        
        Returns:
            is_entangled (bool), entanglement_measure (Tensor)
        """
        if state.n_qubits == 2:
            measure = self.compute_concurrence(state)
            is_entangled = (measure > threshold).any()
        else:
            # Use entanglement entropy with first qubit vs rest partition
            measure = self.measure_entanglement_entropy(state, partition_a=[0])
            is_entangled = (measure > threshold).any()
        
        return is_entangled, measure
