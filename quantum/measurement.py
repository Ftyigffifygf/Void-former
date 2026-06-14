"""Measurement Layer - Quantum to Classical Collapse.

Implements the measurement postulate of quantum mechanics:
- States remain in superposition during processing
- Measurement collapses |ψ⟩ to classical outcome following Born rule: P(i) = |αᵢ|²
- Post-measurement state projects onto measured eigenstate

Collapse Protocols:
1. Hard Collapse: Sample from distribution, project to basis state
2. Soft Collapse: Weighted mixture preserving some quantum information
3. Deferred Collapse: Keep quantum until final output layer
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qubit_state import QuantumStateVector


class CollapseProtocol(Enum):
    """Strategy for collapsing quantum superposition to classical state."""
    HARD = "hard"              # Full collapse to single basis state (Born rule sampling)
    SOFT = "soft"              # Partial collapse (weighted by probabilities)
    EXPECTATION = "expectation"  # Collapse to expectation value
    DEFERRED = "deferred"      # No collapse (keep quantum state)
    ENTROPY_GATED = "entropy_gated"  # Adaptive collapse based on entropy


class MeasurementLayer(nn.Module):
    """Quantum Measurement Layer.
    
    Converts quantum superposition states to classical deterministic or
    probabilistic outputs. Implements measurement postulate with various
    collapse strategies.
    
    This is the bridge between quantum processing and classical output:
    quantum layers → measurement → classical logits/embeddings
    """
    
    def __init__(
        self,
        n_qubits: int,
        d_output: int,
        collapse_protocol: CollapseProtocol = CollapseProtocol.SOFT,
        temperature: float = 1.0,
        entropy_threshold: float = 0.5,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.d_output = d_output
        self.collapse_protocol = collapse_protocol
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.entropy_threshold = entropy_threshold
        
        # Learnable measurement basis (observable operator)
        # In quantum mechanics: ⟨O⟩ = ⟨ψ|O|ψ⟩
        self.observable = nn.Linear(self.state_dim, d_output, bias=True)
        
        # Projection matrix for basis state → classical embedding
        self.basis_projection = nn.Parameter(
            torch.randn(self.state_dim, d_output) / (self.state_dim ** 0.5)
        )
    
    def forward(
        self,
        quantum_state: QuantumStateVector,
        return_collapsed_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[QuantumStateVector], dict]:
        """Measure quantum state and produce classical output.
        
        Args:
            quantum_state: Input quantum state (B, T, 2^n)
            return_collapsed_state: If True, also return post-measurement quantum state
        
        Returns:
            classical_output: (B, T, d_output)
            collapsed_state: Post-measurement quantum state (or None)
            diagnostics: Measurement statistics
        """
        B, T, _ = quantum_state.amplitudes.shape
        
        # Compute Born rule probabilities
        probs = quantum_state.probabilities  # (B, T, 2^n)
        
        # Von Neumann entropy for adaptive collapse
        entropy = quantum_state.measure_entropy()  # (B, T)
        
        # Select collapse protocol
        if self.collapse_protocol == CollapseProtocol.DEFERRED:
            # No collapse - convert quantum to classical via expectation
            classical_output, collapsed_state = self._expectation_value(quantum_state)
        
        elif self.collapse_protocol == CollapseProtocol.HARD:
            # Full collapse via Born rule sampling
            classical_output, collapsed_state = self._hard_collapse(quantum_state, probs)
        
        elif self.collapse_protocol == CollapseProtocol.SOFT:
            # Soft collapse - probability-weighted mixture
            classical_output, collapsed_state = self._soft_collapse(quantum_state, probs)
        
        elif self.collapse_protocol == CollapseProtocol.EXPECTATION:
            # Expectation value (no actual collapse)
            classical_output, collapsed_state = self._expectation_value(quantum_state)
        
        elif self.collapse_protocol == CollapseProtocol.ENTROPY_GATED:
            # Adaptive: hard collapse if low entropy, soft if high entropy
            classical_output, collapsed_state = self._entropy_gated_collapse(
                quantum_state, probs, entropy
            )
        
        else:
            raise ValueError(f"Unknown collapse protocol: {self.collapse_protocol}")
        
        diagnostics = {
            "measurement_entropy": entropy,
            "max_probability": probs.max(dim=-1).values,
            "effective_dimension": torch.exp(entropy),  # e^S = effective # of states
            "collapse_protocol": self.collapse_protocol.value,
        }
        
        if not return_collapsed_state:
            collapsed_state = None
        
        return classical_output, collapsed_state, diagnostics
    
    def _hard_collapse(
        self,
        state: QuantumStateVector,
        probs: torch.Tensor,
    ) -> tuple[torch.Tensor, QuantumStateVector]:
        """Sample from Born rule distribution and project to basis state.
        
        Post-measurement state: |ψ⟩ → |i⟩ with probability |αᵢ|²
        """
        B, T, state_dim = state.amplitudes.shape
        
        # Sample basis state according to Born rule
        # Flatten to (B*T, 2^n) for categorical sampling
        flat_probs = probs.view(B * T, state_dim)
        sampled_indices = torch.multinomial(flat_probs, num_samples=1)  # (B*T, 1)
        sampled_indices = sampled_indices.view(B, T)
        
        # Project sampled indices to classical output
        # Use basis_projection: one-hot → embedding
        one_hot = F.one_hot(sampled_indices, num_classes=state_dim).float()  # (B, T, 2^n)
        classical_output = torch.matmul(one_hot, self.basis_projection)  # (B, T, d_out)
        
        # Post-measurement quantum state: |ψ⟩ → |i_measured⟩
        collapsed_amplitudes = torch.zeros_like(state.amplitudes)
        for b in range(B):
            for t in range(T):
                collapsed_amplitudes[b, t, sampled_indices[b, t]] = 1.0 + 0.0j
        
        collapsed_state = QuantumStateVector(
            amplitudes=collapsed_amplitudes,
            n_qubits=state.n_qubits,
        )
        
        return classical_output, collapsed_state
    
    def _soft_collapse(
        self,
        state: QuantumStateVector,
        probs: torch.Tensor,
    ) -> tuple[torch.Tensor, QuantumStateVector]:
        """Soft collapse: probability-weighted sum (preserves some quantum info).
        
        Classical output = Σᵢ P(i) · embedding(i)
        Quantum state remains in superposition but with adjusted amplitudes.
        """
        # Temperature-scaled probabilities (softer or harder mixing)
        scaled_probs = F.softmax(
            torch.log(probs.clamp(min=1e-10)) / self.temperature.abs().clamp(min=0.1),
            dim=-1
        )
        
        # Weighted sum over basis states
        classical_output = torch.matmul(scaled_probs, self.basis_projection)
        
        # Soft collapsed state: renormalize with temperature scaling
        soft_amplitudes = state.amplitudes * torch.sqrt(scaled_probs.to(state.amplitudes.dtype))
        collapsed_state = QuantumStateVector(
            amplitudes=soft_amplitudes,
            n_qubits=state.n_qubits,
        ).normalize()
        
        return classical_output, collapsed_state
    
    def _expectation_value(
        self,
        state: QuantumStateVector,
    ) -> tuple[torch.Tensor, QuantumStateVector]:
        """Compute expectation value ⟨ψ|O|ψ⟩ without collapsing state.
        
        Uses learned observable operator O. State remains unchanged.
        """
        # Convert complex amplitudes to real representation
        # Concatenate [real, imag] or use magnitude
        amp_real = torch.cat([
            state.amplitudes.real,
            state.amplitudes.imag,
        ], dim=-1)  # (B, T, 2*2^n)
        
        # Learned observable measurement
        # For simplicity, operate on probabilities
        classical_output = self.observable(state.probabilities)
        
        # State unchanged (no collapse)
        return classical_output, state
    
    def _entropy_gated_collapse(
        self,
        state: QuantumStateVector,
        probs: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, QuantumStateVector]:
        """Adaptive collapse based on quantum entropy.
        
        High entropy (highly uncertain) → soft collapse (preserve superposition)
        Low entropy (nearly collapsed) → hard collapse (commit to answer)
        
        This mirrors the VoidFormer's adaptive collapse philosophy!
        """
        B, T = entropy.shape
        
        # Normalize entropy to [0, 1] range
        max_entropy = math.log2(self.state_dim)
        normalized_entropy = (entropy / max_entropy).clamp(0, 1)
        
        # Compute both hard and soft collapse
        hard_output, hard_state = self._hard_collapse(state, probs)
        soft_output, soft_state = self._soft_collapse(state, probs)
        
        # Blend based on entropy: high entropy → soft, low entropy → hard
        # entropy > threshold → soft collapse (preserve uncertainty)
        # entropy < threshold → hard collapse (commit to measurement)
        soft_weight = (normalized_entropy > self.entropy_threshold).float().unsqueeze(-1)
        
        classical_output = soft_weight * soft_output + (1 - soft_weight) * hard_output
        
        # Blended quantum state
        blend = soft_weight.unsqueeze(-1).to(dtype=state.amplitudes.dtype)
        collapsed_amplitudes = (
            blend * soft_state.amplitudes + 
            (1 - blend) * hard_state.amplitudes
        )
        
        collapsed_state = QuantumStateVector(
            amplitudes=collapsed_amplitudes,
            n_qubits=state.n_qubits,
        ).normalize()
        
        return classical_output, collapsed_state
    
    def set_collapse_protocol(self, protocol: CollapseProtocol):
        """Dynamically change measurement strategy."""
        self.collapse_protocol = protocol
    
    def compute_measurement_back_action(
        self,
        state_pre: QuantumStateVector,
        state_post: QuantumStateVector,
    ) -> torch.Tensor:
        """Quantify measurement back-action: how much did measurement disturb state?
        
        Uses quantum fidelity: F(ψ,φ) = |⟨ψ|φ⟩|²
        F = 1: no disturbance, F = 0: maximally disturbed
        """
        return state_pre.fidelity(state_post)


import math  # For log2 in entropy normalization
