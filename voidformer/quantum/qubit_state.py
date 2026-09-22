"""Virtual Qubit State Manager.

Represents quantum states as complex-valued state vectors in Hilbert space:
    |ψ⟩ = α|0⟩ + β|1⟩    where |α|² + |β|² = 1

For n qubits, the state vector lives in 2^n dimensional complex space.
Supports superposition, normalization, partial trace, and phase tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QuantumStateVector:
    """A quantum state vector with complex amplitudes.

    Attributes:
        amplitudes: Complex tensor of shape (batch, 2^n_qubits) representing |ψ⟩
        n_qubits: Number of qubits in the system
        global_phase: Global phase factor (physically unobservable but tracked)
    """
    amplitudes: torch.Tensor  # Complex tensor (B, ..., 2^n)
    n_qubits: int
    global_phase: float = 0.0

    def __post_init__(self):
        assert self.amplitudes.is_complex(), "Amplitudes must be complex-valued"
        expected_dim = 2 ** self.n_qubits
        assert self.amplitudes.shape[-1] == expected_dim, (
            f"State vector dimension {self.amplitudes.shape[-1]} != 2^{self.n_qubits}"
        )

    @property
    def probabilities(self) -> torch.Tensor:
        """Born rule: P(i) = |⟨i|ψ⟩|² = |αᵢ|²"""
        return (self.amplitudes.real ** 2 + self.amplitudes.imag ** 2).clamp(min=0.0)

    def check_normalized(self, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
        """Check if Σ|αᵢ|² ≈ 1 with specified tolerances."""
        total_prob = self.probabilities.sum(dim=-1)
        return torch.allclose(total_prob, torch.ones_like(total_prob), rtol=rtol, atol=atol)

    @property
    def is_normalized(self) -> bool:
        """Check if Σ|αᵢ|² ≈ 1 (property access)."""
        return self.check_normalized()

    def normalize(self) -> QuantumStateVector:
        """Enforce normalization: |ψ⟩ → |ψ⟩/√⟨ψ|ψ⟩"""
        norm = torch.sqrt(self.probabilities.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
        return QuantumStateVector(
            amplitudes=self.amplitudes / norm,
            n_qubits=self.n_qubits,
            global_phase=self.global_phase,
        )

    def measure_entropy(self) -> torch.Tensor:
        """Von Neumann entropy for pure states: S = -Tr(ρ log ρ) = -Σ pᵢ log pᵢ"""
        probs = self.probabilities.clamp(min=1e-10)
        return -(probs * torch.log2(probs)).sum(dim=-1)

    def fidelity(self, other: QuantumStateVector) -> torch.Tensor:
        """Quantum fidelity: F(ψ,φ) = |⟨ψ|φ⟩|²"""
        assert self.n_qubits == other.n_qubits
        inner = (self.amplitudes.conj() * other.amplitudes).sum(dim=-1)
        return (inner.real ** 2 + inner.imag ** 2).clamp(0.0, 1.0)

    def partial_trace(self, keep_qubits: list[int]) -> torch.Tensor:
        """Reduced density matrix probabilities by tracing out qubits not in keep_qubits.

        Computes exact marginal subsystem probability distribution:
            P_keep(a) = Σ_c |ψ_{a, c}|^2
        where 'a' ranges over basis states of keep_qubits and 'c' ranges over traced-out qubits.

        Args:
            keep_qubits: Qubit indices (0 to n-1) to retain in subsystem

        Returns:
            Reduced probability distribution tensor of shape (..., 2^len(keep_qubits))
        """
        assert all(0 <= q < self.n_qubits for q in keep_qubits), f"Invalid keep_qubits: {keep_qubits}"

        # Reshape amplitudes into multi-qubit tensor: (*batch, 2, 2, ..., 2)
        batch_shape = list(self.amplitudes.shape[:-1])
        multi_qubit_shape = batch_shape + [2] * self.n_qubits
        tensor = self.amplitudes.view(*multi_qubit_shape)

        # Determine permutation: batch axes, keep_qubits, trace_qubits
        n_batch = len(batch_shape)
        trace_qubits = [q for q in range(self.n_qubits) if q not in keep_qubits]

        perm = list(range(n_batch)) + [n_batch + q for q in keep_qubits] + [n_batch + q for q in trace_qubits]
        tensor_perm = tensor.permute(*perm)

        k = len(keep_qubits)
        m = len(trace_qubits)

        dim_keep = 2 ** k
        dim_trace = 2 ** m

        flat = tensor_perm.reshape(*batch_shape, dim_keep, dim_trace)

        # Sum squared magnitudes over traced-out qubit axes: Σ_c |ψ_{a, c}|^2
        probs_keep = (flat.real ** 2 + flat.imag ** 2).sum(dim=-1)  # (*batch, 2^k)
        return probs_keep.clamp(min=0.0)


class QubitStateManager(nn.Module):
    """Manages quantum state vectors for a batch of token sequences.

    Each token position can be in a superposition state represented as a
    multi-qubit quantum state vector. Provides initialization in computational
    basis states and superposition states.
    """

    def __init__(
        self,
        n_qubits_per_token: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.n_qubits = n_qubits_per_token
        self.state_dim = 2 ** n_qubits_per_token
        self.device = device
        self.dtype = dtype

    def initialize_computational_basis(
        self,
        batch_size: int,
        seq_len: int,
        basis_state: int = 0,
    ) -> QuantumStateVector:
        """Initialize all tokens in a pure computational basis state |basis_state⟩."""
        assert 0 <= basis_state < self.state_dim

        amplitudes = torch.zeros(
            batch_size, seq_len, self.state_dim,
            dtype=self.dtype,
            device=self.device,
        )
        amplitudes[:, :, basis_state] = 1.0 + 0.0j

        return QuantumStateVector(
            amplitudes=amplitudes,
            n_qubits=self.n_qubits,
        )

    def initialize_uniform_superposition(
        self,
        batch_size: int,
        seq_len: int,
    ) -> QuantumStateVector:
        """Initialize tokens in equal superposition: |ψ⟩ = (1/√2^n) Σᵢ |i⟩"""
        coeff = 1.0 / math.sqrt(self.state_dim)
        amplitudes = torch.full(
            (batch_size, seq_len, self.state_dim),
            coeff + 0.0j,
            dtype=self.dtype,
            device=self.device,
        )

        return QuantumStateVector(
            amplitudes=amplitudes,
            n_qubits=self.n_qubits,
        )

    def initialize_random_superposition(
        self,
        batch_size: int,
        seq_len: int,
        phase_range: float = 2 * math.pi,
    ) -> QuantumStateVector:
        """Initialize random quantum states on the Bloch sphere."""
        magnitudes = torch.rand(
            batch_size, seq_len, self.state_dim,
            device=self.device,
        )
        phases = torch.rand(
            batch_size, seq_len, self.state_dim,
            device=self.device,
        ) * phase_range

        amplitudes = magnitudes * torch.exp(1j * phases)
        amplitudes = amplitudes.to(self.dtype)

        state = QuantumStateVector(
            amplitudes=amplitudes,
            n_qubits=self.n_qubits,
        )
        return state.normalize()

    def from_classical_embedding(
        self,
        classical_tensor: torch.Tensor,  # (B, T, d_model)
        projection_matrix: Optional[torch.Tensor] = None,
    ) -> QuantumStateVector:
        """Convert classical embeddings to quantum state vectors."""
        B, T, d_model = classical_tensor.shape

        if projection_matrix is None:
            projection_matrix = torch.randn(
                d_model, self.state_dim,
                dtype=self.dtype,
                device=self.device,
            ) / math.sqrt(d_model)

        classical_complex = classical_tensor.to(dtype=self.dtype)
        amplitudes = torch.matmul(classical_complex, projection_matrix)

        state = QuantumStateVector(
            amplitudes=amplitudes,
            n_qubits=self.n_qubits,
        )
        return state.normalize()

    def to_classical_embedding(
        self,
        quantum_state: QuantumStateVector,
        d_model: int,
        readout_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert quantum state back to classical embedding space using 100% of readout parameters.

        Args:
            quantum_state: Quantum state vectors (B, T, 2^n)
            d_model: Target classical dimension
            readout_matrix: Optional readout projection matrix

        Returns:
            Classical embeddings (B, T, d_model) [real-valued]
        """
        if readout_matrix is None:
            if d_model % 2 == 0:
                # Readout matrix projects to (2^n, d_model//2) complex, concatenated to d_model real
                readout_matrix = torch.randn(
                    self.state_dim, d_model // 2,
                    dtype=self.dtype,
                    device=self.device,
                ) / math.sqrt(self.state_dim)
            else:
                readout_matrix = torch.randn(
                    self.state_dim, d_model,
                    dtype=self.dtype,
                    device=self.device,
                ) / math.sqrt(self.state_dim)

        projected = torch.matmul(quantum_state.amplitudes, readout_matrix)

        if d_model % 2 == 0 and projected.shape[-1] == d_model // 2:
            return torch.cat([projected.real, projected.imag], dim=-1)
        else:
            return torch.abs(projected)[..., :d_model]
