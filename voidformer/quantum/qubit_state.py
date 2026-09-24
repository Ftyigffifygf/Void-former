"""Virtual Qubit State Manager.

Represents quantum states as complex-valued state vectors in Hilbert space:
    |ψ⟩ = α|0⟩ + β|1⟩    where |α|² + |β|² = 1

For n qubits, the state vector lives in 2^n dimensional complex space.
Supports superposition, normalization, and phase tracking.
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
    amplitudes: torch.Tensor  # Complex tensor (B, 2^n)
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

    def is_normalized(self, rtol: float = 1e-4) -> bool:
        """Check if Σ|αᵢ|² ≈ 1"""
        total_prob = self.probabilities.sum(dim=-1)
        return torch.allclose(total_prob, torch.ones_like(total_prob), rtol=rtol)

    def normalize(self) -> QuantumStateVector:
        """Enforce normalization: |ψ⟩ → |ψ⟩/√⟨ψ|ψ⟩"""
        norm = torch.sqrt(self.probabilities.sum(dim=-1, keepdim=True)).clamp(min=1e-10)
        return QuantumStateVector(
            amplitudes=self.amplitudes / (norm + 1e-10),
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
        """Reduced density matrix by tracing out qubits not in keep_qubits.

        Args:
            keep_qubits: List of qubit indices to keep (0 to n_qubits - 1).

        Returns:
            Reduced density matrix tensor of shape (..., 2^k, 2^k) where k = len(keep_qubits).
        """
        n = self.n_qubits
        keep = list(keep_qubits)
        assert all(0 <= q < n for q in keep), f"keep_qubits {keep} out of range for {n} qubits"

        trace = [q for q in range(n) if q not in keep]

        if not keep:
            batch_dims = self.amplitudes.shape[:-1]
            return torch.ones((*batch_dims, 1, 1), device=self.amplitudes.device, dtype=self.amplitudes.dtype)

        if len(keep) == n:
            a = self.amplitudes.unsqueeze(-1)
            return torch.matmul(a, a.mH)

        batch_dims = self.amplitudes.shape[:-1]
        n_batch = len(batch_dims)
        tensor = self.amplitudes.view(*batch_dims, *([2] * n))

        perm = list(range(n_batch)) + [n_batch + q for q in keep] + [n_batch + q for q in trace]
        tensor_perm = tensor.permute(*perm)

        k = len(keep)
        r = len(trace)
        d_k = 2 ** k
        d_r = 2 ** r

        psi_matrix = tensor_perm.reshape(*batch_dims, d_k, d_r)
        return torch.matmul(psi_matrix, psi_matrix.mH)


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
        """Initialize all tokens in a pure computational basis state |basis_state⟩.

        Args:
            batch_size: Number of sequences
            seq_len: Number of token positions
            basis_state: Which computational basis state (0 to 2^n-1)

        Returns:
            QuantumStateVector with all probability mass on |basis_state⟩
        """
        assert 0 <= basis_state < self.state_dim

        # Shape: (B, T, 2^n)
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
        """Initialize tokens in equal superposition: |ψ⟩ = (1/√2^n) Σᵢ |i⟩

        This is equivalent to applying Hadamard gates to all qubits starting from |0⟩.
        """
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
        """Initialize random quantum states on the Bloch sphere.

        Samples random complex amplitudes and normalizes to ensure valid quantum state.
        """
        # Random amplitudes with random phases
        magnitudes = torch.rand(
            batch_size, seq_len, self.state_dim,
            device=self.device,
        )
        phases = torch.rand(
            batch_size, seq_len, self.state_dim,
            device=self.device,
        ) * phase_range

        # Convert to complex: α = r·e^(iφ)
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
        """Convert classical embeddings to quantum state vectors.

        Maps classical d_model dimensional vectors to 2^n dimensional Hilbert space
        through a learned or random projection, then normalizes.

        Args:
            classical_tensor: Classical token embeddings (B, T, d_model)
            projection_matrix: Optional (d_model, 2^n) complex projection

        Returns:
            Normalized quantum state vectors (B, T, 2^n)
        """
        B, T, d_model = classical_tensor.shape

        if projection_matrix is None:
            # Random complex projection (Xavier initialization for quantum states)
            projection_matrix = torch.randn(
                d_model, self.state_dim,
                dtype=self.dtype,
                device=self.device,
            ) / math.sqrt(d_model)

        # Project to Hilbert space: (B, T, d) @ (d, 2^n) -> (B, T, 2^n)
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
        """Convert quantum state back to classical embedding space.

        Projects from 2^n dimensional Hilbert space back to d_model dimensions.
        Preserves quantum information through amplitude encoding.

        Args:
            quantum_state: Quantum state vectors (B, T, 2^n)
            d_model: Target classical dimension
            readout_matrix: Optional complex readout projection matrix

        Returns:
            Classical embeddings (B, T, d_model) [real-valued]
        """
        if d_model % 2 == 0:
            half_d = d_model // 2
            if readout_matrix is None:
                readout_matrix = torch.randn(
                    self.state_dim, half_d,
                    dtype=self.dtype,
                    device=self.device,
                ) / math.sqrt(self.state_dim)
            elif readout_matrix.shape[-1] == d_model:
                readout_matrix = readout_matrix[..., :half_d]

            # Project: (B, T, 2^n) @ (2^n, d//2) -> (B, T, d//2)
            projected = torch.matmul(quantum_state.amplitudes, readout_matrix)
            return torch.cat([projected.real, projected.imag], dim=-1)
        else:
            if readout_matrix is None:
                readout_matrix = torch.randn(
                    self.state_dim, d_model,
                    dtype=self.dtype,
                    device=self.device,
                ) / math.sqrt(self.state_dim)

            # Project: (B, T, 2^n) @ (2^n, d) -> (B, T, d)
            projected = torch.matmul(quantum_state.amplitudes, readout_matrix)
            return torch.abs(projected)[..., :d_model]
