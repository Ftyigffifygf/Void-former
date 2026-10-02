"""Quantum Superposition Mixture of Experts (MoE) & Quantum Tokenization for VoidFormer.

Implements Quantum Superposition Tokenization and Quantum Superposition MoE
for Argon-inspired quantum state architecture.

Decomposes complex reasoning tasks into parallel sub-tasks in $2^n$-dimensional
Hilbert space across multiple quantum experts with minimal token usage.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .superposition_thinking import (
    QuantumHilbertMemory,
    UnitaryThinkingLoop,
    QuantumAmplitudeOracle,
    SuperposedBornDecoder,
)


class QuantumSuperpositionTokenEmbedder(nn.Module):
    """Maps discrete token IDs into complex quantum superposition state vectors in Hilbert space.

    Directly embeds tokens as complex amplitude distributions with phase factors.
    """

    def __init__(self, vocab_size: int, d_model: int, n_vqc_qubits: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_qubits = n_vqc_qubits
        self.hilbert_dim = 2 ** n_vqc_qubits

        # Embeddings for Real and Imaginary amplitudes
        self.amplitude_embed = nn.Embedding(vocab_size, self.hilbert_dim * 2)
        self.classical_proj = nn.Linear(self.hilbert_dim, d_model)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ids: Token IDs tensor [Batch, Sequence_Len]

        Returns:
            x_classical: Classical embedding [Batch, Sequence_Len, d_model]
            psi: Normalized complex quantum state [Batch, Sequence_Len, hilbert_dim]
        """
        B, T = ids.shape
        raw_embed = self.amplitude_embed(ids)  # [B, T, hilbert_dim * 2]
        raw_embed = raw_embed.view(B, T, self.hilbert_dim, 2)

        state_complex = torch.complex(raw_embed[..., 0], raw_embed[..., 1])
        norm = torch.norm(state_complex, dim=-1, keepdim=True) + 1e-8
        psi = state_complex / norm

        probs = torch.abs(psi) ** 2
        x_classical = self.classical_proj(probs)
        return x_classical, psi


class QuantumSuperpositionExpert(nn.Module):
    """Individual Quantum Expert executing Hilbert space transformations."""

    def __init__(self, d_model: int, n_vqc_qubits: int, thinking_steps: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_vqc_qubits
        self.thinking_steps = thinking_steps
        self.hilbert_dim = 2 ** n_vqc_qubits

        self.hilbert_memory = QuantumHilbertMemory(d_model, n_vqc_qubits)
        self.thinking_loop = UnitaryThinkingLoop(n_vqc_qubits, thinking_steps)
        self.oracle = QuantumAmplitudeOracle(n_vqc_qubits)
        self.born_decoder = SuperposedBornDecoder(d_model, n_vqc_qubits)

        # Expert specific transformation
        self.expert_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process embeddings through quantum expert in Hilbert space."""
        psi = self.hilbert_memory(x)

        for step in range(self.thinking_steps):
            psi = self.thinking_loop(psi, step)
            psi = self.oracle(psi)

        thought_output = self.born_decoder(psi)
        return self.expert_ffn(thought_output)


class QuantumSuperpositionRouter(nn.Module):
    """Fidelity-based Quantum State Router for Superposition MoE."""

    def __init__(self, d_model: int, num_experts: int, n_vqc_qubits: int):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.hilbert_dim = 2 ** n_vqc_qubits

        # Expert Hilbert space centroid state projections (Real + Imaginary)
        self.expert_centroids = nn.Parameter(
            torch.randn(num_experts, self.hilbert_dim, 2)
        )
        self.to_amplitude = nn.Linear(d_model, self.hilbert_dim * 2)

    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings [Batch, Sequence_Len, d_model]
            top_k: Number of experts to select per token

        Returns:
            routing_weights: Normalized probabilities [Batch, Sequence_Len, top_k]
            selected_expert_indices: Selected expert IDs [Batch, Sequence_Len, top_k]
        """
        B, T, D = x.shape
        amp_raw = self.to_amplitude(x).view(B, T, self.hilbert_dim, 2)
        psi_input = torch.complex(amp_raw[..., 0], amp_raw[..., 1])
        psi_input = psi_input / (torch.norm(psi_input, dim=-1, keepdim=True) + 1e-8)

        centroids = torch.complex(self.expert_centroids[..., 0], self.expert_centroids[..., 1])
        centroids = centroids / (torch.norm(centroids, dim=-1, keepdim=True) + 1e-8)

        # Quantum Fidelity Overlap Scores: K(x, e_j) = |\langle \psi(x) | \psi(e_j) \rangle|^2
        fidelity_scores = torch.abs(torch.matmul(psi_input, centroids.conj().T)) ** 2  # [B, T, num_experts]

        top_k = min(top_k, self.num_experts)
        top_k_scores, indices = torch.topk(fidelity_scores, k=top_k, dim=-1)
        routing_weights = F.softmax(top_k_scores, dim=-1)

        return routing_weights, indices


class QuantumSuperpositionMoE(nn.Module):
    """Quantum Superposition Mixture of Experts (MoE) Layer.

    Decomposes complex Hilbert reasoning across multiple parallel Quantum Experts.
    """

    def __init__(
        self,
        d_model: int,
        n_vqc_qubits: int,
        num_experts: int = 4,
        top_k_experts: int = 2,
        thinking_steps: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts

        self.router = QuantumSuperpositionRouter(
            d_model=d_model,
            num_experts=num_experts,
            n_vqc_qubits=n_vqc_qubits,
        )

        self.experts = nn.ModuleList([
            QuantumSuperpositionExpert(
                d_model=d_model,
                n_vqc_qubits=n_vqc_qubits,
                thinking_steps=thinking_steps,
            )
            for _ in range(num_experts)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Classical embedding tensor [Batch, Sequence_Len, d_model]

        Returns:
            MoE-enhanced embedding tensor [Batch, Sequence_Len, d_model]
        """
        B, T, D = x.shape
        routing_weights, indices = self.router(x, top_k=self.top_k_experts)  # [B, T, top_k]

        combined_output = torch.zeros_like(x)

        # Process through top-k experts
        for k in range(self.top_k_experts):
            weight_k = routing_weights[..., k].unsqueeze(-1)  # [B, T, 1]
            expert_idx_k = indices[..., k]  # [B, T]

            # Route tokens to corresponding expert outputs
            expert_outputs = torch.zeros_like(x)
            for expert_i, expert_module in enumerate(self.experts):
                mask = (expert_idx_k == expert_i)  # [B, T]
                if mask.any():
                    # Evaluate expert on all x and mask, or masked elements
                    exp_out = expert_module(x)
                    expert_outputs = torch.where(mask.unsqueeze(-1), exp_out, expert_outputs)

            combined_output = combined_output + weight_k * expert_outputs

        return self.layer_norm(x + combined_output)
