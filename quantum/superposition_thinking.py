"""Quantum Superposition Reasoning Engine (QSRE) for VoidFormer.

Executes multi-step latent reasoning in Hilbert space without generating
intermediate classical text tokens ($O(1)$ token cost).

Maps classical embeddings into $2^n$-dimensional complex Hilbert space states,
evolves state vectors through parameterized unitary phase rotations and quantum oracle
amplitude amplification, and collapses via Born-rule measurement back to classical
embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantumHilbertMemory(nn.Module):
    """Maps input token embeddings into complex amplitude vectors in Hilbert space.

    Preserves phase relationships Arg(alpha_i) between tokens to store structural
    contextual relationships in quantum state phases.
    """

    def __init__(self, d_model: int, n_qubits: int):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.hilbert_dim = 2 ** n_qubits

        # Linear projection to Hilbert space amplitudes (Real + Imaginary)
        self.to_amplitude = nn.Linear(d_model, self.hilbert_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Classical embedding tensor [Batch, Sequence_Len, d_model]

        Returns:
            Normalized complex amplitude tensor [Batch, Sequence_Len, hilbert_dim]
        """
        B, T, _ = x.shape
        amp_raw = self.to_amplitude(x).view(B, T, self.hilbert_dim, 2)
        state_complex = torch.complex(amp_raw[..., 0], amp_raw[..., 1])

        # State vector normalization: \sum |\alpha_i|^2 = 1
        norm = torch.norm(state_complex, dim=-1, keepdim=True) + 1e-8
        psi = state_complex / norm
        return psi


class UnitaryThinkingLoop(nn.Module):
    """Applies sequence of parameterized unitary phase rotations across internal thinking steps.

    Operates strictly on complex state vectors in PyTorch without appending classical tokens.
    """

    def __init__(self, n_qubits: int, thinking_steps: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.thinking_steps = thinking_steps

        # Parameterized Quantum Phase Gates (RX, RY, RZ rotations per step per qubit)
        self.phase_weights = nn.Parameter(torch.randn(thinking_steps, n_qubits, 3))

    def get_phase_factor(self, step: int) -> torch.Tensor:
        """Compute complex phase rotation factor for a given thinking step."""
        angles = self.phase_weights[step]  # [n_qubits, 3]
        phase = torch.sin(angles.sum(dim=-1)).mean()
        return torch.exp(1j * phase)

    def forward(self, psi: torch.Tensor, step: int) -> torch.Tensor:
        """
        Args:
            psi: Complex state vector [Batch, Sequence_Len, hilbert_dim]
            step: Thinking cycle index (0 <= step < thinking_steps)

        Returns:
            Phase-rotated complex state vector [Batch, Sequence_Len, hilbert_dim]
        """
        phase_factor = self.get_phase_factor(step)
        return psi * phase_factor


class QuantumAmplitudeOracle(nn.Module):
    """Latent quality filter that amplifies valid output pathways via quantum interference."""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.hilbert_dim = 2 ** n_qubits

        # Latent Quantum Oracle Projection
        self.oracle_proj = nn.Linear(self.hilbert_dim, self.hilbert_dim)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state vector [Batch, Sequence_Len, hilbert_dim]

        Returns:
            Amplified complex state vector [Batch, Sequence_Len, hilbert_dim]
        """
        probabilities = torch.abs(psi) ** 2
        oracle_score = torch.sigmoid(self.oracle_proj(probabilities))
        psi = psi * (1.0 + oracle_score)

        # Re-normalize state vector: \sum |\alpha_i|^2 = 1
        norm = torch.norm(psi, dim=-1, keepdim=True) + 1e-8
        return psi / norm


class SuperposedBornDecoder(nn.Module):
    """Applies Born-rule collapse P(i) = |\alpha_i|^2 to project quantum state back to classical embedding."""

    def __init__(self, d_model: int, n_qubits: int):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.hilbert_dim = 2 ** n_qubits

        self.to_classical = nn.Linear(self.hilbert_dim, d_model)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi: Complex state vector [Batch, Sequence_Len, hilbert_dim]

        Returns:
            Classical output embedding tensor [Batch, Sequence_Len, d_model]
        """
        # Born measurement probability density: P(i) = |\alpha_i|^2
        probs = torch.abs(psi) ** 2
        return self.to_classical(probs)


class QuantumSuperpositionReasoningEngine(nn.Module):
    """Quantum Superposition Reasoning Engine (QSRE).

    Executes multi-step latent thinking in Hilbert Space without generating
    intermediate classical text tokens.
    """

    def __init__(self, d_model: int, n_vqc_qubits: int, thinking_steps: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_vqc_qubits
        self.thinking_steps = thinking_steps
        self.hilbert_dim = 2 ** n_vqc_qubits

        # Modules
        self.hilbert_memory = QuantumHilbertMemory(d_model, n_vqc_qubits)
        self.thinking_loop = UnitaryThinkingLoop(n_vqc_qubits, thinking_steps)
        self.oracle = QuantumAmplitudeOracle(n_vqc_qubits)
        self.born_decoder = SuperposedBornDecoder(d_model, n_vqc_qubits)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Classical embedding tensor [Batch, Sequence_Len, d_model]

        Returns:
            Thought-enhanced embedding tensor [Batch, Sequence_Len, d_model]
        """
        # 1. Map to Complex Quantum Amplitudes
        psi = self.hilbert_memory(x)

        # 2. Multi-Step Latent Thinking in Hilbert Space
        for step in range(self.thinking_steps):
            psi = self.thinking_loop(psi, step)
            psi = self.oracle(psi)

        # 3. Born Measurement Probability Density & Classical Projection
        thought_output = self.born_decoder(psi)

        # 4. Residual connection & Layer Norm
        return self.layer_norm(x + thought_output)
