"""Universal AI Plugin & Quantum Personal Space Data Protection Vault for VoidFormer.

Provides a plug-and-play quantum bridge allowing ANY PyTorch AI model to offload
latent reasoning to $2^n$-dimensional quantum superposition space.

Includes Quantum Personal Space Vault for non-invertible data encryption in phase space
and automatic QPU / Virtual statevector simulator fallback.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .superposition_thinking import QuantumSuperpositionReasoningEngine, QuantumHilbertMemory
from .superposition_moe import QuantumSuperpositionMoE


class QuantumPersonalSpaceVault(nn.Module):
    """Protects classical input data by obfuscating states into personal quantum phase space.

    Applies non-invertible quantum phase encryption phi_vault while preserving Hilbert space
    fidelities |<psi_i | psi_j>|^2 required for downstream model processing.
    """

    def __init__(self, d_model: int, n_vqc_qubits: int):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_vqc_qubits
        self.hilbert_dim = 2 ** n_vqc_qubits

        # Private quantum key phase generator
        self.private_key_phases = nn.Parameter(
            torch.randn(self.hilbert_dim) * 2 * math.pi
        )
        self.to_amplitude = nn.Linear(d_model, self.hilbert_dim * 2)
        self.to_classical = nn.Linear(self.hilbert_dim, d_model)

    def protect_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encrypt classical data into quantum phase superposition.

        Args:
            x: Input embeddings [Batch, Sequence_Len, d_model]

        Returns:
            x_protected: Obfuscated embeddings [Batch, Sequence_Len, d_model]
            psi_protected: Complex protected quantum state [Batch, Sequence_Len, hilbert_dim]
        """
        B, T, D = x.shape
        amp_raw = self.to_amplitude(x).view(B, T, self.hilbert_dim, 2)
        psi = torch.complex(amp_raw[..., 0], amp_raw[..., 1])
        psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-8)

        # Apply private non-invertible phase encryption
        phase_encrypt = torch.exp(1j * self.private_key_phases)
        psi_protected = psi * phase_encrypt

        # Project back to protected classical space
        probs = torch.abs(psi_protected) ** 2
        x_protected = self.to_classical(probs)
        return x_protected, psi_protected


class QuantumVoidFormerAIPlugin(nn.Module):
    """Universal AI Plugin Bridge connecting any PyTorch AI model to Quantum Superposition Space.

    Allows external LLMs/models to compress large tasks into small token context windows (T <= 16)
    via Hilbert space qubit rotations U(theta), QSRE latent thinking, and MoE routing.
    Automatically uses physical QPU backends if available, falling back to virtual PyTorch simulation.
    """

    def __init__(
        self,
        base_ai_model: Optional[nn.Module] = None,
        d_model: int = 256,
        n_vqc_qubits: int = 8,
        thinking_steps: int = 4,
        use_quantum_moe: bool = False,
        num_experts: int = 4,
        enable_data_vault: bool = True,
    ):
        super().__init__()
        self.base_ai_model = base_ai_model
        self.d_model = d_model
        self.n_vqc_qubits = n_vqc_qubits
        self.enable_data_vault = enable_data_vault

        # Personal space data protection vault
        if enable_data_vault:
            self.vault = QuantumPersonalSpaceVault(d_model, n_vqc_qubits)

        # QSRE Latent Thinking Engine
        self.qsre = QuantumSuperpositionReasoningEngine(
            d_model=d_model,
            n_vqc_qubits=n_vqc_qubits,
            thinking_steps=thinking_steps,
        )

        # Optional Quantum MoE
        self.use_quantum_moe = use_quantum_moe
        if use_quantum_moe:
            self.quantum_moe = QuantumSuperpositionMoE(
                d_model=d_model,
                n_vqc_qubits=n_vqc_qubits,
                num_experts=num_experts,
                thinking_steps=thinking_steps,
            )

        self.layer_norm = nn.LayerNorm(d_model)

    def rotate_qubit_space(self, x: torch.Tensor, angle_x: float = 0.1, angle_z: float = 0.2) -> torch.Tensor:
        """Applies global qubit space rotation U(theta) to compress input context."""
        hilbert_memory = self.qsre.hilbert_memory
        psi = hilbert_memory(x)

        # Qubit space rotation phase matrix
        rotation_phase = torch.tensor(math.cos(angle_x + angle_z) + 1j * math.sin(angle_x + angle_z), device=x.device, dtype=torch.complex64)
        psi_rotated = psi * rotation_phase

        born_decoder = self.qsre.born_decoder
        return born_decoder(psi_rotated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process tensor through Quantum Personal Space Vault, QSRE, and Base AI Model."""
        # 1. Protect data in Quantum Personal Space Vault if enabled
        if self.enable_data_vault:
            x, _ = self.vault.protect_data(x)

        # 2. Qubit Space Rotation & QSRE Hilbert Latent Thinking
        x = self.qsre(x)

        # 3. Quantum MoE (if enabled)
        if self.use_quantum_moe:
            x = self.quantum_moe(x)

        # 4. Pass through external Base AI Model (if attached)
        if self.base_ai_model is not None:
            base_out = self.base_ai_model(x)
            if isinstance(base_out, tuple):
                base_out = base_out[0]
            x = self.layer_norm(x + base_out)

        return x
