"""Initialization and Registry utilities for Voidformer Quantum Processing Infrastructure.

Provides global initialization functions and runtime registry for
VirtualQuantumProcessor instances connected to Voidformer models.
"""

from __future__ import annotations

from typing import Optional, Dict
import torch

from voidformer.quantum.quantum_processor import VirtualQuantumProcessor
from voidformer.quantum.measurement import CollapseProtocol

# Global registry of quantum processors
_QUANTUM_PROCESSOR_REGISTRY: Dict[str, VirtualQuantumProcessor] = {}


def initialize_quantum_processor(
    d_model: int = 256,
    n_qubits_per_token: int = 4,
    max_seq_len: int = 512,
    collapse_protocol: str = "entropy_gated",
    enable_entanglement: bool = True,
    device: Optional[torch.device] = None,
    name: str = "default_processor",
) -> VirtualQuantumProcessor:
    """Initialize and register a Virtual Quantum Processor for Voidformer.

    Args:
        d_model: Classical embedding dimension
        n_qubits_per_token: Number of qubits allocated per token position
        max_seq_len: Maximum sequence length
        collapse_protocol: Measurement collapse strategy ("hard", "soft", "entropy_gated", etc.)
        enable_entanglement: Enable inter-token entanglement layer
        device: Torch device (CPU/CUDA)
        name: Registry key for this processor

    Returns:
        Initialized VirtualQuantumProcessor instance
    """
    protocol_map = {
        "hard": CollapseProtocol.HARD,
        "soft": CollapseProtocol.SOFT,
        "expectation": CollapseProtocol.EXPECTATION,
        "deferred": CollapseProtocol.DEFERRED,
        "entropy_gated": CollapseProtocol.ENTROPY_GATED,
    }

    protocol = protocol_map.get(collapse_protocol.lower(), CollapseProtocol.ENTROPY_GATED)

    processor = VirtualQuantumProcessor(
        d_model=d_model,
        n_qubits_per_token=n_qubits_per_token,
        max_seq_len=max_seq_len,
        collapse_protocol=protocol,
        enable_entanglement=enable_entanglement,
        device=device,
    )

    _QUANTUM_PROCESSOR_REGISTRY[name] = processor
    return processor


def get_quantum_processor(name: str = "default_processor") -> Optional[VirtualQuantumProcessor]:
    """Retrieve registered quantum processor by name."""
    return _QUANTUM_PROCESSOR_REGISTRY.get(name)


def reset_quantum_registry():
    """Clear registered quantum processors."""
    global _QUANTUM_PROCESSOR_REGISTRY
    _QUANTUM_PROCESSOR_REGISTRY = {}
