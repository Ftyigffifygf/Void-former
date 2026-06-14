"""Quantum Computing Simulator Core for Voidformer.

This module provides a virtual quantum processor infrastructure that operates
on high-dimensional state vectors with superposition, entanglement, and 
quantum gate operations.
"""

from .qubit_state import QubitStateManager, QuantumStateVector
from .quantum_gates import (
    QuantumGateRegistry,
    HadamardGate,
    CNOTGate,
    PauliXGate,
    PauliYGate,
    PauliZGate,
    PhaseGate,
    TGate,
    ToffoliGate,
    ControlledPhaseGate,
)
from .measurement import MeasurementLayer, CollapseProtocol
from .quantum_processor import VirtualQuantumProcessor, QuantumCircuit, QuantumAlgorithm
from .entanglement import EntanglementManager, BellStateGenerator
from .qiml import (
    QuantumInspiredNeuralLayer,
    TensorNetworkLayer,
    QuantumEvolutionaryOptimizer,
    QuantumKernelAttention,
)
from .temporal_coherence import (
    QuantumClock,
    CoherenceWindow,
    DecoherenceModel,
    DecoherenceMetrics,
    TimeAwareQuantumProcessor,
    create_time_aware_processor,
)

__all__ = [
    "QubitStateManager",
    "QuantumStateVector",
    "QuantumGateRegistry",
    "HadamardGate",
    "CNOTGate",
    "PauliXGate",
    "PauliYGate",
    "PauliZGate",
    "PhaseGate",
    "TGate",
    "ToffoliGate",
    "ControlledPhaseGate",
    "MeasurementLayer",
    "CollapseProtocol",
    "VirtualQuantumProcessor",
    "QuantumCircuit",
    "QuantumAlgorithm",
    "EntanglementManager",
    "BellStateGenerator",
    "QuantumInspiredNeuralLayer",
    "TensorNetworkLayer",
    "QuantumEvolutionaryOptimizer",
    "QuantumKernelAttention",
    "QuantumClock",
    "CoherenceWindow",
    "DecoherenceModel",
    "DecoherenceMetrics",
    "TimeAwareQuantumProcessor",
    "create_time_aware_processor",
]
