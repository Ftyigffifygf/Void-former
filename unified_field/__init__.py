"""Unified Field Theory in Computing - Virtual Machine.

A grand unification framework that integrates:
1. Classical Computing (deterministic logic)
2. Quantum Computing (superposition, entanglement)
3. Probabilistic Computing (stochastic processes)
4. Neural Computing (adaptive learning)
5. Temporal Computing (time-evolution)

All paradigms unified under a single computational field theory.

THEORETICAL FOUNDATION:
    The unified computational field Ψ(x,t) encompasses all computational
    paradigms as different excitation modes of the same underlying field.
    
    Master Equation: iℏ ∂Ψ/∂t = Ĥ_unified · Ψ
    
    This allows a SINGLE virtual machine to execute programs across
    classical, quantum, probabilistic, neural, and temporal paradigms
    seamlessly, with automatic mode transitions.
"""

from .field_theory import (
    UnifiedComputationalField,
    FieldMode,
    FieldTransition,
)

from .universal_vm import (
    UniversalVirtualMachine,
    ComputationalState,
    ExecutionContext,
    Instruction,
    InstructionType,
    create_universal_vm,
)

from .field_operators import (
    ClassicalOperator,
    QuantumOperator,
    ProbabilisticOperator,
    NeuralOperator,
    TemporalOperator,
    UnifiedOperator,
    QuantumGate,
    ClassicalLogicGate,
)

from .field_equations import (
    FieldEquations,
    HamiltonianEvolution,
    LagrangianDynamics,
    InformationFlow,
    FieldConfiguration,
    compute_field_configuration,
)

__all__ = [
    # Field Theory
    "UnifiedComputationalField",
    "FieldMode",
    "FieldTransition",
    
    # Virtual Machine
    "UniversalVirtualMachine",
    "ComputationalState",
    "ExecutionContext",
    "Instruction",
    "InstructionType",
    "create_universal_vm",
    
    # Field Operators
    "ClassicalOperator",
    "QuantumOperator",
    "ProbabilisticOperator",
    "NeuralOperator",
    "TemporalOperator",
    "UnifiedOperator",
    "QuantumGate",
    "ClassicalLogicGate",
    
    # Field Equations
    "FieldEquations",
    "HamiltonianEvolution",
    "LagrangianDynamics",
    "InformationFlow",
    "FieldConfiguration",
    "compute_field_configuration",
]

# Version
__version__ = "1.0.0"
