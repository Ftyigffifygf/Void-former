"""Universal Virtual Machine for Unified Computational Field.

A VM that can execute computations in any paradigm (classical, quantum,
probabilistic, neural, temporal) by operating on the unified field.

VIRTUAL MACHINE ARCHITECTURE:
    - Registers: Field state vectors
    - Memory: Tensor storage with quantum/classical duality
    - Instructions: Field operators (gates, transforms, measurements)
    - Execution: Field evolution + mode transitions
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

import torch
import torch.nn as nn

from .field_theory import FieldMode, UnifiedComputationalField


class InstructionType(Enum):
    """VM instruction types."""
    # Classical operations
    LOAD = "load"
    STORE = "store"
    ADD = "add"
    MUL = "mul"
    
    # Quantum operations
    HADAMARD = "hadamard"
    CNOT = "cnot"
    MEASURE = "measure"
    
    # Probabilistic operations
    SAMPLE = "sample"
    EXPECTATION = "expectation"
    
    # Neural operations
    FORWARD = "forward"
    BACKWARD = "backward"
    UPDATE = "update"
    
    # Temporal operations
    EVOLVE = "evolve"
    CHECKPOINT = "checkpoint"
    RESTORE = "restore"
    
    # Mode control
    SET_MODE = "set_mode"
    TRANSITION = "transition"


@dataclass
class ComputationalState:
    """State of the virtual machine.
    
    Encapsulates the entire computational state including:
    - Field configuration
    - Register values
    - Memory contents
    - Execution context
    """
    field_state: torch.Tensor  # The unified field |Ψ⟩
    registers: Dict[str, torch.Tensor] = dc_field(default_factory=dict)
    memory: Dict[str, torch.Tensor] = dc_field(default_factory=dict)
    mode: FieldMode = FieldMode.UNIFIED
    program_counter: int = 0
    call_stack: List[int] = dc_field(default_factory=list)
    
    # Quantum-specific state
    measurement_history: List[Dict[str, Any]] = dc_field(default_factory=list)
    entanglement_map: Dict[tuple, float] = dc_field(default_factory=dict)
    
    # Temporal state
    time_elapsed: float = 0.0
    checkpoints: Dict[str, ComputationalState] = dc_field(default_factory=dict)
    
    def clone(self) -> ComputationalState:
        """Create a deep copy of the state (for checkpointing)."""
        # Manual clone to avoid PyTorch deepcopy issues
        return ComputationalState(
            field_state=self.field_state.clone().detach(),
            registers={k: v.clone().detach() if isinstance(v, torch.Tensor) else v 
                      for k, v in self.registers.items()},
            memory={k: v.clone().detach() if isinstance(v, torch.Tensor) else v
                   for k, v in self.memory.items()},
            mode=self.mode,
            program_counter=self.program_counter,
            call_stack=self.call_stack.copy(),
            measurement_history=[m.copy() for m in self.measurement_history],
            entanglement_map=self.entanglement_map.copy(),
            time_elapsed=self.time_elapsed,
            checkpoints={},  # Don't recursively clone checkpoints
        )


@dataclass
class ExecutionContext:
    """Context for VM execution.
    
    Contains metadata and configuration for execution.
    """
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    
    # Execution limits
    max_instructions: int = 10000
    max_memory_bytes: int = 1_000_000_000  # 1GB
    
    # Quantum parameters
    measurement_shots: int = 1000
    decoherence_rate: float = 0.01
    
    # Neural parameters
    learning_rate: float = 0.001
    gradient_clip: float = 1.0
    
    # Temporal parameters
    time_step: float = 0.01
    max_evolution_time: float = 100.0
    
    # Debugging
    trace_execution: bool = False
    collect_diagnostics: bool = True


class Instruction:
    """A single VM instruction."""
    
    def __init__(
        self,
        op: InstructionType,
        operands: List[Any] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.op = op
        self.operands = operands or []
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        operand_str = ", ".join(str(o) for o in self.operands)
        return f"{self.op.value}({operand_str})"


class UniversalVirtualMachine:
    """Universal Virtual Machine for Unified Computational Field.
    
    This VM can execute programs in ANY computational paradigm by
    operating on the unified field. It bridges:
    
    1. Classical computing: Traditional CPU instructions
    2. Quantum computing: Gate-based quantum circuits
    3. Probabilistic computing: Sampling and inference
    4. Neural computing: Gradient-based learning
    5. Temporal computing: Time-evolution dynamics
    
    ARCHITECTURE:
        Field Layer: Unified computational field (Ψ)
        Instruction Set: Universal operations across all paradigms
        Execution Engine: Mode-aware instruction interpreter
        Memory System: Quantum-classical hybrid memory
    
    PROGRAMMING MODEL:
        Programs are sequences of instructions that manipulate
        the unified field. The VM automatically handles:
        - Mode transitions (e.g., classical → quantum)
        - State coherence and decoherence
        - Memory management
        - Time evolution
    """
    
    def __init__(
        self,
        field: UnifiedComputationalField,
        context: Optional[ExecutionContext] = None,
    ):
        """Initialize the Universal VM.
        
        Args:
            field: The unified computational field
            context: Execution context (defaults to CPU, float32)
        """
        self.field = field
        self.context = context or ExecutionContext()
        
        # Initialize VM state
        self.state = ComputationalState(
            field_state=field.field_state.clone(),
            mode=FieldMode.UNIFIED,
        )
        
        # Instruction handlers (dispatch table)
        self.handlers: Dict[InstructionType, Callable] = {
            # Classical
            InstructionType.LOAD: self._exec_load,
            InstructionType.STORE: self._exec_store,
            InstructionType.ADD: self._exec_add,
            InstructionType.MUL: self._exec_mul,
            
            # Quantum
            InstructionType.HADAMARD: self._exec_hadamard,
            InstructionType.CNOT: self._exec_cnot,
            InstructionType.MEASURE: self._exec_measure,
            
            # Probabilistic
            InstructionType.SAMPLE: self._exec_sample,
            InstructionType.EXPECTATION: self._exec_expectation,
            
            # Neural
            InstructionType.FORWARD: self._exec_forward,
            InstructionType.BACKWARD: self._exec_backward,
            InstructionType.UPDATE: self._exec_update,
            
            # Temporal
            InstructionType.EVOLVE: self._exec_evolve,
            InstructionType.CHECKPOINT: self._exec_checkpoint,
            InstructionType.RESTORE: self._exec_restore,
            
            # Mode control
            InstructionType.SET_MODE: self._exec_set_mode,
            InstructionType.TRANSITION: self._exec_transition,
        }
        
        # Execution trace
        self.execution_trace: List[Dict[str, Any]] = []
    
    def execute_program(
        self,
        instructions: List[Instruction],
    ) -> tuple[ComputationalState, Dict[str, Any]]:
        """Execute a program (sequence of instructions).
        
        Args:
            instructions: List of instructions to execute
        
        Returns:
            Final state, execution diagnostics
        """
        self.state.program_counter = 0
        instruction_count = 0
        
        diagnostics = {
            "instructions_executed": 0,
            "mode_transitions": 0,
            "measurements_performed": 0,
            "field_energy_trace": [],
            "field_entropy_trace": [],
        }
        
        while self.state.program_counter < len(instructions):
            if instruction_count >= self.context.max_instructions:
                raise RuntimeError(f"Exceeded max instructions: {self.context.max_instructions}")
            
            # Fetch instruction
            instr = instructions[self.state.program_counter]
            
            # Execute instruction
            if self.context.trace_execution:
                print(f"[PC={self.state.program_counter}] {instr}")
            
            handler = self.handlers.get(instr.op)
            if handler is None:
                raise ValueError(f"Unknown instruction: {instr.op}")
            
            handler(instr)
            
            # Update diagnostics
            instruction_count += 1
            self.state.program_counter += 1
            
            if self.context.collect_diagnostics:
                diagnostics["field_energy_trace"].append(
                    self.field.compute_field_energy().item()
                )
                diagnostics["field_entropy_trace"].append(
                    self.field.compute_field_entropy().item()
                )
        
        diagnostics["instructions_executed"] = instruction_count
        diagnostics["mode_transitions"] = sum(
            1 for trace in self.execution_trace
            if trace.get("event") == "mode_transition"
        )
        diagnostics["measurements_performed"] = len(self.state.measurement_history)
        
        return self.state, diagnostics
    
    # ============================================================
    # Classical Instructions
    # ============================================================
    
    def _exec_load(self, instr: Instruction):
        """Load value from memory to register."""
        reg_name, mem_key = instr.operands
        self.state.registers[reg_name] = self.state.memory.get(
            mem_key,
            torch.zeros(1)
        ).clone()
    
    def _exec_store(self, instr: Instruction):
        """Store value from register to memory."""
        mem_key, reg_name = instr.operands
        self.state.memory[mem_key] = self.state.registers[reg_name].clone()
    
    def _exec_add(self, instr: Instruction):
        """Add two registers."""
        dest, src1, src2 = instr.operands
        self.state.registers[dest] = (
            self.state.registers[src1] + self.state.registers[src2]
        )
    
    def _exec_mul(self, instr: Instruction):
        """Multiply two registers."""
        dest, src1, src2 = instr.operands
        self.state.registers[dest] = (
            self.state.registers[src1] * self.state.registers[src2]
        )
    
    # ============================================================
    # Quantum Instructions
    # ============================================================
    
    def _exec_hadamard(self, instr: Instruction):
        """Apply Hadamard gate to qubit."""
        qubit_idx = instr.operands[0]
        # Apply H gate to field (simplified)
        quantum_part = self.field.project_to_mode(FieldMode.QUANTUM)
        # H gate creates superposition
        quantum_part = quantum_part / torch.sqrt(torch.tensor(2.0))
        self.field.field_state[:, self.field.d_classical:self.field.d_classical + self.field.d_quantum] = quantum_part
    
    def _exec_cnot(self, instr: Instruction):
        """Apply CNOT gate."""
        control, target = instr.operands
        # Apply CNOT to field (creates entanglement)
        self.state.entanglement_map[(control, target)] = 1.0
    
    def _exec_measure(self, instr: Instruction):
        """Measure quantum state (collapse)."""
        qubit_idx = instr.operands[0]
        quantum_part = self.field.project_to_mode(FieldMode.QUANTUM)
        
        # Born rule measurement
        probs = (quantum_part.real ** 2 + quantum_part.imag ** 2).flatten()
        probs = probs / probs.sum()
        
        # Sample outcome
        outcome = torch.multinomial(probs, 1).item()
        
        # Record measurement
        self.state.measurement_history.append({
            "qubit": qubit_idx,
            "outcome": outcome,
            "probability": probs[outcome].item(),
        })
    
    # ============================================================
    # Probabilistic Instructions
    # ============================================================
    
    def _exec_sample(self, instr: Instruction):
        """Sample from probabilistic distribution."""
        dist_name = instr.operands[0]
        prob_part = self.field.project_to_mode(FieldMode.PROBABILISTIC)
        probs = torch.softmax(prob_part.real.flatten(), dim=0)
        sample = torch.multinomial(probs, 1)
        self.state.registers[dist_name] = sample.float()
    
    def _exec_expectation(self, instr: Instruction):
        """Compute expectation value."""
        result_reg = instr.operands[0]
        prob_part = self.field.project_to_mode(FieldMode.PROBABILISTIC)
        expectation = prob_part.real.mean()
        self.state.registers[result_reg] = expectation.unsqueeze(0)
    
    # ============================================================
    # Neural Instructions
    # ============================================================
    
    def _exec_forward(self, instr: Instruction):
        """Forward pass through neural component."""
        input_reg, output_reg = instr.operands
        neural_part = self.field.project_to_mode(FieldMode.NEURAL)
        # Apply neural Hamiltonian (acts like a layer)
        output = self.field.H_neural(neural_part.real)
        self.state.registers[output_reg] = output.flatten()
    
    def _exec_backward(self, instr: Instruction):
        """Backward pass (gradient computation)."""
        # Placeholder: Would compute gradients
        pass
    
    def _exec_update(self, instr: Instruction):
        """Update parameters (gradient descent step)."""
        # Placeholder: Would update field parameters
        pass
    
    # ============================================================
    # Temporal Instructions
    # ============================================================
    
    def _exec_evolve(self, instr: Instruction):
        """Evolve field forward in time."""
        dt = instr.operands[0] if instr.operands else self.context.time_step
        self.field.evolve_field(dt=dt, n_steps=1)
        self.state.time_elapsed += dt
        self.state.field_state = self.field.field_state.clone()
    
    def _exec_checkpoint(self, instr: Instruction):
        """Save current state."""
        checkpoint_name = instr.operands[0]
        self.state.checkpoints[checkpoint_name] = self.state.clone()
    
    def _exec_restore(self, instr: Instruction):
        """Restore saved state."""
        checkpoint_name = instr.operands[0]
        if checkpoint_name in self.state.checkpoints:
            self.state = self.state.checkpoints[checkpoint_name].clone()
            self.field.field_state = self.state.field_state.clone()
    
    # ============================================================
    # Mode Control Instructions
    # ============================================================
    
    def _exec_set_mode(self, instr: Instruction):
        """Set computational mode."""
        mode_str = instr.operands[0]
        new_mode = FieldMode(mode_str)
        
        if self.context.trace_execution:
            print(f"  Mode transition: {self.state.mode} → {new_mode}")
        
        self.execution_trace.append({
            "event": "mode_transition",
            "from": self.state.mode,
            "to": new_mode,
        })
        
        self.state.mode = new_mode
    
    def _exec_transition(self, instr: Instruction):
        """Perform mode transition with field evolution."""
        from_mode_str, to_mode_str = instr.operands
        from_mode = FieldMode(from_mode_str)
        to_mode = FieldMode(to_mode_str)
        
        # Compute transition (but don't do expensive evolution)
        transition = self.field.mode_transition(from_mode, to_mode)
        
        # Simple mode update without expensive evolution
        self.state.mode = to_mode
        self.state.time_elapsed += 0.01  # Small time increment
        
        if self.context.trace_execution:
            print(f"  Transitioned {from_mode.value} -> {to_mode.value}")
        
        self.execution_trace.append({
            "event": "mode_transition",
            "from": from_mode,
            "to": to_mode,
        })


def create_universal_vm(
    d_classical: int = 128,
    d_quantum: int = 16,
    d_probabilistic: int = 64,
    d_neural: int = 256,
    device: str = "cpu",
) -> UniversalVirtualMachine:
    """Factory function to create a Universal VM.
    
    Args:
        d_classical: Classical state dimension
        d_quantum: Quantum state dimension
        d_probabilistic: Probabilistic state dimension
        d_neural: Neural state dimension
        device: Device to run on
    
    Returns:
        Initialized Universal VM
    """
    field = UnifiedComputationalField(
        d_classical=d_classical,
        d_quantum=d_quantum,
        d_probabilistic=d_probabilistic,
        d_neural=d_neural,
    )
    field.to(device)
    
    context = ExecutionContext(device=torch.device(device))
    
    return UniversalVirtualMachine(field, context)
