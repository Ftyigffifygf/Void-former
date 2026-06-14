"""Virtual Quantum Clock and Temporal Coherence Engine.

Implements time-based quantum decoherence, coherence windows, and scheduled
measurement collapse for the VoidQuantum framework.

PHYSICAL BASIS:
    T₂* = Transverse relaxation time (coherence lifetime)
    Γ(t) = Decoherence rate as function of time
    ρ(t) = ρ₀ · e^(-t/T₂*) (density matrix decay)
    
EXECUTION MODEL:
    1. Start quantum computation with coherence time budget
    2. Virtual clock tracks elapsed time
    3. Decoherence accumulates as t → T_coherence
    4. Forced measurement at timeframe deadline
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum

import torch
import torch.nn as nn

from .qubit_state import QuantumStateVector


class DecoherenceModel(Enum):
    """Physical decoherence models."""
    EXPONENTIAL = "exponential"           # ρ(t) ∝ e^(-t/T₂)
    GAUSSIAN = "gaussian"                 # ρ(t) ∝ e^(-(t/T₂)²)
    POWER_LAW = "power_law"               # ρ(t) ∝ (1 + t/T₂)^(-α)
    LINEAR = "linear"                     # ρ(t) = 1 - t/T_coherence
    AMPLITUDE_DAMPING = "amplitude_damping"  # T₁ relaxation


@dataclass
class CoherenceWindow:
    """Defines a temporal execution window for quantum computation.
    
    Attributes:
        coherence_time: Maximum coherence lifetime (seconds or ticks)
        start_time: Window start timestamp
        elapsed_time: Current elapsed time
        decoherence_rate: Rate of coherence loss (1/coherence_time)
        current_fidelity: Current state fidelity [0, 1]
        is_expired: Whether coherence window has expired
    """
    coherence_time: float                           # T₂* coherence lifetime
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    decoherence_rate: float = field(init=False)
    current_fidelity: float = 1.0
    is_expired: bool = False
    virtual_ticks: int = 0
    max_virtual_ticks: int = 2000
    
    def __post_init__(self):
        self.decoherence_rate = 1.0 / self.coherence_time
    
    def tick(self, delta_time: Optional[float] = None) -> float:
        """Advance virtual quantum clock.
        
        Args:
            delta_time: Time increment (auto-compute if None)
        
        Returns:
            New fidelity value after time evolution
        """
        if delta_time is None:
            current_time = time.time()
            delta_time = current_time - (self.start_time + self.elapsed_time)
        
        self.elapsed_time += delta_time
        self.virtual_ticks += 1
        
        # Check expiration
        if self.elapsed_time >= self.coherence_time:
            self.is_expired = True
            self.current_fidelity = 0.0
        
        if self.virtual_ticks >= self.max_virtual_ticks:
            self.is_expired = True
        
        return self.current_fidelity
    
    def get_remaining_time(self) -> float:
        """Get remaining coherence time."""
        return max(0.0, self.coherence_time - self.elapsed_time)
    
    def get_time_fraction(self) -> float:
        """Get fraction of coherence time elapsed [0, 1]."""
        return min(1.0, self.elapsed_time / self.coherence_time)
    
    def get_remaining_ticks(self) -> int:
        """Get remaining virtual ticks."""
        return max(0, self.max_virtual_ticks - self.virtual_ticks)


@dataclass
class DecoherenceMetrics:
    """Metrics tracking quantum decoherence over time."""
    time_elapsed: float
    fidelity: float                     # State fidelity [0, 1]
    purity: float                       # Tr(ρ²) for mixed states
    entropy: float                      # von Neumann entropy
    decoherence_factor: float           # Coherence loss multiplier
    measurement_readiness: float        # How ready for forced measurement
    noise_level: float                  # Environmental noise accumulation
    
    def to_dict(self) -> dict:
        return {
            "time_elapsed": self.time_elapsed,
            "fidelity": self.fidelity,
            "purity": self.purity,
            "entropy": self.entropy,
            "decoherence_factor": self.decoherence_factor,
            "measurement_readiness": self.measurement_readiness,
            "noise_level": self.noise_level,
        }


class QuantumClock(nn.Module):
    """Virtual Quantum Clock for temporal state management.
    
    Tracks quantum computation time and manages coherence windows.
    Simulates realistic quantum decoherence as time progresses.
    
    PHYSICAL MODEL:
        Fidelity decay: F(t) = F₀ · e^(-t/T₂*)
        Purity decay: Tr(ρ²(t)) = Tr(ρ₀²) · e^(-2t/T₂*)
        Entropy growth: S(t) = S₀ + (S_max - S₀)(1 - e^(-t/T₂*))
    """
    
    def __init__(
        self,
        coherence_time: float = 0.5,        # seconds
        max_virtual_ticks: int = 2000,      # virtual time steps
        decoherence_model: DecoherenceModel = DecoherenceModel.EXPONENTIAL,
        noise_temperature: float = 0.01,    # Environmental noise level
        enable_automatic_collapse: bool = True,
    ):
        super().__init__()
        self.coherence_time = coherence_time
        self.max_virtual_ticks = max_virtual_ticks
        self.decoherence_model = decoherence_model
        self.noise_temperature = noise_temperature
        self.enable_automatic_collapse = enable_automatic_collapse
        
        # Active coherence window
        self.active_window: Optional[CoherenceWindow] = None
        
        # Metrics history
        self.metrics_history: List[DecoherenceMetrics] = []
        
        # Learnable decoherence parameters
        self.T2_star = nn.Parameter(torch.tensor(coherence_time))
        self.noise_scale = nn.Parameter(torch.tensor(noise_temperature))
        
        # Statistics
        self.register_buffer("total_operations", torch.tensor(0))
        self.register_buffer("forced_collapses", torch.tensor(0))
    
    def start_coherence_window(self) -> CoherenceWindow:
        """Initialize a new coherence time window.
        
        Returns:
            New CoherenceWindow with fresh quantum coherence budget
        """
        self.active_window = CoherenceWindow(
            coherence_time=self.T2_star.item(),
            max_virtual_ticks=self.max_virtual_ticks,
        )
        self.metrics_history = []
        return self.active_window
    
    def tick(self, delta_time: Optional[float] = None) -> CoherenceWindow:
        """Advance the quantum clock by one tick.
        
        Args:
            delta_time: Time increment (auto if None)
        
        Returns:
            Updated coherence window
        """
        if self.active_window is None:
            self.start_coherence_window()
        
        self.active_window.tick(delta_time)
        self.total_operations += 1
        
        return self.active_window
    
    def compute_decoherence_factor(self, t: float) -> float:
        """Compute decoherence factor D(t) ∈ [0, 1].
        
        Args:
            t: Elapsed time
        
        Returns:
            Decoherence factor (1 = perfect coherence, 0 = fully decohered)
        """
        T2 = self.T2_star.item()
        
        if self.decoherence_model == DecoherenceModel.EXPONENTIAL:
            # Exponential decay: e^(-t/T₂)
            return math.exp(-t / T2)
        
        elif self.decoherence_model == DecoherenceModel.GAUSSIAN:
            # Gaussian decay: e^(-(t/T₂)²)
            return math.exp(-(t / T2) ** 2)
        
        elif self.decoherence_model == DecoherenceModel.POWER_LAW:
            # Power law: (1 + t/T₂)^(-2)
            return 1.0 / (1.0 + t / T2) ** 2
        
        elif self.decoherence_model == DecoherenceModel.LINEAR:
            # Linear decay: 1 - t/T₂
            return max(0.0, 1.0 - t / T2)
        
        elif self.decoherence_model == DecoherenceModel.AMPLITUDE_DAMPING:
            # Amplitude damping: √(1 - (1 - e^(-t/T₁)))
            return math.sqrt(math.exp(-t / T2))
        
        return 1.0
    
    def apply_decoherence(
        self,
        quantum_state: QuantumStateVector,
    ) -> tuple[QuantumStateVector, DecoherenceMetrics]:
        """Apply time-dependent decoherence to quantum state.
        
        Simulates environmental decoherence by:
        1. Computing decoherence factor from elapsed time
        2. Adding thermal noise to amplitudes
        3. Reducing off-diagonal density matrix elements
        4. Tracking fidelity and purity
        
        Args:
            quantum_state: Input quantum state
        
        Returns:
            Decohered quantum state, decoherence metrics
        """
        if self.active_window is None:
            self.start_coherence_window()
        
        t = self.active_window.elapsed_time
        
        # Compute decoherence factor
        D_t = self.compute_decoherence_factor(t)
        
        # Apply decoherence to amplitudes
        # Physical model: ρ(t) = D(t)·ρ₀ + (1-D(t))·ρ_thermal
        
        # 1. Scale coherent amplitudes by D(t)
        coherent_part = quantum_state.amplitudes * D_t
        
        # 2. Add thermal noise (mixed state contribution)
        noise_level = self.noise_scale.item() * (1 - D_t)
        if noise_level > 0 and self.training:
            thermal_noise = torch.randn_like(quantum_state.amplitudes) * noise_level
            coherent_part = coherent_part + thermal_noise
        
        # 3. Gradual phase randomization (off-diagonal decay)
        # Random phases accumulate, destroying interference
        if D_t < 0.9 and self.training:
            phase_noise = torch.randn(
                *quantum_state.amplitudes.shape,
                device=quantum_state.amplitudes.device
            ) * (1 - D_t) * 0.1
            phase_perturbation = torch.exp(1j * phase_noise.to(quantum_state.amplitudes.dtype))
            coherent_part = coherent_part * phase_perturbation
        
        # Create decohered state
        decohered_state = QuantumStateVector(
            amplitudes=coherent_part,
            n_qubits=quantum_state.n_qubits,
            global_phase=quantum_state.global_phase,
        ).normalize()
        
        # Compute fidelity with original state
        fidelity = quantum_state.fidelity(decohered_state).mean().item()
        
        # Compute purity Tr(ρ²) (1 = pure, <1 = mixed)
        probs = decohered_state.probabilities
        purity = (probs ** 2).sum(dim=-1).mean().item()
        
        # von Neumann entropy
        entropy = decohered_state.measure_entropy().mean().item()
        
        # Measurement readiness (should collapse when → 1)
        time_frac = self.active_window.get_time_fraction()
        measurement_readiness = min(1.0, time_frac + (1 - D_t) * 0.5)
        
        # Noise accumulation
        noise_level_out = noise_level + (1 - D_t) * self.noise_scale.item()
        
        # Create metrics
        metrics = DecoherenceMetrics(
            time_elapsed=t,
            fidelity=fidelity,
            purity=purity,
            entropy=entropy,
            decoherence_factor=D_t,
            measurement_readiness=measurement_readiness,
            noise_level=noise_level_out,
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Update window fidelity
        self.active_window.current_fidelity = fidelity
        
        return decohered_state, metrics
    
    def should_force_collapse(self) -> bool:
        """Check if coherence window has expired and forced collapse needed.
        
        Returns:
            True if state must be measured immediately
        """
        if self.active_window is None:
            return False
        
        return (
            self.active_window.is_expired or
            self.active_window.current_fidelity < 0.1 or
            self.active_window.get_time_fraction() >= 1.0
        )
    
    def get_status_report(self) -> dict:
        """Get current quantum clock status.
        
        Returns:
            Dictionary with timing and coherence information
        """
        if self.active_window is None:
            return {"status": "no_active_window"}
        
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "virtual_time_elapsed": self.active_window.elapsed_time,
            "virtual_ticks": self.active_window.virtual_ticks,
            "max_ticks": self.active_window.max_virtual_ticks,
            "remaining_time": self.active_window.get_remaining_time(),
            "remaining_ticks": self.active_window.get_remaining_ticks(),
            "time_fraction": self.active_window.get_time_fraction(),
            "current_fidelity": self.active_window.current_fidelity,
            "is_expired": self.active_window.is_expired,
            "force_collapse_required": self.should_force_collapse(),
            "total_operations": self.total_operations.item(),
            "forced_collapses": self.forced_collapses.item(),
            "latest_metrics": latest_metrics.to_dict() if latest_metrics else None,
        }
    
    def reset(self):
        """Reset quantum clock for new computation."""
        self.active_window = None
        self.metrics_history = []


class TimeAwareQuantumProcessor(nn.Module):
    """Quantum processor with temporal coherence enforcement.
    
    Wraps standard quantum processor with:
    1. Automatic coherence window management
    2. Time-tracked state evolution
    3. Decoherence simulation
    4. Scheduled forced measurement at deadline
    """
    
    def __init__(
        self,
        base_processor: nn.Module,          # VirtualQuantumProcessor
        coherence_time: float = 0.5,
        max_virtual_ticks: int = 2000,
        decoherence_model: DecoherenceModel = DecoherenceModel.EXPONENTIAL,
        noise_temperature: float = 0.01,
        auto_collapse_on_deadline: bool = True,
    ):
        super().__init__()
        self.base_processor = base_processor
        self.quantum_clock = QuantumClock(
            coherence_time=coherence_time,
            max_virtual_ticks=max_virtual_ticks,
            decoherence_model=decoherence_model,
            noise_temperature=noise_temperature,
            enable_automatic_collapse=auto_collapse_on_deadline,
        )
        self.auto_collapse = auto_collapse_on_deadline
        
        # Execution logs
        self.execution_log: List[dict] = []
    
    def forward(
        self,
        classical_input: torch.Tensor,
        log_temporal_evolution: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Process input through time-aware quantum pipeline.
        
        Args:
            classical_input: Classical embeddings (B, T, d_model)
            log_temporal_evolution: Log timing at each step
        
        Returns:
            Output tensor, diagnostics with temporal information
        """
        # Start coherence window
        window = self.quantum_clock.start_coherence_window()
        
        diagnostics = {
            "temporal_evolution": [],
            "forced_collapse": False,
        }
        
        # Phase 1: Encode to quantum
        self.quantum_clock.tick()
        quantum_state = self.base_processor.encode_classical_to_quantum(classical_input)
        
        if log_temporal_evolution:
            status = self.quantum_clock.get_status_report()
            diagnostics["temporal_evolution"].append({
                "phase": "encode",
                **status
            })
        
        # Phase 2: Execute quantum circuit with decoherence
        self.quantum_clock.tick()
        quantum_state, circuit_diag = self.base_processor.execute_quantum_circuit(quantum_state)
        
        # Apply decoherence after circuit
        quantum_state, decohere_metrics = self.quantum_clock.apply_decoherence(quantum_state)
        
        if log_temporal_evolution:
            status = self.quantum_clock.get_status_report()
            diagnostics["temporal_evolution"].append({
                "phase": "circuit_execution",
                **status,
                "decoherence_metrics": decohere_metrics.to_dict(),
            })
        
        # Phase 3: Entanglement (if time permits)
        if not self.quantum_clock.should_force_collapse():
            self.quantum_clock.tick()
            quantum_state, entangle_diag = self.base_processor.apply_entanglement_layer(
                quantum_state
            )
            
            # Apply decoherence after entanglement
            quantum_state, decohere_metrics = self.quantum_clock.apply_decoherence(quantum_state)
            
            if log_temporal_evolution:
                status = self.quantum_clock.get_status_report()
                diagnostics["temporal_evolution"].append({
                    "phase": "entanglement",
                    **status,
                    "decoherence_metrics": decohere_metrics.to_dict(),
                })
        else:
            diagnostics["entanglement_skipped"] = "coherence_expired"
        
        # Phase 4: Forced measurement at deadline
        self.quantum_clock.tick()
        
        if self.quantum_clock.should_force_collapse():
            diagnostics["forced_collapse"] = True
            self.quantum_clock.forced_collapses += 1
            
            # Force to hard collapse protocol
            from .measurement import MeasurementLayer, CollapseProtocol
            forced_measurement = MeasurementLayer(
                n_qubits=quantum_state.n_qubits,
                d_output=classical_input.shape[-1],
                collapse_protocol=CollapseProtocol.HARD,
            )
            classical_output, _, measure_diag = forced_measurement(quantum_state)
            
            diagnostics["forced_measurement_reason"] = "coherence_deadline_reached"
        else:
            # Standard measurement
            classical_output, measure_diag = self.base_processor.measure_quantum_to_classical(
                quantum_state
            )
        
        if log_temporal_evolution:
            status = self.quantum_clock.get_status_report()
            diagnostics["temporal_evolution"].append({
                "phase": "measurement",
                **status,
            })
        
        # Final status
        diagnostics["final_status"] = self.quantum_clock.get_status_report()
        diagnostics["metrics_history"] = [m.to_dict() for m in self.quantum_clock.metrics_history]
        
        # Reset for next computation
        self.quantum_clock.reset()
        
        return classical_output, diagnostics
    
    def get_execution_summary(self) -> dict:
        """Get summary of all executions."""
        return {
            "total_operations": self.quantum_clock.total_operations.item(),
            "total_forced_collapses": self.quantum_clock.forced_collapses.item(),
            "forced_collapse_rate": (
                self.quantum_clock.forced_collapses.item() /
                max(1, self.quantum_clock.total_operations.item())
            ),
        }


def create_time_aware_processor(
    base_processor: nn.Module,
    coherence_time_ms: float = 500.0,      # milliseconds
    virtual_ticks: int = 2000,
    decoherence_model: str = "exponential",
    noise_temp: float = 0.01,
) -> TimeAwareQuantumProcessor:
    """Factory function for time-aware quantum processor.
    
    Args:
        base_processor: Base VirtualQuantumProcessor
        coherence_time_ms: Coherence window in milliseconds
        virtual_ticks: Maximum virtual time steps
        decoherence_model: Type of decoherence ("exponential", "gaussian", etc.)
        noise_temp: Environmental noise temperature
    
    Returns:
        TimeAwareQuantumProcessor with temporal enforcement
    """
    model_map = {
        "exponential": DecoherenceModel.EXPONENTIAL,
        "gaussian": DecoherenceModel.GAUSSIAN,
        "power_law": DecoherenceModel.POWER_LAW,
        "linear": DecoherenceModel.LINEAR,
        "amplitude_damping": DecoherenceModel.AMPLITUDE_DAMPING,
    }
    
    return TimeAwareQuantumProcessor(
        base_processor=base_processor,
        coherence_time=coherence_time_ms / 1000.0,  # Convert to seconds
        max_virtual_ticks=virtual_ticks,
        decoherence_model=model_map.get(decoherence_model, DecoherenceModel.EXPONENTIAL),
        noise_temperature=noise_temp,
        auto_collapse_on_deadline=True,
    )
