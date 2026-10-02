"""System Hardware Resource Auto-Tuner for Virtual Quantum Simulation.

Inspects host system specifications (CPU cores, available RAM, CUDA/GPU memory)
to dynamically configure optimal PyTorch thread counts, memory limits, and max
simulated qubit dimensions for statevector simulation.
"""

from __future__ import annotations

import os
import math
from typing import Dict, Any, Optional
import torch


class QuantumHardwareResourceTuner:
    """Inspects host device hardware specs and configures optimal simulation parameters."""

    def __init__(self, target_memory_fraction: float = 0.75):
        self.target_memory_fraction = target_memory_fraction

    @staticmethod
    def inspect_system_specs() -> Dict[str, Any]:
        """Inspect host CPU cores, memory, and CUDA GPU devices."""
        cpu_count = os.cpu_count() or 1

        # Check total and available memory
        total_ram_gb = 8.0
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_ram_gb = mem.total / (1024 ** 3)
            available_ram_gb = mem.available / (1024 ** 3)
        except ImportError:
            available_ram_gb = total_ram_gb * 0.5

        # Check GPU availability
        has_cuda = torch.cuda.is_available()
        gpu_name = None
        gpu_memory_gb = 0.0
        if has_cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                has_cuda = False

        return {
            "cpu_cores": cpu_count,
            "total_ram_gb": round(total_ram_gb, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "has_cuda": has_cuda,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_memory_gb, 2),
        }

    def compute_optimal_simulation_config(self) -> Dict[str, Any]:
        """Calculates max simulated qubits, optimal thread count, and device selection."""
        specs = self.inspect_system_specs()

        # Target memory allocation in bytes
        if specs["has_cuda"] and specs["gpu_memory_gb"] > 1.0:
            usable_memory_bytes = specs["gpu_memory_gb"] * (1024 ** 3) * self.target_memory_fraction
            device = "cuda"
        else:
            usable_memory_bytes = specs["available_ram_gb"] * (1024 ** 3) * self.target_memory_fraction
            device = "cpu"

        # Complex128 = 16 bytes per amplitude
        bytes_per_amplitude = 16
        max_amplitudes = usable_memory_bytes / bytes_per_amplitude

        # Max safe qubits for statevector simulation: 2^n <= max_amplitudes
        max_qubits = max(2, int(math.floor(math.log2(max(1, max_amplitudes)))))

        # Configure PyTorch CPU thread allocation
        optimal_threads = max(1, specs["cpu_cores"])
        if device == "cpu":
            try:
                torch.set_num_threads(optimal_threads)
            except Exception:
                pass

        return {
            "system_specs": specs,
            "selected_device": device,
            "optimal_num_threads": optimal_threads,
            "max_safe_simulated_qubits": max_qubits,
            "usable_memory_bytes": int(usable_memory_bytes),
            "target_memory_fraction": self.target_memory_fraction,
        }
