"""Demo for Temporal Coherence System & Virtual Quantum Clock.

Demonstrates quantum clock tracking, decoherence models (exponential, gaussian, power law),
fidelity decay, and forced collapse at deadline.

Usage:
    python -m voidformer.demo_temporal_quantum
"""

from __future__ import annotations

import torch

from voidformer.quantum_init import initialize_quantum_processor
from voidformer.quantum.temporal_coherence import create_time_aware_processor, DecoherenceModel


def run_demo() -> None:
    print("=" * 65)
    print("     VOIDFORMER TEMPORAL COHERENCE & QUANTUM CLOCK DEMO        ")
    print("=" * 65)

    d_model = 64
    seq_len = 8
    batch_size = 2

    base_processor = initialize_quantum_processor(
        d_model=d_model,
        n_qubits_per_token=3,
        max_seq_len=16,
    )

    print("\n1. Initializing Time-Aware Quantum Processor...")
    time_processor = create_time_aware_processor(
        base_processor=base_processor,
        coherence_time_ms=500.0,
        virtual_ticks=2000,
        decoherence_model=DecoherenceModel.EXPONENTIAL,
        noise_temp=0.01,
    )

    x = torch.randn(batch_size, seq_len, d_model)

    print("\n2. Simulating Time Evolution & Decoherence...")
    output, diagnostics = time_processor(x, log_temporal_evolution=True)

    print(f"   Final Output Shape: {tuple(output.shape)}")
    print(f"   Forced Collapse Occurred: {diagnostics.get('forced_collapse', False)}")

    if "temporal_evolution" in diagnostics:
        print("\n3. Temporal Evolution Timeline:")
        for step in diagnostics["temporal_evolution"]:
            phase = step.get("phase", "step")
            elapsed = step.get("virtual_time_elapsed", 0.0) * 1000.0
            fidelity = step.get("current_fidelity", 1.0)
            dec_factor = step.get("decoherence_metrics", {}).get("decoherence_factor", 1.0)
            print(f"   - Phase: {phase:<25} | Time: {elapsed:6.2f} ms | Fidelity: {fidelity:.4f} | Dec Factor: {dec_factor:.4f}")

    print("\n" + "=" * 65)
    print("             TEMPORAL COHERENCE DEMO COMPLETED                ")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
