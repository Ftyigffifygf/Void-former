"""Demo for Temporal Quantum Coherence & Virtual Quantum Clock System.

Usage:
    python -m voidformer.demo_temporal_quantum
"""

from __future__ import annotations

import torch
from voidformer.quantum_init import initialize_quantum_processor
from voidformer.quantum.temporal_coherence import create_time_aware_processor


def main():
    print("=" * 70)
    print("      ⏰ VOIDFORMER TEMPORAL COHERENCE & DECOHERENCE DEMO ⏰      ")
    print("=" * 70)

    d_model = 64
    batch_size = 2
    seq_len = 4

    base_processor = initialize_quantum_processor(
        d_model=d_model,
        n_qubits_per_token=3,
        collapse_protocol="entropy_gated",
        enable_entanglement=True,
    )

    models = ["exponential", "gaussian", "power_law", "linear", "amplitude_damping"]

    dummy_data = torch.randn(batch_size, seq_len, d_model)

    for model_type in models:
        print(f"\n--- Testing Decoherence Model: {model_type.upper()} ---")
        time_processor = create_time_aware_processor(
            base_processor,
            coherence_time_ms=500.0,
            virtual_ticks=1000,
            decoherence_model=model_type,
            noise_temp=0.01,
        )

        out, diagnostics = time_processor(dummy_data, log_temporal_evolution=True)

        print(f"  Processed Output Shape: {out.shape}")
        print(f"  Virtual Time Elapsed  : {diagnostics.get('virtual_time_elapsed_ms', 0):.2f} ms")
        print(f"  Final Fidelity        : {diagnostics.get('final_status', {}).get('current_fidelity', 1.0):.4f}")
        print(f"  Forced Collapse Triggered: {diagnostics.get('forced_collapse', False)}")

    print("\n" + "=" * 70)
    print("              TEMPORAL DEMO COMPLETED SUCCESSFULLY!                ")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
