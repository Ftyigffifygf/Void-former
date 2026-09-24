# VoidFormer: Virtual Quantum Computing Simulator & Processor

**🔬 Quantum-Enhanced Neural Architecture** — A virtual quantum computing simulator
integrated with deep learning architectures. The simulator models quantum principles in classical software:

1. **⚛️ Quantum Superposition**: Tokens are mapped to state vectors `|ψ⟩ = Σᵢ αᵢ|i⟩` in 2^n Hilbert space
2. **🔗 Entanglement**: Non-local correlations between token positions via cross-token operations & Bell states
3. **🎯 Quantum Gates**: Unitary matrix operators (H, X, Y, Z, CNOT, Toffoli) manipulating complex state vectors
4. **📊 Measurement**: Born rule collapse `P(i) = |αᵢ|²` projecting quantum states to classical outputs

> ℹ️ This is a **classical software simulation** of quantum computing operations. It simulates complex-valued state vectors,
> unitary gate transformations, and probabilistic measurement for neural representations.

```
Classical:  x ∈ ℝ^d
           ↓
Quantum:    |ψ⟩ = Σᵢ αᵢ|i⟩  where αᵢ ∈ ℂ, Σ|αᵢ|² = 1
           ↓ [Quantum Gates: U|ψ⟩]
           ↓ [Entanglement: U_ij(θ)]
           ↓ [Measurement: collapse]
Classical:  observed state with P(i) = |αᵢ|²
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test quantum processor
python -m voidformer.quantum_init

# 3. Run comprehensive quantum core demo
python -m voidformer.demo_quantum

# 4. Test temporal coherence system
python -m voidformer.demo_temporal_quantum

# 5. Quick integration test
python -m voidformer.test_quantum_simple

# 6. Train model (tiny config)
python -m voidformer.train --config voidformer/configs/tiny.yaml --model-type quantum --steps 50

# 7. Generate text
python -m voidformer.infer --config voidformer/configs/tiny.yaml --model-type quantum --prompt "quantum computing is"
```

## ⚡ Features at a Glance

| Component | Description | Status |
|-----------|-------------|--------|
| 🧮 **Qubit State Manager** | Complex state vectors in 2^n Hilbert space | ✅ |
| 🚪 **Quantum Gates** | H, X, Y, Z, CNOT, Toffoli, Phase, T | ✅ |
| 🔗 **Entanglement** | Bell states, GHZ, cross-token unitary coupling | ✅ |
| 📏 **Measurement** | 5 collapse protocols (hard/soft/deferred/expectation/entropy-gated) | ✅ |
| 🔄 **Quantum Algorithms** | Grover, QFT, VQE simulation | ✅ |
| 🎯 **Quantum Attention** | Fidelity-based K(x,y) = \|⟨ψ(x)\|ψ(y)⟩\|² | ✅ |
| 🧩 **Tensor Networks** | MPS/Tensor-train compression | ✅ |
| 🤖 **Quantum LM** | Full quantum-enhanced language model | ✅ |
| ⏰ **Temporal Coherence** | Virtual quantum clock with decoherence | ✅ |
| 📉 **Decoherence Models** | Exponential, Gaussian, Power Law, Linear | ✅ |
| ⚡ **Forced Collapse** | Automatic measurement at deadline | ✅ |

---

## Quantum Architecture

```
                    ┌────────── Token ──────────┐
                    │                           │
              Classical Embedding (ℝ^d)
                    │
                    ↓
        ╔═══════════════════════════════════╗
        ║  QUANTUM STATE ENCODER            ║
        ║  Classical → Quantum Superposition║
        ║  ℝ^d → ℂ^(2^n) Hilbert Space     ║
        ╚═══════════════════════════════════╝
                    │
        |ψ⟩ = Σᵢ αᵢ|i⟩  (quantum state)
                    │
        ╔═══════════════════════════════════╗
        ║  QUANTUM GATE CIRCUIT             ║
        ║  • Hadamard (superposition)       ║
        ║  • CNOT (entanglement)            ║
        ║  • Pauli X/Y/Z (rotations)        ║
        ║  • Phase gates                    ║
        ╚═══════════════════════════════════╝
                    │
        ╔═══════════════════════════════════╗
        ║  ENTANGLEMENT LAYER               ║
        ║  Inter-token quantum correlations ║
        ║  Cross-token unitary coupling     ║
        ╚═══════════════════════════════════╝
                    │
        ╔═══════════════════════════════════╗
        ║  QUANTUM KERNEL ATTENTION         ║
        ║  Fidelity: K(x,y) = |⟨ψ(x)|ψ(y)⟩|²║
        ╚═══════════════════════════════════╝
                    │
        ╔═══════════════════════════════════╗
        ║  TENSOR NETWORK FFN (optional)    ║
        ║  MPS/Tensor-train decomposition   ║
        ╚═══════════════════════════════════╝
                    │
              QuantumVoidFormerBlock × N
                    │
        ╔═══════════════════════════════════╗
        ║  MEASUREMENT LAYER                ║
        ║  Quantum → Classical collapse     ║
        ║  Born rule: P(i) = |αᵢ|²         ║
        ╚═══════════════════════════════════╝
                    │
                 LM head
```

## Project Layout

```
voidformer/
├── quantum/                    # QUANTUM COMPUTING CORE
│   ├── qubit_state.py         #   State vectors, partial trace, fidelity
│   ├── quantum_gates.py       #   H, CNOT, X, Y, Z, Toffoli, Phase gates
│   ├── entanglement.py        #   Cross-token entanglement & von Neumann entropy
│   ├── measurement.py         #   Born rule collapse, 5 protocols
│   ├── quantum_processor.py   #   Virtual quantum CPU, circuit execution
│   ├── qiml.py               #   Quantum-inspired ML (tensor networks, QKA)
│   └── __init__.py
├── quantum_init.py            # QUANTUM PROCESSOR REGISTRY
├── models/
│   ├── quantum_voidformer.py  # QUANTUM-ENHANCED MODEL
│   ├── voidformer.py          # Classical dual-state model
│   └── __init__.py
├── layers/                    # Transformer layers
├── configs/                   # Model configurations (tiny/small/base)
├── training/                  # Training loops, loss functions, trainer
├── datasets/                  # Data loaders and tokenizers
├── experiments/               # Research scripts and ablations
├── visualization/             # Plotting tools
├── utils/                     # Config, logging, seeds
├── main.py                    # Dispatcher CLI
├── train.py                   # Training entry point
├── infer.py                   # Inference entry point
├── demo_quantum.py            # Comprehensive quantum demo
├── demo_temporal_quantum.py   # Temporal coherence demo
└── test_quantum_simple.py     # Integration test
```

## Quickstart Code Example

```python
from voidformer.quantum_init import initialize_quantum_processor
import torch

# Initialize quantum processor
processor = initialize_quantum_processor(
    d_model=256,
    n_qubits_per_token=4,
    collapse_protocol='entropy_gated',
    enable_entanglement=True
)

# Process data through quantum pipeline
input_data = torch.randn(2, 10, 256)
output, diagnostics = processor(input_data)

print('Quantum Processing Complete!')
print(f'Input entropy: {diagnostics["initial_quantum_entropy"]:.4f}')
print(f'Gates applied: {diagnostics["gates_applied"]}')
print(f'Entanglement: {diagnostics.get("is_entangled", False)}')
```

## Measurement Collapse Protocols

| Protocol | Behavior | Use Case |
|----------|----------|----------|
| `hard` | Full Born rule sampling → basis state (STE gradient flow) | Final output, discrete commitments |
| `soft` | Temperature-scaled probability mixture | Intermediate layers, continuous flow |
| `expectation` | Expectation value ⟨ψ\|O\|ψ⟩ via observable O | Analysis, expected value estimation |
| `deferred` | Outputs basis projection while preserving quantum state | Chained quantum operations |
| `entropy_gated` | Adaptive: high entropy→soft, low→hard | Default uncertainty-aware measurement |

## Mathematical Foundations

### Quantum State Representation

Every token embedding is mapped to a quantum state vector in Hilbert space:

```
|ψ⟩ = Σᵢ αᵢ|i⟩    where αᵢ ∈ ℂ, Σ|αᵢ|² = 1
```

### Entanglement Entropy via Partial Trace

For bipartition A|B, reduced density matrix $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$ yields von Neumann entanglement entropy:

```
S(ρ_A) = -Tr(ρ_A log₂ ρ_A) = -Σᵢ λᵢ log₂ λᵢ
```

where $\lambda_i$ are eigenvalues of $\rho_A$.

### Quantum Kernel Attention

Replaces softmax attention with quantum fidelity:
```
K(x, y) = |⟨ψ(x)|ψ(y)⟩|²
```

Attention weights reflect state overlap in Hilbert space.

---

## Citation

```bibtex
@misc{voidformer_quantum2026,
  title  = {VoidFormer: A Virtual Quantum Computing Simulator for Neural Language Models},
  year   = {2026},
  note   = {Virtual quantum simulation, entanglement, and measurement-based neural architecture}
}
```
