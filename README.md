# VoidFormer: Virtual Quantum Computing Simulator & Processor

**🔬 Quantum-Enhanced Neural Architecture** — A research-grade quantum computing simulator
integrated with deep learning. The architecture operates on true quantum principles:

1. **⚛️ Quantum Superposition**: Tokens exist as state vectors `|ψ⟩ = Σᵢ αᵢ|i⟩` in 2^n Hilbert space
2. **🔗 Entanglement**: Non-local quantum correlations between tokens via CNOT, Bell states
3. **🎯 Quantum Gates**: Unitary operators (H, X, Y, Z, CNOT, Toffoli) manipulate quantum states
4. **📊 Measurement**: Born rule collapse `P(i) = |αᵢ|²` converts quantum → classical output

> ✨ This **IS** quantum computing simulation. It uses complex-valued state vectors,
> unitary gate operations, and quantum measurement to process information.

```
Classical:  |T⟩ = α|S_c⟩ + β|S_v⟩ + γ·I(|S_c⟩,|S_v⟩)
           ↓
Quantum:    |ψ⟩ = Σᵢ αᵢ|i⟩  where αᵢ ∈ ℂ, Σ|αᵢ|² = 1
           ↓ [Quantum Gates: U|ψ⟩]
           ↓ [Entanglement: CNOT]
           ↓ [Measurement: collapse]
Classical:  observed state with P(i) = |αᵢ|²
```

💫 Data remains in **probabilistic quantum superposition** during processing and only
collapses to deterministic output at measurement.

---

## 🚀 Quick Start

```bash
# 1. Test quantum processor
python -m voidformer.quantum_init

# 2. Run comprehensive demos
python -m voidformer.demo_quantum

# 3. Test temporal coherence system
python -m voidformer.demo_temporal_quantum

# 4. Quick integration test
python -m voidformer.test_quantum_simple
```

## ⚡ Features at a Glance

| Component | Description | Status |
|-----------|-------------|--------|
| 🧮 **Qubit State Manager** | Complex state vectors in 2^n Hilbert space | ✅ |
| 🚪 **Quantum Gates** | H, X, Y, Z, CNOT, Toffoli, Phase, T | ✅ |
| 🔗 **Entanglement** | Bell states, GHZ, learned patterns | ✅ |
| 📏 **Measurement** | 5 collapse protocols (hard/soft/entropy-gated) | ✅ |
| 🔄 **Quantum Algorithms** | Grover, QFT, VQE | ✅ |
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
        ║  Bell states, GHZ states          ║
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
├── quantum/                    # 🆕 QUANTUM COMPUTING CORE
│   ├── qubit_state.py         #   State vectors, superposition, normalization
│   ├── quantum_gates.py       #   H, CNOT, X, Y, Z, Toffoli, Phase gates
│   ├── entanglement.py        #   Bell states, GHZ states, concurrence
│   ├── measurement.py         #   Born rule collapse, protocols
│   ├── quantum_processor.py   #   Virtual quantum CPU, circuit execution
│   ├── qiml.py               #   Quantum-inspired ML (tensor networks, QKA)
│   └── __init__.py
├── quantum_init.py            # 🆕 QUANTUM PROCESSOR REGISTRY & ENTRY POINT
├── models/
│   ├── quantum_voidformer.py  # 🆕 QUANTUM-ENHANCED MODEL
│   ├── voidformer.py          #   (Legacy classical model)
│   └── __init__.py
├── layers/                    #   Classical transformer layers (legacy)
├── configs/                   #   Model configurations (tiny/small/base)
├── training/                  #   Training loops and losses
├── datasets/                  #   Data loaders and tokenizers
├── experiments/               #   Research scripts and analysis
├── visualization/             #   Plotting tools
├── utils/                     #   Config, logging, seeds
├── tests/                     #   Unit tests
├── main.py                    #   Dispatcher CLI
├── train.py                   #   Training entry point
└── infer.py                   #   Inference entry point
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test quantum processor
python -m voidformer.quantum_init

# 3. Train quantum-enhanced model (tiny config)
python -m voidformer.train \
    --config voidformer/configs/tiny.yaml \
    --model-type quantum \
    --steps 100

# 4. Inference with quantum processing
python -m voidformer.infer \
    --config voidformer/configs/tiny.yaml \
    --model-type quantum \
    --prompt "quantum entanglement enables" \
    --use-quantum

# 5. Run quantum algorithm demo
python -c "
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
print(f'Input entropy: {diagnostics[\"initial_quantum_entropy\"]:.4f}')
print(f'Gates applied: {diagnostics[\"gates_applied\"]}')
print(f'Entanglement: {diagnostics.get(\"is_entangled\", False)}')
"
```

## Quantum Computing Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Qubit State Manager** | Complex state vectors `|ψ⟩` in 2^n Hilbert space | ✅ Implemented |
| **Quantum Gates** | H, X, Y, Z, CNOT, Toffoli, Phase, T | ✅ Implemented |
| **Entanglement** | Bell states, GHZ states, learned entanglement patterns | ✅ Implemented |
| **Measurement** | Born rule collapse with adaptive protocols | ✅ Implemented |
| **Quantum Circuits** | User-defined gate sequences | ✅ Implemented |
| **Quantum Algorithms** | Grover, QFT, VQE | ✅ Implemented |
| **Quantum Kernel Attention** | Fidelity-based attention mechanism | ✅ Implemented |
| **Tensor Networks** | MPS/Tensor-train FFN layers | ✅ Implemented |
| **QIML** | Quantum-inspired evolutionary optimizer | ✅ Implemented |
| **⏰ Virtual Quantum Clock** | Temporal coherence enforcement | ✅ Implemented |
| **📉 Decoherence Simulation** | 5 physical models (exponential, gaussian, etc.) | ✅ Implemented |
| **⚡ Forced Collapse** | Automatic measurement at deadline | ✅ Implemented |

## Temporal Coherence System

The system includes a **Virtual Quantum Clock** that enforces realistic time-based quantum decoherence:

### Coherence Windows
```python
from voidformer.quantum.temporal_coherence import create_time_aware_processor

# Create processor with 500ms coherence window
processor = create_time_aware_processor(
    base_processor,
    coherence_time_ms=500,      # Quantum coherence lifetime
    virtual_ticks=2000,          # Maximum operations allowed
    decoherence_model="exponential",
    noise_temp=0.01,
)

# Process with temporal enforcement
output, diagnostics = processor(data, log_temporal_evolution=True)

# Check temporal evolution
for step in diagnostics["temporal_evolution"]:
    print(f"Phase: {step['phase']}")
    print(f"  Time: {step['virtual_time_elapsed']*1000:.1f} ms")
    print(f"  Fidelity: {step['current_fidelity']:.4f}")
    print(f"  Decoherence: D(t)={step['decoherence_metrics']['decoherence_factor']:.4f}")
```

### Decoherence Models

| Model | Physics | Formula |
|-------|---------|---------|
| **Exponential** | Random phase accumulation | D(t) = e^(-t/T₂*) |
| **Gaussian** | Quasi-static noise | D(t) = e^(-(t/T₂*)²) |
| **Power Law** | 1/f noise | D(t) = (1 + t/T₂*)^(-α) |
| **Linear** | Constant decay | D(t) = 1 - t/T_coherence |
| **Amplitude Damping** | Energy relaxation | D(t) = √(e^(-t/T₁)) |

### Forced Collapse

When coherence time expires, the system automatically forces measurement:

```python
if diagnostics["forced_collapse"]:
    print(f"⚠️  Forced collapse occurred!")
    print(f"Reason: {diagnostics['forced_measurement_reason']}")
    print(f"Time: {diagnostics['final_status']['virtual_time_elapsed']*1000:.1f} ms")
    print(f"Final fidelity: {diagnostics['final_status']['current_fidelity']:.4f}")
```

**Forced collapse triggers**:
- Time budget exhausted (t ≥ T_coherence)
- Fidelity too low (F < 0.1)
- Virtual tick budget exhausted

## Measurement Collapse Protocols

| Protocol | Behavior | Use Case |
|----------|----------|----------|
| `hard` | Full Born rule sampling → single basis state | Final output, deterministic tasks |
| `soft` | Probability-weighted mixture | Intermediate layers, gradient flow |
| `expectation` | Expectation value ⟨ψ\|O\|ψ⟩ | Analysis, no collapse needed |
| `deferred` | No collapse, keep quantum | Chained quantum operations |
| `entropy_gated` | Adaptive: high entropy→soft, low→hard | Default, uncertainty-aware |

## Quantum Algorithms

The system supports executing quantum algorithms on neural network embeddings:

```python
from voidformer.quantum import QuantumAlgorithm
from voidformer.models import QuantumVoidFormer

model = QuantumVoidFormer(vocab_size=50000, d_model=512, n_qubits_per_token=5)

# Run Grover's search algorithm
output = model.forward(
    tokens,
    quantum_algorithm=QuantumAlgorithm.GROVER_SEARCH
)

# Run Quantum Fourier Transform
output = model.forward(
    tokens,
    quantum_algorithm=QuantumAlgorithm.QUANTUM_FOURIER_TRANSFORM
)
```

## Mathematical Foundations

### Quantum State Representation

Every token embedding is mapped to a quantum state vector in Hilbert space:

```
|ψ⟩ = Σᵢ αᵢ|i⟩    where αᵢ ∈ ℂ, Σ|αᵢ|² = 1
```

Properties:
- **Superposition**: State exists in multiple basis states simultaneously
- **Normalization**: Total probability = 1 (unitary evolution)
- **Phase**: Complex phases enable quantum interference

### Quantum Gates (Unitary Operators)

All gates preserve norm: `U†U = I`

**Hadamard** (creates equal superposition):
```
H = 1/√2 [[1,  1],
          [1, -1]]

H|0⟩ = (|0⟩ + |1⟩)/√2
```

**CNOT** (creates entanglement):
```
CNOT = [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]]

CNOT·(H⊗I)|00⟩ = (|00⟩ + |11⟩)/√2  (Bell state)
```

**Pauli Gates** (rotations):
```
X = [[0, 1],     Y = [[0, -i],     Z = [[1,  0],
     [1, 0]]          [i,  0]]          [0, -1]]
```

### Quantum Measurement

**Born Rule**: Measurement outcome probability
```
P(measuring state |i⟩) = |⟨i|ψ⟩|² = |αᵢ|²
```

Post-measurement state collapse:
```
|ψ⟩ = Σᵢ αᵢ|i⟩  →  |i_measured⟩  with probability |αᵢ|²
```

### Entanglement Measures

**Von Neumann Entropy** (for pure states):
```
S = -Tr(ρ log ρ) = -Σᵢ pᵢ log pᵢ
```

**Concurrence** (2-qubit entanglement):
```
C = 2|α₀α₃ - α₁α₂|  ∈ [0, 1]
```
- C = 0: separable (no entanglement)
- C = 1: maximally entangled (Bell state)

### Quantum Kernel Attention

Replace softmax attention with quantum fidelity:
```
K(x, y) = |⟨ψ(x)|ψ(y)⟩|²

where ψ: ℝ^d → ℂ^(2^n) embeds classical to quantum
```

Attention weights based on quantum state overlap (interference patterns).

### Tensor Network Decomposition

Matrix Product State representation:
```
W = Σ A₁(i₁) A₂(i₂) ... Aₙ(iₙ)
```

where each Aₖ has shape `(bond_dim, local_dim, bond_dim)`.

Compression ratio: `d_in × d_out / (n_cores × bond_dim²)`

See `notebooks/voidformer_math.md` for complete derivations.

## Citation

```
@misc{voidformer_quantum2026,
  title  = {VoidFormer: A Virtual Quantum Computing Simulator for Neural Language Models},
  year   = {2026},
  note   = {Quantum superposition, entanglement, and measurement-based neural architecture}
}
```

---

## Legacy Classical VoidFormer

The original dual-state classical VoidFormer (quantum-inspired but not true quantum computing)
is preserved in `models/voidformer.py` and `layers/` for comparison. The quantum-enhanced
version in `models/quantum_voidformer.py` represents the full quantum computing transformation.
