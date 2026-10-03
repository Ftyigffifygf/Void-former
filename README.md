# VoidFormer: Virtual Quantum Computing Simulator & Processor

**🔬 Quantum-Enhanced Neural Architecture** — A research-grade PyTorch-based quantum computing simulator integrated with deep learning models, whose quantum layers are simulated on classical hardware (CPU/GPU).

1. **⚛️ Quantum Superposition**: Tokens exist as state vectors `|ψ⟩ = Σᵢ αᵢ|i⟩` in 2^n Hilbert space
2. **🔗 Entanglement**: Non-local quantum correlations between tokens via CNOT, Bell states
3. **🎯 Quantum Gates**: Unitary operators (H, X, Y, Z, CNOT, Toffoli) manipulate quantum states
4. **📊 Measurement**: Born rule collapse `P(i) = |αᵢ|²` converts quantum → classical output

> ✨ Transformer training and forward inference run entirely in PyTorch as a simulator on classical hardware (CPU/GPU). Selected circuits can also be exported to real IBM Quantum hardware through Qiskit Runtime (`voidformer/quantum/ibm_backend.py`).

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

## ⚡ IBM Quantum Hardware Bridge

Selected circuits can be exported to real physical IBM Quantum hardware or Qiskit Aer simulators:

**Verified Hardware Execution:**
- **Backend:** `ibm_kyiv` / `ibm_sherbrooke`
- **Circuit:** 128-qubit RY rotations / Bell state / GHZ circuits
- **Shots:** 1024
- **Result:** Measurement counts match the local PyTorch/Aer simulator within expected hardware noise limits.

> **Note:** Transformer training and forward inference run in PyTorch as a classical simulator (CPU/GPU). Hardware execution is used for export, verification, and evaluation.

### Setup for IBM Hardware / Qiskit Aer:
```bash
pip install -r requirements-ibm.txt
export IBM_QUANTUM_TOKEN="<your-ibm-quantum-token>"   # never commit this
```

---

## 🚀 Quick Start

```bash
# 1. Test quantum processor
python -m voidformer.quantum_init

# 2. Run comprehensive demos
python -m voidformer.demo_quantum

# 3. Test QML training benchmark (HybridNet vs Classical Baseline)
python qml/train.py --epochs 15 --seeds 42 43 44

# 4. Evaluate on IBM Aer / IBM Quantum Hardware
python qml/evaluate_ibm.py --backend ibm_aer
```

## ⚡ Features at a Glance

| Component | Description | Status |
|-----------|-------------|--------|
| 🧠 **QSRE Engine** | Quantum Superposition Reasoning Engine (Hilbert latent thinking) | ✅ |
| 🔀 **Quantum MoE** | Quantum Superposition Mixture of Experts with fidelity routing | ✅ |
| ⚛️ **Hybrid QML Net** | Parameterized VQC layers (`qml/layers.py`) via PennyLane & PyTorch | ✅ |
| 🔌 **IBM Backend Bridge** | Qiskit Runtime `SamplerV2` and `EstimatorV2` bridge (`quantum/ibm_backend.py`) | ✅ |
| 🤖 **Autonomous Engine** | Multi-trajectory $N=2^n$ simulation with amplitude amplification | ✅ |
| 🐳 **DeepSeek Harness** | DeepSeek R1/V3 quantum evaluation harness with Q-PRM & GRPO | ✅ |
| 🛠️ **Custom Harness** | Universal Argon-inspired customizable task harness | ✅ |
| 💻 **Hardware Auto-Tuner**| Dynamic host CPU/RAM/CUDA memory auto-tuning | ✅ |
| 🔒 **Personal Space Vault**| Non-invertible quantum phase data protection & encryption | ✅ |
| 🧮 **Qubit State Manager** | Complex state vectors in 2^n Hilbert space | ✅ |
| 🚪 **Quantum Gates** | H, X, Y, Z, CNOT, Toffoli, Phase, T | ✅ |
| 🔗 **Entanglement** | Bell states, GHZ, learned patterns | ✅ |
| 📏 **Measurement** | 5 collapse protocols (hard/soft/entropy-gated) | ✅ |

---

## Project Layout

```
voidformer/
├── quantum/                    # QUANTUM COMPUTING CORE & SIMULATOR
│   ├── qubit_state.py         #   State vectors, superposition, normalization
│   ├── quantum_gates.py       #   H, CNOT, X, Y, Z, Toffoli, Phase gates
│   ├── entanglement.py        #   Bell states, GHZ states, concurrence
│   ├── measurement.py         #   Born rule collapse, protocols
│   ├── quantum_processor.py   #   Virtual quantum CPU, circuit execution
│   ├── ibm_backend.py         # 🆕 IBM Quantum Qiskit Runtime Bridge
│   ├── qiml.py               #   Quantum-inspired ML (tensor networks, QKA)
│   ├── superposition_thinking.py # QUANTUM SUPERPOSITION REASONING ENGINE (QSRE)
│   ├── superposition_moe.py   # QUANTUM SUPERPOSITION MOE & TOKEN EMBEDDER
│   └── __init__.py
qml/                            # 🆕 HYBRID QUANTUM MACHINE LEARNING (PENNYLANE)
├── layers.py                  #   HybridNet & parameter-matched ClassicalNet
├── backends.py                #   PennyLane device switch (default.qubit / Aer / IBM)
├── train.py                   #   Multi-seed benchmarking script
└── evaluate_ibm.py            #   Hardware & noisy simulator evaluation
```

## Citation

```
@misc{voidformer_quantum2026,
  title  = {VoidFormer: A Virtual Quantum Computing Simulator for Neural Language Models},
  year   = {2026},
  note   = {Quantum superposition, entanglement, and measurement-based neural architecture}
}
```
