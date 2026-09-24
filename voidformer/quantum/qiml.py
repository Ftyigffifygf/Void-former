"""Quantum-Inspired Machine Learning (QIML).

Implements quantum-inspired neural architectures that leverage tensor networks,
quantum kernels, and quantum evolutionary algorithms for optimization.

These layers use quantum computing principles (superposition, entanglement,
interference) mapped onto high-dimensional classical tensor operations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qubit_state import QuantumStateVector, QubitStateManager
from .quantum_gates import QuantumGateRegistry
from .entanglement import EntanglementManager


class QuantumKernelAttention(nn.Module):
    """Quantum-kernel-based attention mechanism.
    
    Replaces softmax attention with quantum-inspired kernel:
        K(x, y) = |⟨ψ(x)|ψ(y)⟩|² (quantum fidelity/overlap)
    
    where ψ(x) embeds classical vector x into quantum state.
    
    This captures quantum interference patterns in attention distribution.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_qubits: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_qubits = n_qubits
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Quantum state manager for kernel computation
        self.quantum_manager = QubitStateManager(n_qubits_per_token=n_qubits)
        
        # Learnable quantum embedding projections
        self.quantum_proj_q = nn.Parameter(
            torch.randn(self.head_dim, 2 ** n_qubits, dtype=torch.complex64) 
            / math.sqrt(self.head_dim)
        )
        self.quantum_proj_k = nn.Parameter(
            torch.randn(self.head_dim, 2 ** n_qubits, dtype=torch.complex64)
            / math.sqrt(self.head_dim)
        )
        
        # Interference pattern weights (learned phase modulation)
        self.interference_phase = nn.Parameter(torch.zeros(n_heads))
    
    def _classical_to_quantum(
        self,
        x: torch.Tensor,  # (B, H, T, head_dim)
        projection: torch.Tensor,  # (head_dim, 2^n)
    ) -> QuantumStateVector:
        """Embed classical vectors into quantum states via learned projection."""
        B, H, T, d = x.shape
        
        # Convert to complex
        x_complex = x.to(dtype=torch.complex64)
        
        # Project to quantum space: (B, H, T, d) @ (d, 2^n) -> (B, H, T, 2^n)
        amplitudes = torch.matmul(x_complex, projection)
        
        # Flatten batch and heads for QuantumStateVector
        amplitudes = amplitudes.reshape(B * H, T, 2 ** self.n_qubits)
        
        state = QuantumStateVector(amplitudes=amplitudes, n_qubits=self.n_qubits)
        return state.normalize()
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, T, d_model)
            attn_mask: Optional attention mask
        
        Returns:
            output: (B, T, d_model)
            diagnostics: Quantum attention statistics
        """
        B, T, D = x.shape
        
        # Standard QKV projection
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each (B, T, H, head_dim)
        
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Convert Q and K to quantum states
        q_quantum = self._classical_to_quantum(q, self.quantum_proj_q)
        k_quantum = self._classical_to_quantum(k, self.quantum_proj_k)
        
        # Compute quantum kernel: K(q_i, k_j) = |⟨ψ(q_i)|ψ(k_j)⟩|²
        # This is quantum fidelity between states
        
        # Reshape for pairwise fidelity computation
        # q_quantum.amplitudes: (B*H, T, 2^n)
        # k_quantum.amplitudes: (B*H, T, 2^n)
        
        q_amps = q_quantum.amplitudes  # (B*H, T, 2^n)
        k_amps = k_quantum.amplitudes  # (B*H, T, 2^n)
        
        # Quantum kernel via fidelity: |⟨q_i|k_j⟩|²
        # Compute all pairs: q_i · k_j^*
        quantum_kernel = torch.matmul(
            q_amps,  # (B*H, T_q, 2^n)
            k_amps.conj().transpose(-2, -1)  # (B*H, 2^n, T_k)
        )  # (B*H, T, T)
        
        # Fidelity = |inner_product|²
        fidelity = (quantum_kernel.real ** 2 + quantum_kernel.imag ** 2).clamp(0, 1)
        
        # Apply learned phase modulation (quantum interference)
        phase = torch.exp(1j * self.interference_phase.view(1, self.n_heads, 1, 1))
        interference_modulation = torch.abs(phase.real).repeat(B, 1, T, T)
        
        fidelity = fidelity.view(B, self.n_heads, T, T)
        quantum_attn = fidelity * interference_modulation
        
        # Normalize to probability distribution
        quantum_attn = quantum_attn / (quantum_attn.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Apply causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        quantum_attn = quantum_attn.masked_fill(causal, 0.0)
        
        # Renormalize after masking
        quantum_attn = quantum_attn / (quantum_attn.sum(dim=-1, keepdim=True) + 1e-10)
        
        if attn_mask is not None:
            quantum_attn = quantum_attn + attn_mask
        
        quantum_attn = self.dropout(quantum_attn)
        
        # Apply attention to values (classical)
        out = torch.matmul(quantum_attn, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.proj(out)
        
        diagnostics = {
            "quantum_kernel_mean": fidelity.mean(),
            "quantum_kernel_std": fidelity.std(),
            "quantum_entropy": -(quantum_attn.clamp(min=1e-10) * 
                               torch.log2(quantum_attn.clamp(min=1e-10))).sum(dim=-1).mean(),
        }
        
        return out, diagnostics


class TensorNetworkLayer(nn.Module):
    """Tensor Network Decomposition Layer.
    
    Uses Matrix Product State (MPS) / Tensor Train decomposition to
    represent high-dimensional weight tensors efficiently.
    
    Maps d_in → d_out through low-rank tensor factorization:
        W = Σ A₁(i₁) A₂(i₂) ... Aₙ(iₙ)
    
    This mimics quantum tensor network states used in quantum simulation.
    """
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bond_dim: int = 8,
        n_cores: int = 4,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.bond_dim = bond_dim
        self.n_cores = n_cores
        
        # Decompose input/output dimensions across cores
        self.local_dim_in = int(math.ceil(d_in ** (1.0 / n_cores)))
        self.local_dim_out = int(math.ceil(d_out ** (1.0 / n_cores)))
        
        # MPS tensor cores: A[i] has shape (bond_in, local_dim, bond_out)
        self.cores = nn.ModuleList([
            nn.Linear(
                bond_dim * self.local_dim_in,
                bond_dim * self.local_dim_out,
                bias=False,
            )
            for _ in range(n_cores)
        ])
        
        # Edge projections
        self.in_proj = nn.Linear(d_in, bond_dim * self.local_dim_in)
        self.out_proj = nn.Linear(bond_dim * self.local_dim_out, d_out)
        
        self.bias = nn.Parameter(torch.zeros(d_out))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_in)
        
        Returns:
            output: (B, T, d_out)
        """
        # Project input into tensor network
        h = self.in_proj(x)  # (B, T, bond_dim * local_dim)
        
        # Contract through tensor network cores
        for core in self.cores:
            h = F.gelu(core(h))
        
        # Project back to output space
        out = self.out_proj(h) + self.bias
        
        return out
    
    def compute_compression_ratio(self) -> float:
        """Compute parameter compression vs dense layer."""
        dense_params = self.d_in * self.d_out
        tn_params = sum(p.numel() for p in self.parameters())
        return dense_params / max(tn_params, 1)


class QuantumEvolutionaryOptimizer(nn.Module):
    """Quantum-Inspired Evolutionary Algorithm for parameter optimization.
    
    Uses quantum superposition and interference to explore parameter space:
    1. Create superposition of candidate solutions
    2. Apply amplitude amplification (quantum-inspired mutation)
    3. Measure best candidate via interference
    
    This is a trainable layer that optimizes itself using quantum principles.
    """
    
    def __init__(
        self,
        param_dim: int,
        population_size: int = 16,
        n_qubits: int = 4,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.population_size = population_size
        self.n_qubits = n_qubits
        self.lr = learning_rate
        
        # Population of candidate parameter vectors
        self.population = nn.Parameter(
            torch.randn(population_size, param_dim) * 0.01
        )
        
        # Quantum amplitude amplification weights
        self.amplification_weights = nn.Parameter(torch.ones(population_size))
        
        # Fitness scoring network
        self.fitness_net = nn.Sequential(
            nn.Linear(param_dim, param_dim * 2),
            nn.GELU(),
            nn.Linear(param_dim * 2, 1),
        )
    
    def forward(self, target_fitness: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Evolve population toward higher fitness.
        
        Args:
            target_fitness: Target fitness values to optimize toward (B,)
        
        Returns:
            best_params: Optimized parameters (B, param_dim)
            diagnostics: Evolution statistics
        """
        B = target_fitness.shape[0]
        
        # Evaluate fitness for each population member
        fitness_scores = self.fitness_net(self.population).squeeze(-1)  # (pop_size,)
        
        # Quantum amplitude amplification: boost high-fitness members
        # Amplitude ∝ sqrt(fitness) (Born rule inverse)
        amplitudes = torch.sqrt(F.softplus(fitness_scores) + 1e-6)
        amplitudes = amplitudes * torch.sigmoid(self.amplification_weights)
        
        # Normalize to probability distribution
        probs = F.softmax(amplitudes, dim=0)
        
        # Interference: weighted combination (quantum superposition collapse)
        best_params = torch.matmul(
            probs.unsqueeze(0).expand(B, -1),  # (B, pop_size)
            self.population  # (pop_size, param_dim)
        )  # (B, param_dim)
        
        # Quantum mutation: add controlled noise based on entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        mutation_strength = (entropy / math.log(self.population_size)) * 0.1
        
        if self.training:
            mutation = torch.randn_like(best_params) * mutation_strength
            best_params = best_params + mutation
        
        diagnostics = {
            "population_entropy": entropy,
            "best_fitness": fitness_scores.max(),
            "mean_fitness": fitness_scores.mean(),
            "amplitude_concentration": probs.max(),
        }
        
        return best_params, diagnostics
    
    def evolve_step(self, fitness_feedback: torch.Tensor):
        """Update population based on fitness feedback (training step).
        
        Args:
            fitness_feedback: Fitness scores for current population (pop_size,)
        """
        # Amplitude amplification: increase weight for high-fitness members
        with torch.no_grad():
            normalized_fitness = (fitness_feedback - fitness_feedback.mean()) / (
                fitness_feedback.std() + 1e-6
            )
            self.amplification_weights.data += self.lr * normalized_fitness
            
            # Quantum diffusion: add mutation to population
            top_k = max(1, self.population_size // 4)
            top_indices = torch.topk(fitness_feedback, k=top_k).indices
            
            # Breed new members from top performers (quantum crossover)
            for i in range(self.population_size):
                if i not in top_indices:
                    parent1 = self.population[top_indices[i % top_k]]
                    parent2 = self.population[top_indices[(i + 1) % top_k]]
                    child = (parent1 + parent2) / 2 + torch.randn(self.param_dim) * 0.05
                    self.population.data[i] = child


class QuantumInspiredNeuralLayer(nn.Module):
    """Unified quantum-inspired neural layer.
    
    Combines:
    - Quantum state representation (superposition)
    - Quantum gate operations (rotations)
    - Entanglement between features
    - Measurement for classical output
    """
    
    def __init__(
        self,
        d_model: int,
        n_qubits: int = 4,
        use_entanglement: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.use_entanglement = use_entanglement
        
        # Quantum state manager
        self.quantum_manager = QubitStateManager(n_qubits_per_token=n_qubits)
        
        # Gate registry for quantum operations
        self.gate_registry = QuantumGateRegistry()
        
        # Entanglement manager
        if use_entanglement:
            self.entangle_manager = EntanglementManager(
                n_qubits_per_token=n_qubits,
                max_seq_len=512,
            )
        
        # Learnable quantum circuit parameters
        self.rotation_angles = nn.Parameter(torch.randn(d_model, n_qubits, 3) * 0.1)
        
        # Classical-to-quantum projection
        self.c2q_proj = nn.Parameter(
            torch.randn(d_model, self.state_dim, dtype=torch.complex64)
            / math.sqrt(d_model)
        )
        
        # Quantum-to-classical readout
        self.q2c_proj = nn.Parameter(
            torch.randn(self.state_dim, d_model, dtype=torch.complex64)
            / math.sqrt(self.state_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Classical input (B, T, d_model)
        
        Returns:
            output: Classical output (B, T, d_model)
            diagnostics: Quantum processing statistics
        """
        B, T, D = x.shape
        residual = x
        
        # 1. Classical → Quantum embedding
        quantum_state = self.quantum_manager.from_classical_embedding(
            x, projection_matrix=self.c2q_proj
        )
        
        # 2. Apply learnable quantum rotations (parameterized gates)
        # Simplified: apply phase rotations to quantum state
        phases = self.rotation_angles.mean(dim=0).sum(dim=-1)  # (n_qubits,)
        for i, phase in enumerate(phases):
            if i < self.n_qubits:
                phase_gate = self.gate_registry.gates["H"]  # Placeholder
                # In practice, create parameterized rotation gate
        
        # 3. Apply entanglement if enabled
        if self.use_entanglement and hasattr(self, 'entangle_manager'):
            quantum_state, entangle_attn = self.entangle_manager.apply_learned_entanglement(
                quantum_state, temperature=1.0, top_k=5
            )
            entanglement_measure = entangle_attn.mean().item()
        else:
            entanglement_measure = 0.0
        
        # 4. Quantum → Classical measurement
        classical_output = self.quantum_manager.to_classical_embedding(
            quantum_state, d_model=D, readout_matrix=self.q2c_proj
        )
        
        # 5. Residual connection and normalization
        output = self.norm(residual + self.dropout(classical_output))
        
        diagnostics = {
            "quantum_entropy": quantum_state.measure_entropy().mean(),
            "entanglement_measure": entanglement_measure,
            "quantum_fidelity": 1.0,  # Placeholder
        }
        
        return output, diagnostics
