"""Variational Quantum Circuit (VQC) Autograd Layer.

Provides parameter-shift autograd function and classical <-> quantum projection layer.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .qubit_state import QuantumStateVector
from .quantum_gates import RXGate, RYGate, RZGate, CNOTGate


def compute_shot_expectation_and_variance(
    counts: dict[str, int],
    n_qubits: int,
    shots: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Pauli-Z expectation values and shot-noise variance estimates from measurement counts."""
    total_shots = shots or sum(counts.values()) or 1
    exp_vals = torch.zeros(n_qubits, dtype=dtype, device=device)

    for bitstring, count in counts.items():
        clean_bitstring = bitstring.replace(" ", "")
        prob = count / total_shots
        for q in range(n_qubits):
            bit = int(clean_bitstring[-(q + 1)])
            sign = 1.0 if bit == 0 else -1.0
            exp_vals[q] += sign * prob

    variances = (1.0 - exp_vals ** 2) / float(total_shots)
    return exp_vals, variances


def run_vqc_statevector(
    angles: torch.Tensor,
    n_qubits: int,
    n_layers: int,
) -> torch.Tensor:
    """Execute VQC statevector simulation for a single set of rotation angles."""
    dim = 2 ** n_qubits
    amps = torch.zeros(1, 1, dim, dtype=torch.complex128, device=angles.device)
    amps[0, 0, 0] = 1.0 + 0j
    state = QuantumStateVector(amplitudes=amps, n_qubits=n_qubits)

    idx = 0
    cnot = CNOTGate()
    for l in range(n_layers):
        for q in range(n_qubits):
            rx = RXGate(theta=angles[idx])
            ry = RYGate(theta=angles[idx + 1])
            rz = RZGate(theta=angles[idx + 2])
            idx += 3
            state = rx.apply(state, [q])
            state = ry.apply(state, [q])
            state = rz.apply(state, [q])

        if n_qubits > 1:
            for q in range(n_qubits - 1):
                state = cnot.apply(state, [q, q + 1])

    probs = state.probabilities.squeeze(0).squeeze(0)
    exp_vals = torch.zeros(n_qubits, dtype=angles.dtype, device=angles.device)

    for q in range(n_qubits):
        mask = torch.tensor(
            [1.0 if ((k >> q) & 1) == 0 else -1.0 for k in range(dim)],
            dtype=angles.dtype,
            device=angles.device,
        )
        exp_vals[q] = torch.sum(probs * mask)

    return exp_vals


def run_vqc_aer(
    angles: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    shots: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute VQC on Qiskit Aer simulator with measurement counts and shot noise variance."""
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit_aer import AerSimulator

    qc = QiskitCircuit(n_qubits)
    idx = 0
    angles_list = angles.detach().cpu().numpy().tolist()

    for l in range(n_layers):
        for q in range(n_qubits):
            qc.rx(angles_list[idx], q)
            qc.ry(angles_list[idx + 1], q)
            qc.rz(angles_list[idx + 2], q)
            idx += 3
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

    qc.measure_all()
    sim = AerSimulator()
    job = sim.run(qc, shots=shots)
    counts = job.result().get_counts(qc)

    exp_vals, variances = compute_shot_expectation_and_variance(
        counts, n_qubits, shots=shots, dtype=angles.dtype, device=angles.device
    )
    return exp_vals, variances


class VQCAutogradFunction(torch.autograd.Function):
    """PyTorch Autograd wrapper for VQC execution using parameter-shift rule."""

    @staticmethod
    def forward(
        ctx,
        angles: torch.Tensor,
        n_qubits: int,
        n_layers: int,
        backend: str = "statevector",
        shots: int = 1024,
    ) -> torch.Tensor:
        ctx.save_for_backward(angles)
        ctx.n_qubits = n_qubits
        ctx.n_layers = n_layers
        ctx.backend = backend
        ctx.shots = shots
        ctx.orig_shape = angles.shape

        angles_flat = angles.reshape(-1, angles.shape[-1])
        N, K = angles_flat.shape

        out_list = []
        for i in range(N):
            if backend == "aer":
                exp_vals, _ = run_vqc_aer(angles_flat[i], n_qubits, n_layers, shots=shots)
            else:
                exp_vals = run_vqc_statevector(angles_flat[i], n_qubits, n_layers)
            out_list.append(exp_vals)

        out_tensor = torch.stack(out_list, dim=0)
        res_shape = list(angles.shape[:-1]) + [n_qubits]
        return out_tensor.reshape(*res_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (angles,) = ctx.saved_tensors
        n_qubits = ctx.n_qubits
        n_layers = ctx.n_layers
        backend = ctx.backend
        shots = ctx.shots

        angles_flat = angles.reshape(-1, angles.shape[-1])
        grad_flat = grad_output.reshape(-1, n_qubits)
        N, K = angles_flat.shape

        shift = math.pi / 2.0
        grad_angles = torch.zeros_like(angles_flat)

        for i in range(N):
            curr_angles = angles_flat[i]
            for k in range(K):
                shift_plus = curr_angles.clone()
                shift_plus[k] += shift
                if backend == "aer":
                    exp_plus, _ = run_vqc_aer(shift_plus, n_qubits, n_layers, shots=shots)
                else:
                    exp_plus = run_vqc_statevector(shift_plus, n_qubits, n_layers)

                shift_minus = curr_angles.clone()
                shift_minus[k] -= shift
                if backend == "aer":
                    exp_minus, _ = run_vqc_aer(shift_minus, n_qubits, n_layers, shots=shots)
                else:
                    exp_minus = run_vqc_statevector(shift_minus, n_qubits, n_layers)

                j_k = (exp_plus - exp_minus) / 2.0
                grad_angles[i, k] = torch.sum(grad_flat[i] * j_k)

        return grad_angles.reshape(ctx.orig_shape), None, None, None, None


def execute_vqc(
    angles: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    backend: str = "statevector",
    shots: int = 1024,
) -> torch.Tensor:
    """Execute VQC through autograd wrapper using parameter-shift gradients."""
    return VQCAutogradFunction.apply(angles, n_qubits, n_layers, backend, shots)


class VQCLayer(nn.Module):
    """Classical ↔ Quantum Projection Variational Quantum Circuit Layer."""

    def __init__(
        self,
        d_model: int,
        n_vqc_qubits: int = 4,
        n_vqc_layers: int = 2,
        backend: str = "statevector",
        shots: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_vqc_qubits = n_vqc_qubits
        self.n_vqc_layers = n_vqc_layers
        self.backend = backend
        self.shots = shots

        self.k_angles = n_vqc_layers * n_vqc_qubits * 3
        self.down_proj = nn.Linear(d_model, self.k_angles)
        self.up_proj = nn.Linear(n_vqc_qubits, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        residual = x
        angles = self.down_proj(x)
        exp_vals = execute_vqc(
            angles=angles,
            n_qubits=self.n_vqc_qubits,
            n_layers=self.n_vqc_layers,
            backend=self.backend,
            shots=self.shots,
        )
        vqc_out = self.up_proj(exp_vals)
        out = self.norm(residual + self.dropout(vqc_out))

        diagnostics = {
            "vqc_mean_expectation": exp_vals.mean().item(),
            "vqc_std_expectation": exp_vals.std().item() if exp_vals.numel() > 1 else 0.0,
            "backend": self.backend,
            "n_vqc_qubits": self.n_vqc_qubits,
            "n_vqc_layers": self.n_vqc_layers,
            "shots": self.shots,
        }

        return out, diagnostics
