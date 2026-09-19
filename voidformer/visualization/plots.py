"""Matplotlib visualisation utilities for VoidFormer diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def plot_entropy_heatmap(diagnostics: list[dict], out_path: str | Path) -> Path:
    """Per-layer mean void-attention entropy across token positions."""
    L = len(diagnostics)
    T = diagnostics[0]["void_entropy"].shape[-1]
    mat = np.zeros((L, T))
    for i, d in enumerate(diagnostics):
        ent = _to_np(d["void_entropy"]).mean(axis=(0, 1))   # avg over batch+heads
        mat[i] = ent
    fig, ax = plt.subplots(figsize=(10, 0.4 * L + 1))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    ax.set_title("Void-attention entropy per layer")
    fig.colorbar(im, ax=ax, label="entropy (nats)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_collapse_trajectory(diagnostics: list[dict], out_path: str | Path) -> Path:
    """Per-layer mean τ — collapse strength across depth."""
    L = len(diagnostics)
    taus = np.array([_to_np(d["tau"]).mean() for d in diagnostics])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, L + 1), taus, marker="o", color="#c0392b")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean τ (collapse strength)")
    ax.set_ylim(0, 1)
    ax.set_title("Adaptive collapse trajectory across depth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_void_state_evolution(void_states_per_layer: list[torch.Tensor], out_path: str | Path) -> Path:
    """L2 norm of the void state across layers (avg over batch & token)."""
    norms = [float(v.norm(dim=-1).mean().item()) for v in void_states_per_layer]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(norms) + 1), norms, marker="s", color="#2980b9")
    ax.set_xlabel("layer")
    ax.set_ylabel("‖void-state‖₂")
    ax.set_title("Void-state magnitude evolution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_route_weights(diagnostics: list[dict], out_path: str | Path) -> Path:
    """Mean dynamic-routing weights per layer (classical / void / merged)."""
    rows = []
    for d in diagnostics:
        rw = d.get("route_weights")
        if rw is None:
            continue
        rows.append(_to_np(rw).mean(axis=(0, 1)))
    if not rows:
        return Path(out_path)
    mat = np.stack(rows, axis=0)                                # (L, 3)
    L = mat.shape[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(
        range(1, L + 1),
        mat[:, 0], mat[:, 1], mat[:, 2],
        labels=["classical", "void", "merged"],
        colors=["#2c3e50", "#9b59b6", "#16a085"],
        alpha=0.85,
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("routing weight")
    ax.set_title("Dynamic routing across depth")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_interference_coefs(diagnostics: list[dict], out_path: str | Path) -> Path:
    """Mean α, β, γ+, γ-, δ per layer."""
    names = ["α (classical)", "β (void)", "γ+ (i+)", "γ- (i-)", "δ (mult)"]
    L = len(diagnostics)
    mat = np.zeros((L, 5))
    for i, d in enumerate(diagnostics):
        mat[i] = _to_np(d["coefs"]).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(5):
        ax.plot(range(1, L + 1), mat[:, k], marker="o", label=names[k])
    ax.set_xlabel("layer")
    ax.set_ylabel("coefficient magnitude")
    ax.set_title("Interference engine coefficients per layer")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_geodesic_path(
    geodesic_segments: list,
    geodesic_distance,
    out_path: str | Path,
) -> Path:
    """Per-token Riemannian arc-length per layer, and cumulative geodesic.

    Visualises the discrete approximation of
        d(I,U) = ∫₀¹ √( Σ_ij I_ij ẋⁱ ẋʲ ) dt
    where x(t) is the layer-indexed trajectory of the collapsed state.
    """
    if not geodesic_segments:
        return Path(out_path)
    # (L, T)  — mean over batch
    segs = np.stack([_to_np(s).mean(axis=0) for s in geodesic_segments], axis=0)
    cum = np.cumsum(segs, axis=0)
    L, T = segs.shape
    total = _to_np(geodesic_distance).mean(axis=0) if geodesic_distance is not None else cum[-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    im0 = axes[0].imshow(segs, aspect="auto", cmap="viridis")
    axes[0].set_xlabel("token position")
    axes[0].set_ylabel("layer ℓ")
    axes[0].set_title("‖Δx‖_{I_ℓ}  per layer")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(cum, aspect="auto", cmap="magma")
    axes[1].set_xlabel("token position")
    axes[1].set_ylabel("layer ℓ")
    axes[1].set_title("cumulative geodesic  d(I,U)|_{0..ℓ}")
    fig.colorbar(im1, ax=axes[1])

    axes[2].plot(np.arange(T), total, color="#e67e22", marker="o", ms=3)
    axes[2].set_xlabel("token position")
    axes[2].set_ylabel("total geodesic d(I, U)")
    axes[2].set_title("Per-token Riemannian path length")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def plot_geodesic_distribution(geodesic_distance, out_path: str | Path) -> Path:
    """Histogram of per-token geodesic lengths across the batch."""
    if geodesic_distance is None:
        return Path(out_path)
    arr = _to_np(geodesic_distance).reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=40, color="#8e44ad", alpha=0.85, edgecolor="white")
    ax.axvline(float(np.mean(arr)), color="#c0392b", linestyle="--", label=f"mean={np.mean(arr):.3f}")
    ax.set_xlabel("d(I, U)")
    ax.set_ylabel("# tokens")
    ax.set_title("Distribution of Riemannian path lengths")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)
