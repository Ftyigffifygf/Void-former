"""VoidFormer training objectives.

Total loss:
    L = L_lm
      + λ_e · L_entropy_preservation
      + λ_s · L_semantic_consistency
      + λ_c · L_collapse_reg
      + λ_i · L_interference_stability
      + λ_a · L_ambiguity_retention
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_entropy_preservation: float = 0.05
    lambda_semantic_consistency: float = 0.10
    lambda_collapse_reg: float = 0.02
    lambda_interference_stability: float = 0.01
    lambda_ambiguity_retention: float = 0.05
    lambda_geodesic: float = 0.02
    geodesic_target: float = 1.0


class VoidFormerLosses(nn.Module):
    def __init__(self, weights: LossWeights | dict) -> None:
        super().__init__()
        if isinstance(weights, dict):
            weights = LossWeights(
                **{k: weights[k] for k in weights if k in LossWeights.__dataclass_fields__}
            )
        self.w = weights

    def forward(
        self,
        out: Any,
        targets: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # A. Language modelling (shift targets internally)
        logits = out.logits[:, :-1, :].contiguous()
        tgt = targets[:, 1:].contiguous()
        l_lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        device = l_lm.device

        l_ent = torch.tensor(0.0, device=device)
        l_amb = torch.tensor(0.0, device=device)
        l_col = torch.tensor(0.0, device=device)
        l_int = torch.tensor(0.0, device=device)

        diagnostics_list = getattr(out, "layer_diagnostics", None) or getattr(out, "diagnostics", [])
        n_layers = max(len(diagnostics_list), 1)

        for i, d in enumerate(diagnostics_list):
            depth = (i + 1) / n_layers
            if isinstance(d, dict) and "void_entropy" in d and "tau" in d and "coefs" in d:
                ent = d["void_entropy"].mean()
                tau = d["tau"]
                coefs = d["coefs"]

                # B. Entropy preservation
                l_ent = l_ent + (1.0 - depth) * (-ent)
                # F. Ambiguity retention
                l_amb = l_amb + (1.0 - depth) * tau.mean()
                # D. Collapse regularisation
                mid_w = 1.0 - abs(depth - 0.5) * 2.0
                l_col = l_col + mid_w * ((tau - 0.5) ** 2).mean()
                # E. Interference stability
                l_int = l_int + coefs.pow(2).mean()

        l_ent = l_ent / n_layers
        l_amb = l_amb / n_layers
        l_col = l_col / n_layers
        l_int = l_int / n_layers

        # C. Semantic consistency
        classical_states = getattr(out, "classical_output", getattr(out, "classical_states", None))
        if classical_states is not None and classical_states.shape[1] > 1:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)                                  # (B, T-1)
            emb = F.embedding(pred, embedding_weight)                          # (B, T-1, d)
            hidden = classical_states[:, :-1, :]
            l_sem = 1.0 - F.cosine_similarity(hidden, emb, dim=-1).mean()
        else:
            l_sem = torch.tensor(0.0, device=device)

        # G. Riemannian geodesic regulariser
        l_geo = torch.tensor(0.0, device=device)
        geodesic_dist = getattr(out, "geodesic_distance", None)
        if geodesic_dist is not None:
            gd = geodesic_dist                                              # (B, T)
            tgt_val = float(self.w.geodesic_target)
            l_geo = ((gd - tgt_val) ** 2).mean() + 0.1 * gd.var(dim=1).mean()

        total = (
            l_lm
            + self.w.lambda_entropy_preservation * l_ent
            + self.w.lambda_semantic_consistency * l_sem
            + self.w.lambda_collapse_reg * l_col
            + self.w.lambda_interference_stability * l_int
            + self.w.lambda_ambiguity_retention * l_amb
            + self.w.lambda_geodesic * l_geo
        )
        return total, {
            "loss/total": total.detach(),
            "loss/lm": l_lm.detach(),
            "loss/entropy_preservation": l_ent.detach(),
            "loss/semantic_consistency": l_sem.detach(),
            "loss/collapse_reg": l_col.detach(),
            "loss/interference_stability": l_int.detach(),
            "loss/ambiguity_retention": l_amb.detach(),
            "loss/geodesic": l_geo.detach(),
        }
