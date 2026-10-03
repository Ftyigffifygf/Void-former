"""Distillation Loss Objectives for Teacher-Student Knowledge Transfer in VoidFormer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Distillation loss combining text-level cross-entropy and logit KL divergence."""

    def __init__(self, temperature: float = 2.0, alpha_kl: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha_kl = alpha_kl

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Student model logits [Batch, Sequence_Len, Vocab_Size]
            targets: Target token IDs [Batch, Sequence_Len]
            teacher_logits: Optional teacher model logits [Batch, Sequence_Len, Vocab_Size]

        Returns:
            distill_loss: Scalar distillation loss
        """
        # Text-level task loss (cross-entropy on teacher answers/targets)
        s_logits_shift = student_logits[:, :-1, :].contiguous()
        tgt_shift = targets[:, 1:].contiguous()
        text_loss = F.cross_entropy(
            s_logits_shift.reshape(-1, s_logits_shift.size(-1)),
            tgt_shift.reshape(-1),
        )

        if teacher_logits is not None and teacher_logits.shape == student_logits.shape:
            # KL divergence distillation loss on matching logits
            s_log_probs = F.log_softmax(s_logits_shift / self.temperature, dim=-1)
            t_probs = F.softmax(teacher_logits[:, :-1, :].contiguous() / self.temperature, dim=-1)

            kl_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (self.temperature ** 2)
            return (1.0 - self.alpha_kl) * text_loss + self.alpha_kl * kl_loss

        return text_loss
