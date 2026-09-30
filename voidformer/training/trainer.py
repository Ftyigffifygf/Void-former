"""Trainer — minimal, AMP-aware, TensorBoard-aware."""

from __future__ import annotations

import math
import os
import time
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models import VoidFormerModel
from ..utils.logging import get_logger
from .losses import VoidFormerLosses


class Trainer:
    def __init__(
        self,
        model: VoidFormerModel,
        loss_fn: VoidFormerLosses,
        train_loader: Iterable,
        cfg: dict,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.cfg = cfg
        t = cfg["training"]
        self.device = self._resolve_device(t.get("device", "auto"))
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=t["lr"],
            weight_decay=t.get("weight_decay", 0.0),
        )
        self.warmup = t.get("warmup_steps", 0)
        self.total_steps = t.get("total_steps", 1000)
        self.grad_clip = t.get("grad_clip", 1.0)
        self.log_every = t.get("log_every", 50)
        self.amp = bool(t.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.log = get_logger("voidformer.train")

        self.tb = None
        if cfg.get("experiment", {}).get("tensorboard", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                outdir = cfg["experiment"]["output_dir"]
                os.makedirs(outdir, exist_ok=True)
                self.tb = SummaryWriter(outdir)
            except Exception as e:                                          # noqa: BLE001
                self.log.warning("TensorBoard disabled: %s", e)

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    def _lr_at(self, step: int) -> float:
        base = self.cfg["training"]["lr"]
        if step < self.warmup:
            return base * step / max(1, self.warmup)
        progress = (step - self.warmup) / max(1, self.total_steps - self.warmup)
        return 0.5 * base * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def fit(self, max_steps: int | None = None) -> dict:
        steps = max_steps or self.total_steps
        emb_w = self.model.embed.classical_emb.weight
        self.model.train()
        step = 0
        t0 = time.time()
        history: list[dict] = []
        loader_iter = iter(self.train_loader)
        while step < steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)
            ids = batch["input_ids"].to(self.device)
            tgt = batch.get("targets", ids).to(self.device)

            for g in self.optim.param_groups:
                g["lr"] = self._lr_at(step)

            self.optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                out = self.model(ids, return_diagnostics=True)
                loss, log = self.loss_fn(out, tgt, emb_w)

            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optim.step()

            if step % self.log_every == 0 or step == steps - 1:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in log.items())
                self.log.info("step %d | %s | lr=%.2e", step, msg, self._lr_at(step))
                if self.tb is not None:
                    for k, v in log.items():
                        self.tb.add_scalar(k, v.item(), step)
                    self.tb.add_scalar("lr", self._lr_at(step), step)

            history.append({k: float(v.item()) for k, v in log.items()})
            step += 1

        if self.tb is not None:
            self.tb.flush()

        elapsed = time.time() - t0
        self.log.info("training done — %d steps in %.1fs", steps, elapsed)
        return {"history": history, "elapsed": elapsed}
