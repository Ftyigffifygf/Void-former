"""Training Entry Point for VoidFormer models (Classical & Quantum).

Usage:
    python -m voidformer.train --config voidformer/configs/tiny.yaml --model-type quantum --steps 100
"""

from __future__ import annotations

import argparse
import sys
import torch
from torch.utils.data import DataLoader

from voidformer.datasets import ToyCorpus, CharTokenizer, build_tokenizer
from voidformer.models import VoidFormerModel, QuantumVoidFormer
from voidformer.training import Trainer, VoidFormerLosses
from voidformer.utils import load_config, set_seed, get_logger


def main():
    parser = argparse.ArgumentParser(description="Train VoidFormer (Classical or Quantum)")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to config yaml")
    parser.add_argument("--model-type", type=str, choices=["classical", "quantum"], default="quantum", help="Model type to train")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))
    log = get_logger("voidformer.train")

    tokenizer = build_tokenizer(cfg)
    vocab_size = tokenizer.vocab_size
    cfg["model"]["vocab_size"] = vocab_size

    # Build dataset and dataloader
    ds = ToyCorpus(tokenizer=tokenizer, block_size=cfg["model"]["max_seq_len"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    if args.model_type == "quantum":
        log.info("Initializing QuantumVoidFormer model...")
        model = QuantumVoidFormer(
            vocab_size=vocab_size,
            d_model=cfg["model"]["d_model"],
            n_layers=cfg["model"]["n_layers"],
            n_heads=cfg["model"]["n_heads"],
            n_qubits_per_token=cfg["model"].get("n_qubits_per_token", 3),
            max_seq_len=cfg["model"]["max_seq_len"],
            collapse_protocol=cfg["model"].get("collapse_protocol", "entropy_gated"),
            enable_entanglement=cfg["model"].get("enable_entanglement", True),
        )
    else:
        log.info("Initializing classical VoidFormerModel...")
        model = VoidFormerModel(**cfg["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    log.info(f"Starting training ({args.model_type}) for {args.steps} steps...")
    model.train()
    step = 0
    dl_iter = iter(dl)

    while step < args.steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        x, y = batch["input_ids"], batch["targets"]

        optimizer.zero_grad()
        out = model(x)
        logits = out.logits
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        step += 1
        if step % max(1, args.steps // 5) == 0 or step == args.steps:
            log.info(f"Step {step}/{args.steps} | Loss: {loss.item():.4f}")

    log.info("Training complete!")


if __name__ == "__main__":
    main()
