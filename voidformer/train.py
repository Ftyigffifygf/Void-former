"""Training entry point for Quantum VoidFormer & Classical VoidFormer.

Usage:
    python -m voidformer.train --config voidformer/configs/tiny.yaml --model-type quantum --steps 100
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from voidformer.datasets import build_tokenizer, ToyCorpus
from voidformer.models import VoidFormerModel
from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.training import Trainer, VoidFormerLosses
from voidformer.utils import load_config, set_seed, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VoidFormer model")
    parser.add_argument("--config", default="voidformer/configs/tiny.yaml", help="Path to config YAML file")
    parser.add_argument("--model-type", choices=["quantum", "classical"], default="quantum", help="Model type")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    set_seed(cfg["training"].get("seed", 42))
    log = get_logger("voidformer.train")

    tok = build_tokenizer(cfg)
    cfg["model"]["vocab_size"] = tok.vocab_size

    log.info("Building dataset...")
    corpus = ToyCorpus(tokenizer=tok, block_size=cfg["model"].get("max_seq_len", 128))
    train_loader = DataLoader(corpus, batch_size=cfg["training"].get("batch_size", 4), shuffle=True)

    log.info(f"Instantiating {args.model_type} model...")
    if args.model_type == "quantum":
        model = QuantumVoidFormer(
            vocab_size=cfg["model"]["vocab_size"],
            d_model=cfg["model"].get("d_model", 64),
            n_layers=cfg["model"].get("n_layers", 2),
            n_heads=cfg["model"].get("n_heads", 2),
            d_ff=cfg["model"].get("d_ff", 128),
            max_seq_len=cfg["model"].get("max_seq_len", 128),
        )
    else:
        model = VoidFormerModel(**cfg["model"])

    log.info(f"Model parameters: {model.num_params():,}")

    loss_fn = VoidFormerLosses(cfg.get("loss", {}))

    if args.model_type == "quantum":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
        criterion = torch.nn.CrossEntropyLoss()

        total_steps = cfg["training"]["total_steps"]
        log.info(f"Starting quantum training for {total_steps} steps...")
        step = 0
        loader_iter = iter(train_loader)

        while step < total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            ids = batch["input_ids"].to(device)
            targets = batch.get("targets", ids).to(device)

            optimizer.zero_grad()
            out = model(ids, return_diagnostics=True)
            loss = criterion(out.logits.view(-1, cfg["model"]["vocab_size"]), targets.view(-1))
            loss.backward()
            optimizer.step()

            if step % cfg["training"].get("log_every", 10) == 0 or step == total_steps - 1:
                log.info(f"Step {step}/{total_steps} | Loss: {loss.item():.4f}")

            step += 1

        log.info("Quantum training complete!")
    else:
        trainer = Trainer(model, loss_fn, train_loader, cfg)
        trainer.fit(max_steps=args.steps)


if __name__ == "__main__":
    main()
