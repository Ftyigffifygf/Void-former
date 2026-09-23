"""Inference entry point for VoidFormer & Quantum VoidFormer.

Usage:
    python -m voidformer.infer --config voidformer/configs/tiny.yaml --model-type quantum --prompt "quantum entanglement enables" --use-quantum
"""

from __future__ import annotations

import argparse

import torch

from voidformer.datasets import build_tokenizer
from voidformer.models import VoidFormerModel
from voidformer.models.quantum_voidformer import QuantumVoidFormer
from voidformer.utils import load_config, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with VoidFormer model")
    parser.add_argument("--config", default="voidformer/configs/tiny.yaml", help="Path to config YAML file")
    parser.add_argument("--model-type", choices=["quantum", "classical"], default="quantum", help="Model type")
    parser.add_argument("--prompt", default="quantum computing is", help="Generation prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--use-quantum", action="store_true", default=True, help="Use quantum processing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger("voidformer.infer")

    tok = build_tokenizer(cfg)
    cfg["model"]["vocab_size"] = tok.vocab_size

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

    model.eval()

    prompt_ids = tok.encode(args.prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    log.info(f"Generating from prompt: '{args.prompt}'...")
    if args.model_type == "quantum":
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_quantum=args.use_quantum,
        )
    else:
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    generated_text = tok.decode(output_ids[0].tolist())
    print("\n" + "=" * 50)
    print("GENERATED TEXT:")
    print("=" * 50)
    print(generated_text)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
