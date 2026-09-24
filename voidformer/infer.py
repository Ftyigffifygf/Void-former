"""Inference Entry Point for VoidFormer models (Classical & Quantum).

Usage:
    python -m voidformer.infer --config voidformer/configs/tiny.yaml --model-type quantum --prompt "quantum entanglement" --use-quantum
"""

from __future__ import annotations

import argparse
import torch

from voidformer.datasets import build_tokenizer
from voidformer.models import VoidFormerModel, QuantumVoidFormer
from voidformer.utils import load_config, set_seed, get_logger


def main():
    parser = argparse.ArgumentParser(description="Inference with VoidFormer")
    parser.add_argument("--config", type=str, default="voidformer/configs/tiny.yaml", help="Path to config yaml")
    parser.add_argument("--model-type", type=str, choices=["classical", "quantum"], default="quantum", help="Model type")
    parser.add_argument("--prompt", type=str, default="quantum entanglement enables", help="Prompt text")
    parser.add_argument("--use-quantum", action="store_true", help="Enable quantum simulation processing during inference")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Number of new tokens to generate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))
    log = get_logger("voidformer.infer")

    tokenizer = build_tokenizer(cfg)
    vocab_size = tokenizer.vocab_size
    cfg["model"]["vocab_size"] = vocab_size

    if args.model_type == "quantum" or args.use_quantum:
        log.info("Loading QuantumVoidFormer model...")
        model = QuantumVoidFormer(
            vocab_size=vocab_size,
            d_model=cfg["model"]["d_model"],
            n_layers=cfg["model"]["n_layers"],
            n_heads=cfg["model"]["n_heads"],
            n_qubits_per_token=cfg["model"].get("n_qubits_per_token", 3),
            max_seq_len=cfg["model"]["max_seq_len"],
            collapse_protocol=cfg["model"].get("collapse_protocol", "entropy_gated"),
            enable_entanglement=True,
        )
    else:
        log.info("Loading classical VoidFormerModel...")
        model = VoidFormerModel(**cfg["model"])

    model.eval()

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    log.info(f"Prompt: '{args.prompt}'")

    generated_ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            curr_input = generated_ids[:, -model.max_seq_len:]
            if isinstance(model, QuantumVoidFormer):
                out = model(curr_input, return_diagnostics=True)
            else:
                out = model(curr_input)

            next_token_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    generated_text = tokenizer.decode(generated_ids[0].tolist())
    log.info(f"Generated text: '{generated_text}'")
    print("\n--- INFERENCE RESULT ---")
    print(generated_text)
    print("------------------------\n")


if __name__ == "__main__":
    main()
