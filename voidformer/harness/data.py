"""Data Loader Utilities for Harness Package."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets.toy_corpus import ToyCorpus
    from datasets.tokenizers import CharTokenizer, build_tokenizer
except ImportError:
    from voidformer.datasets.toy_corpus import ToyCorpus
    from voidformer.datasets.tokenizers import CharTokenizer, build_tokenizer


class SyntheticDataset(Dataset):
    """Synthetic dataset generator for benchmark reproducibility."""
    def __init__(self, vocab_size: int = 128, seq_len: int = 64, num_samples: int = 100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.data[idx]
        y = torch.roll(x, -1, dims=0)
        return {"input_ids": x, "labels": y}


def create_dataloader(
    vocab_size: int = 128,
    seq_len: int = 64,
    batch_size: int = 8,
    num_samples: int = 100,
) -> DataLoader:
    """Create PyTorch DataLoader for synthetic corpus data."""
    dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
