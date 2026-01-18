# logp_credit/data/gsm8k.py
from __future__ import annotations

from datasets import load_dataset


def load_gsm8k(split: str):
    for name, cfg in [("gsm8k", "main"), ("openai/gsm8k", "main")]:
        try:
            return load_dataset(name, cfg, split=split)
        except Exception:
            pass
    raise RuntimeError("Failed to load GSM8K via datasets. Try: pip install -U datasets")
