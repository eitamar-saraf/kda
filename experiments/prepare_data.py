"""Tokenise a slice of FineWeb-Edu into flat uint16 shards.

nanoGPT's format: one contiguous array of token ids per split, memory-mapped at train
time. At GPT-2's 50257-token vocabulary the ids fit in uint16, so a billion tokens is
2 GB on disk and the loader is a slice.

Streaming means we never materialise the full dataset -- we pull until the token budget
is met and stop, which matters because the full sample-10BT is ~20 GB of text.

Usage::

    python -m experiments.prepare_data --tokens 1_000_000_000 --out /mnt/ssd2/kda/data
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def build(out_dir: Path, n_tokens: int, val_tokens: int, dataset: str, name: str | None):
    import tiktoken
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    stream = load_dataset(dataset, name=name, split="train", streaming=True)

    total = n_tokens + val_tokens
    buf = np.empty(total, dtype=np.uint16)
    n = 0
    t0 = time.time()
    for i, row in enumerate(stream):
        ids = enc.encode_ordinary(row["text"])
        ids.append(eot)                       # document boundary
        take = min(len(ids), total - n)
        buf[n : n + take] = np.asarray(ids[:take], dtype=np.uint16)
        n += take
        if n >= total:
            break
        if i % 20000 == 0 and i:
            rate = n / (time.time() - t0)
            print(f"  {n / 1e6:8.1f}M / {total / 1e6:.0f}M tokens "
                  f"({rate / 1e6:.2f}M tok/s, {(total - n) / rate / 60:.0f} min left)",
                  flush=True)

    if n < total:
        raise RuntimeError(f"stream exhausted at {n} tokens, wanted {total}")

    # Validation is taken from the tail, so it is documents the model never trained on.
    val = buf[n_tokens:]
    train = buf[:n_tokens]
    for split, arr in (("train", train), ("val", val)):
        path = out_dir / f"{split}.bin"
        arr.tofile(path)
        print(f"wrote {path}  {arr.size:,} tokens  {path.stat().st_size / 2**30:.2f} GiB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/mnt/ssd2/kda/data")
    p.add_argument("--tokens", type=int, default=1_000_000_000)
    p.add_argument("--val-tokens", type=int, default=5_000_000)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT")
    a = p.parse_args()
    build(Path(a.out), a.tokens, a.val_tokens, a.dataset, a.name)


if __name__ == "__main__":
    main()
