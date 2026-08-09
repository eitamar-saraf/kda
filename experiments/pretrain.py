"""Sec. 5.2: pretrain small language models across the hybrid ratio.

The paper's headline is that a 3:1 KDA-to-attention hybrid beats full attention at
matched FLOPs. That result comes from 1.4T tokens at 48B parameters. This is the same
comparison three orders of magnitude down: ~50M non-embedding parameters, ~1B tokens
of FineWeb-Edu, context 2048, on one RTX 3090 per run.

Being explicit about what that can and cannot show: at this scale the honest
expectation is that the configurations land close together, and any ordering needs to
survive the noise before it means anything. What the sweep *can* establish is the
shape -- whether adding a few global-attention layers to a linear stack buys what the
paper says it buys, and where pure linear attention starts to hurt.

Configurations mirror Table 1: 0:1 (all attention), 1:1, 3:1, 7:1, and pure KDA.

Usage::

    python -m experiments.pretrain --ratio 3 --out /mnt/data/kda/runs/pretrain
    python -m experiments.pretrain --sweep --out /mnt/data/kda/runs/pretrain
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from kda.model import LanguageModel, ModelConfig

#: Table 1's hybrid ratios. None = pure linear, 0 = pure full attention.
RATIOS = {"attn_only": 0, "1to1": 1, "3to1": 3, "7to1": 7, "kda_only": None}


@dataclass
class PretrainConfig:
    ratio_name: str = "3to1"
    variant: str = "kda"
    data_dir: str = "/mnt/ssd2/kda/data"
    seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    batch_size: int = 8
    grad_accum: int = 8               # effective batch 64 sequences = 131k tokens
    steps: int = 8000
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 40
    seed: int = 0
    compile: bool = False


class Batches:
    """Memory-mapped uint16 token stream; random windows, nanoGPT style."""

    def __init__(self, path: Path, seq_len: int, batch_size: int, device, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        if len(self.data) < seq_len + 1:
            raise RuntimeError(f"{path} has only {len(self.data)} tokens")

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, self.batch_size)
        x = np.stack([self.data[i : i + self.seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64) for i in ix])
        return (torch.from_numpy(x).to(self.device, non_blocking=True),
                torch.from_numpy(y).to(self.device, non_blocking=True))


def run(cfg: PretrainConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(cfg.data_dir)

    model_cfg = ModelConfig(
        vocab_size=50304,                    # GPT-2 vocab, padded to a multiple of 64
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        variant=cfg.variant,
        hybrid_ratio=RATIOS[cfg.ratio_name],
        rope=True,                           # see note below
        attn_conv=False,                     # a real LM baseline is a plain transformer
        max_timescale=2 * cfg.seq_len,
    )
    # NoPE is the paper's choice for the *hybrid*, where KDA layers carry position.
    # It is not a fair setting for the 0:1 all-attention baseline, which would then
    # have no positional information at all, so RoPE stays on everywhere and the
    # comparison is about the token mixer rather than about positional encoding.
    model = LanguageModel(model_cfg).to(dev)
    n_params = model.num_params()
    log(f"{cfg.ratio_name}: {n_params/1e6:.1f}M non-embedding params, plan={''.join('A' if p=='full' else 'K' for p in model.plan)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    train = Batches(data / "train.bin", cfg.seq_len, cfg.batch_size, dev, cfg.seed)
    val = Batches(data / "val.bin", cfg.seq_len, cfg.batch_size, dev, 1234)

    def lr_at(step):
        if step < cfg.warmup:
            return cfg.lr * step / max(1, cfg.warmup)
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        f = cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        return cfg.lr * f

    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def evaluate():
        model.eval()
        val.rng = np.random.default_rng(1234)      # same windows every time
        tot = 0.0
        for _ in range(cfg.eval_batches):
            x, y = val()
            with amp:
                loss, _ = model.loss(x, y)
            tot += loss.item()
        model.train()
        return tot / cfg.eval_batches

    history = []
    t0 = time.time()
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            vl = evaluate()
            history.append({"step": step, "val_loss": vl, "val_ppl": math.exp(min(20, vl)),
                            "tokens": step * tokens_per_step, "wall": time.time() - t0})
            log(f"  step {step:6d}  val loss {vl:.4f}  ppl {math.exp(min(20, vl)):7.2f}  "
                f"{step * tokens_per_step / 1e6:7.1f}M tokens  {time.time() - t0:6.0f}s")
        if step == cfg.steps:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = train()
            with amp:
                loss, _ = model.loss(x, y)
            (loss / cfg.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    return {
        "config": asdict(cfg),
        "hybrid_ratio": RATIOS[cfg.ratio_name],
        "layer_plan": model.plan,
        "num_params": n_params,
        "history": history,
        "final_val_loss": history[-1]["val_loss"],
        "final_val_ppl": history[-1]["val_ppl"],
        "wall_seconds": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--ratios", default="attn_only,1to1,3to1,7to1,kda_only")
    p.add_argument("--ratio", default="3to1")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--data-dir", default="/mnt/ssd2/kda/data")
    p.add_argument("--out", default="runs/pretrain")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in a.ratios.split(",")] if a.sweep else [a.ratio]

    for name in names:
        dest = out / f"{name}_s{a.seed}.json"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        cfg = PretrainConfig(ratio_name=name, steps=a.steps, data_dir=a.data_dir, seed=a.seed)
        res = run(cfg)
        dest.write_text(json.dumps(res, indent=2))
        print(f"-> {dest.name}  val ppl {res['final_val_ppl']:.2f}  {res['wall_seconds']:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
