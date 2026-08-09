"""Sec. 5.1: train every architecture on palindrome, MQAR and stack tracking.

Reproduces the shape of Figure 4 -- peak accuracy as sequence length grows, and how
fast each model converges at a fixed length. All variants share the layer body from
:mod:`kda.layers`, so any separation in the curves is attributable to the memory
mechanism rather than to incidental differences in parameterisation.

Usage::

    python -m experiments.synthetic --task mqar --variant kda --seq-len 1024
    python -m experiments.synthetic --sweep --out /mnt/data/kda/runs/synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kda.model import LanguageModel, ModelConfig
from kda.tasks import mqar, palindrome, stack

#: The six family members plus a full-attention reference.
VARIANTS = [
    "kda", "gated_deltanet", "gla", "deltanet", "mamba2", "linear_attention", "softmax",
]


def build_task(task: str, seq_len: int):
    """Task configs that scale sensibly with sequence length."""
    if task == "mqar":
        n_pairs = max(4, seq_len // 32)
        cfg = mqar.MQARConfig(
            seq_len=seq_len,
            num_pairs=n_pairs,
            num_queries=max(2, n_pairs // 2),
            num_keys=max(32, 2 * n_pairs),
            num_values=32,
            num_noise=16,
        )
        return cfg, mqar.make_batch
    if task == "palindrome":
        return palindrome.PalindromeConfig(seq_len=seq_len, num_symbols=32), palindrome.make_batch
    if task == "stack":
        return stack.StackConfig(seq_len=seq_len, num_stacks=64, num_symbols=32), stack.make_batch
    raise KeyError(task)


@dataclass
class RunConfig:
    task: str = "mqar"
    variant: str = "kda"
    seq_len: int = 1024
    steps: int = 6000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.1
    warmup: int = 200
    d_model: int = 256
    n_heads: int = 2
    n_layers: int = 2
    seed: int = 0
    eval_every: int = 100
    eval_batches: int = 4
    device: str = "cuda"
    amp: bool = True


def run(cfg: RunConfig, log=print) -> dict:
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    task_cfg, make_batch = build_task(cfg.task, cfg.seq_len)

    is_softmax = cfg.variant == "softmax"
    model_cfg = ModelConfig(
        vocab_size=task_cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        variant="kda" if is_softmax else cfg.variant,
        hybrid_ratio=0 if is_softmax else None,
        # Attention gets RoPE and the same short convolution the linear layers carry.
        # Without the convolution it stalls at ~33% on MQAR while KDA solves it -- not
        # because attention is worse at recall, but because a previous-token operation
        # is handed to the linear models for free and attention has to learn it. The
        # measured difference: rope+conv 1.00, rope only 0.32, conv only 0.92, neither
        # 0.34. A baseline crippled that way would make every result here meaningless.
        rope=True,
        attn_conv=True,
        # Initialise the gated variants to be able to hold dependencies across the
        # whole sequence. Without this the decay init alone decides the result:
        # see kda.layers.timescale_dt_range.
        max_timescale=2 * cfg.seq_len,
    )
    model = LanguageModel(model_cfg).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < cfg.warmup:
            return step / max(1, cfg.warmup)
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    train_gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen_seed = cfg.seed + 10_000
    use_amp = cfg.amp and dev.type == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False)

    @torch.no_grad()
    def evaluate():
        """Fresh sequences from a fixed seed, so every checkpoint sees the same eval."""
        model.eval()
        g = torch.Generator().manual_seed(eval_gen_seed)
        accs, losses = [], []
        for _ in range(cfg.eval_batches):
            tok, tgt, mask = make_batch(task_cfg, cfg.batch_size, g)
            tok, tgt, mask = tok.to(dev), tgt.to(dev), mask.to(dev)
            with amp_ctx:
                loss, acc = model.loss(tok, tgt, mask)
            losses.append(loss.item())
            accs.append(acc.item())
        model.train()
        return sum(losses) / len(losses), sum(accs) / len(accs)

    history = []
    t0 = time.time()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            loss_e, acc_e = evaluate()
            history.append({"step": step, "loss": loss_e, "acc": acc_e,
                            "wall": time.time() - t0})
            log(f"  [{cfg.task}/{cfg.variant}/T={cfg.seq_len}/lr={cfg.lr:g}] "
                f"step {step:6d}  loss {loss_e:.4f}  acc {acc_e:.4f}")
            if acc_e > 0.999 and step > 0:
                log("  solved; stopping early")
                break
        if step == cfg.steps:
            break

        for gparam in opt.param_groups:
            gparam["lr"] = cfg.lr * lr_at(step)

        tok, tgt, mask = make_batch(task_cfg, cfg.batch_size, train_gen)
        tok, tgt, mask = tok.to(dev), tgt.to(dev), mask.to(dev)
        with amp_ctx:
            loss, _ = model.loss(tok, tgt, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    best = max(h["acc"] for h in history)
    return {
        "config": asdict(cfg),
        "vocab_size": task_cfg.vocab_size,
        "num_params": model.num_params(),
        "gate_params": sum(
            b.mixer.gate_params() for b in model.blocks if hasattr(b.mixer, "gate_params")
        ),
        "history": history,
        "best_acc": best,
        "final_acc": history[-1]["acc"],
        "steps_to_90": next((h["step"] for h in history if h["acc"] >= 0.9), None),
        "wall_seconds": time.time() - t0,
    }


def sweep(tasks, out_dir: Path, seq_lens=(256, 512, 1024, 2048),
          lr=1e-3, steps=6000, seeds=(0, 1, 2), device="cuda"):
    """The Figure 4 grid: every variant, every length, several seeds.

    Seeds are not optional here. These tasks are solved through a sharp phase
    transition rather than a smooth climb, so a single run reports whether that
    transition happened to fire inside the step budget, which is close to a coin flip
    near the length where a model starts to struggle. A pilot at 2500 steps had KDA at
    0.97 and Gated DeltaNet at 0.04 for T=1024, then the order reversed at T=2048 --
    noise that would have read as a clean architectural finding from one seed each.

    The learning rate is fixed at 1e-3 (chosen in a pilot over {1e-3, 5e-4}) and the
    budget spent on seeds instead, because knowing the spread matters more here than
    squeezing the last point out of a per-variant LR.

    Each run writes its own JSON, so the sweep is resumable and a crash costs one run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def do(task, variant, seq_len, seed):
        p = out_dir / task / f"{variant}_T{seq_len}_lr{lr:g}_s{seed}.json"
        if p.exists():
            return json.loads(p.read_text())
        cfg = RunConfig(task=task, variant=variant, seq_len=seq_len, lr=lr,
                        steps=steps, seed=seed, device=device)
        res = run(cfg, log=lambda *a: None)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
        return res

    for task in tasks:
        for seq_len in seq_lens:
            for variant in VARIANTS:
                accs = []
                for seed in seeds:
                    res = do(task, variant, seq_len, seed)
                    accs.append(res["best_acc"])
                mean = sum(accs) / len(accs)
                print(f"  [{task}] T={seq_len:5d} {variant:17s} "
                      f"mean={mean:.3f}  seeds={[f'{a:.2f}' for a in accs]}", flush=True)
    print("sweep complete")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--tasks", default="mqar,palindrome,stack")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--task", default="mqar", choices=["mqar", "palindrome", "stack"])
    p.add_argument("--variant", default="kda", choices=VARIANTS)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.sweep:
        sweep(
            [t.strip() for t in a.tasks.split(",") if t.strip()],
            Path(a.out or "runs/synthetic"),
            steps=a.steps, device=a.device,
            seeds=tuple(int(x) for x in a.seeds.split(",")),
        )
        return

    cfg = RunConfig(
        task=a.task, variant=a.variant, seq_len=a.seq_len, steps=a.steps,
        batch_size=a.batch_size, lr=a.lr, seed=a.seed, device=a.device,
    )
    result = run(cfg)
    print(f"best acc {result['best_acc']:.4f}  params {result['num_params']:,}  "
          f"{result['wall_seconds']:.0f}s")
    if a.out:
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
