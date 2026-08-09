"""Sec. 3.2 / 6.2: what the specialised DPLR variant actually buys, measured.

Three things get timed here.

1. **Kernel throughput (Figure 2).** KDA's chunk kernel against the *general* DPLR
   kernel computing the same recurrence. The paper's claim is roughly 2x, and the
   reason is structural: a general DPLR transition ``(D - a b^T)`` carries two free
   vectors, so the chunkwise derivation needs four second-level chunk matmuls and a
   ``1/Gamma`` reciprocal that forces secondary chunking in fp32. Binding ``a = b = k``
   -- which is what makes it a *delta* rule rather than an arbitrary low-rank update --
   collapses those to two and removes the reciprocal.

2. **Prefill (Figure 7a).** Whole-model forward at increasing context, for pure
   attention, the 3:1 hybrid, and pure KDA. This is where quadratic attention stops
   being affordable.

3. **Decode and memory (Figure 7b).** Per-token latency and, more importantly, the
   bytes each architecture must keep. Attention stores a KV cache that grows with
   context; KDA stores a ``Dk x Dv`` state that does not. That is the 75% number.

All numbers come from the GPU this is run on -- an RTX 3090 in our case, not the
paper's H800 -- so absolute values differ from the paper. The ratios are the point.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kda.model import LanguageModel, ModelConfig


def cuda_time(fn, warmup=3, iters=10) -> float:
    """Milliseconds per call, with the GPU actually synchronised."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def make_kda_inputs(b, h, t, dk, dv, device="cuda", dtype=torch.bfloat16):
    q = torch.randn(b, t, h, dk, device=device, dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn(b, t, h, dk, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    v = torch.randn(b, t, h, dv, device=device, dtype=dtype)
    g = (-torch.nn.functional.softplus(
        torch.randn(b, t, h, dk, device=device, dtype=torch.float32)
    ) * 0.1).to(dtype)                       # log alpha, in (-inf, 0]
    beta = torch.rand(b, t, h, device=device, dtype=dtype)
    return q, k, v, g, beta


def dplr_equivalent(q, k, v, g, beta):
    """Express KDA in the general DPLR form, so the two kernels do the same work.

    KDA:  S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T
              = (Diag(a_t) - b_t k_t (a_t * k_t)^T) S_{t-1} + b_t k_t v_t^T

    fla's DPLR kernel computes ``S_t = (Diag(exp(gk)) + a b^T) S_{t-1} + k v^T``, so
    ``a = -beta * k``, ``b = alpha * k``, and beta folds into v since the DPLR write
    carries no coefficient of its own.
    """
    a = -(beta[..., None] * k)
    b = g.float().exp().to(k.dtype) * k
    return q, k, v * beta[..., None], a, b, g


def bench_kernels(lengths, b=1, h=16, dk=128, dv=128, backward=False):
    """Figure 2: our chunk kernel vs general DPLR vs flash attention."""
    import fla.ops as ops
    from kda.chunk import chunk_linear_attn

    rows = []
    for t in lengths:
        q, k, v, g, beta = make_kda_inputs(b, h, t, dk, dv)
        scale = dk ** -0.5
        entry = {"seq_len": t}

        def timed(label, fn):
            try:
                entry[label] = cuda_time(fn)
            except torch.cuda.OutOfMemoryError:
                entry[label] = None
                torch.cuda.empty_cache()
            except Exception as e:                      # kernel may not support a shape
                entry[label] = None
                entry[label + "_error"] = str(e)[:120]
                torch.cuda.empty_cache()

        timed("kda", lambda: ops.chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale))

        qd, kd, vd, ad, bd, gd = dplr_equivalent(q, k, v, g, beta)
        timed("dplr", lambda: ops.chunk_dplr_delta_rule(
            q=qd, k=kd, v=vd, a=ad, b=bd, gk=gd, scale=scale))

        # flash attention, same shapes, [B, H, T, D]
        qa, ka, va = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        timed("flash_attn", lambda: torch.nn.functional.scaled_dot_product_attention(
            qa, ka, va, is_causal=True))

        # our readable reference, only where it is affordable
        if t <= 8192:
            timed("ours_chunk", lambda: chunk_linear_attn(
                qa, ka, va, beta=beta.transpose(1, 2), log_alpha=g.transpose(1, 2),
                gate="channel", delta=True, scale=scale, chunk_size=64))

        rows.append(entry)
        parts = " ".join(
            f"{key}={entry[key]:8.2f}ms" for key in ("kda", "dplr", "flash_attn", "ours_chunk")
            if entry.get(key) is not None
        )
        speedup = (entry["dplr"] / entry["kda"]) if entry.get("dplr") and entry.get("kda") else float("nan")
        print(f"  T={t:6d}  {parts}   dplr/kda={speedup:.2f}x")
    return rows


def bench_model(lengths, d_model=1024, n_layers=8, n_heads=8, vocab=32000):
    """Figure 7a plus the memory story: prefill cost and what must be cached."""
    rows = []
    configs = {
        "full_attention": dict(hybrid_ratio=0),
        "hybrid_3to1": dict(hybrid_ratio=3),
        "pure_kda": dict(hybrid_ratio=None),
    }
    for t in lengths:
        entry = {"seq_len": t}
        for name, kw in configs.items():
            cfg = ModelConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                              n_heads=n_heads, variant="kda", **kw)
            model = LanguageModel(cfg).cuda().eval()
            tok = torch.randint(0, vocab, (1, t), device="cuda")
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    entry[name] = cuda_time(lambda: model(tok), warmup=2, iters=5)
            except torch.cuda.OutOfMemoryError:
                entry[name] = None
            torch.cuda.empty_cache()

            # what each layer type must retain to decode token t+1
            n_full = model.plan.count("full")
            n_lin = model.plan.count("linear")
            head_dim = d_model // n_heads
            kv_bytes = n_full * 2 * t * d_model * 2                      # K and V, bf16
            state_bytes = n_lin * n_heads * head_dim * head_dim * 2      # Dk x Dv per head
            entry[name + "_cache_mb"] = (kv_bytes + state_bytes) / 2**20
            del model
        rows.append(entry)
        print(f"  T={t:7d}  " + "  ".join(
            f"{n}={entry[n]:7.1f}ms/{entry[n + '_cache_mb']:7.1f}MB"
            for n in configs if entry.get(n) is not None))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--kernel-lengths", default="2048,4096,8192,16384,32768,65536")
    p.add_argument("--model-lengths", default="1024,4096,16384,65536,262144")
    p.add_argument("--skip-model", action="store_true")
    a = p.parse_args()

    torch.manual_seed(0)
    results = {"device": torch.cuda.get_device_name(0)}

    print("== kernel throughput (Figure 2) ==")
    results["kernels"] = bench_kernels([int(x) for x in a.kernel_lengths.split(",")])

    if not a.skip_model:
        print("== prefill and cache (Figure 7) ==")
        results["model"] = bench_model([int(x) for x in a.model_lengths.split(",")])

    if a.out:
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
