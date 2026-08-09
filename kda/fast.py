"""Production Triton kernels for each family member, via flash-linear-attention.

Why this file exists. ``kda/chunk.py`` is written to be *read*: it follows Eq. 2-9
step by step and computes the decay-weighted Gram matrices with an explicit loop so
that nothing overflows and nothing is hidden. That costs about 25x a fused kernel,
which is fine for tests and figures and hopeless for a training sweep.

So the repo keeps both, with a hard link between them: :mod:`tests.test_fast_backend`
asserts that for every variant the fla kernel computes what our reference computes.
The readable code is what the writeup explains; the fast code is what produces the
numbers; and the tests are what make it legitimate to say those are the same thing.

Layout adapters live here too -- fla uses ``[B, T, H, D]`` and log-space gates, we use
``[B, H, T, D]``.

The mapping (see the table in :mod:`kda.recurrent`):

    linear_attention  chunk_linear_attn(normalize=False)
    mamba2            chunk_simple_gla        beta folded into v
    gla               chunk_gla
    deltanet          chunk_delta_rule
    gated_deltanet    chunk_gated_delta_rule
    kda               chunk_kda               <- Eq. 1
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["fla_available", "fast_linear_attn", "FLA_KERNELS"]


def fla_available() -> bool:
    try:
        import fla.ops  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


FLA_KERNELS = {
    "linear_attention": "chunk_linear_attn",
    "mamba2": "chunk_simple_gla",
    "gla": "chunk_gla",
    "deltanet": "chunk_delta_rule",
    "gated_deltanet": "chunk_gated_delta_rule",
    "kda": "chunk_kda",
}


def _t(x: Tensor) -> Tensor:
    """(B, H, T, D) <-> (B, T, H, D)."""
    return x.transpose(1, 2).contiguous()


def fast_linear_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor | None = None,
    log_alpha: Tensor | None = None,
    *,
    gate: str = "channel",
    delta: bool = True,
    scale: float = 1.0,
    initial_state: Tensor | None = None,
    return_state: bool = False,
):
    """Dispatch to the right fla kernel for this (gate, delta) combination.

    Args and returns match :func:`kda.chunk.chunk_linear_attn`, so the two are
    drop-in swappable and the layer can choose a backend at construction time.
    """
    import fla.ops as ops

    b, h, t, dk = q.shape

    # The Triton kernels require their operands to share a dtype. Under autocast q/k/v
    # arrive as bf16 while the decay gate is computed in fp32 (it is a cumulative
    # product, so the layer builds it in wide arithmetic on purpose). In KDA the decay
    # feeds the A matrix directly, so a mixed pair reaches tl.dot and the kernel
    # aborts. Harmonise here rather than forcing the layer to give up gate precision:
    # fla upcasts g to fp32 internally for the cumulative sum regardless.
    dt = v.dtype
    cast = lambda x: None if x is None else x.to(dt)
    q, k, v = cast(q), cast(k), cast(v)
    beta, log_alpha = cast(beta), cast(log_alpha)
    if initial_state is not None:
        initial_state = initial_state.to(torch.float32)

    kw = dict(scale=scale, initial_state=initial_state, output_final_state=True)

    if delta and gate == "channel":
        o, s = ops.chunk_kda(q=_t(q), k=_t(k), v=_t(v), g=_t(log_alpha),
                             beta=_t(beta.unsqueeze(-1)).squeeze(-1), **kw)
    elif delta and gate == "scalar":
        o, s = ops.chunk_gated_delta_rule(q=_t(q), k=_t(k), v=_t(v),
                                          g=_t(log_alpha.unsqueeze(-1)).squeeze(-1),
                                          beta=_t(beta.unsqueeze(-1)).squeeze(-1), **kw)
    elif delta and gate == "none":
        o, s = ops.chunk_delta_rule(q=_t(q), k=_t(k), v=_t(v),
                                    beta=_t(beta.unsqueeze(-1)).squeeze(-1), **kw)
    elif not delta and gate == "channel":
        # GLA writes k v^T with no coefficient; fold beta into v to stay general.
        o, s = ops.chunk_gla(q=_t(q), k=_t(k), v=_t(v * beta[..., None]),
                             g=_t(log_alpha), **kw)
    elif not delta and gate == "scalar":
        o, s = ops.chunk_simple_gla(q=_t(q), k=_t(k), v=_t(v * beta[..., None]),
                                    g=_t(log_alpha.unsqueeze(-1)).squeeze(-1), **kw)
    else:  # not delta, gate == "none"
        o, s = ops.chunk_linear_attn(q=_t(q), k=_t(k), v=_t(v * beta[..., None]),
                                     scale=scale, initial_state=initial_state,
                                     output_final_state=True, normalize=False)

    o = o.transpose(1, 2)
    return (o, s) if return_state else o
