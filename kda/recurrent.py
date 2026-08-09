"""One recurrence, six architectures.

Every model in ``kda.textbook`` is the same update with two switches flipped:

    S_t = (I - b_t k_t k_t^T)^[delta] . G_t . S_{t-1} + b_t k_t v_t^T
    o_t = S_t^T q_t

where ``G_t`` is what the state gets multiplied by before the write:

    gate="none"      G_t = I                 no forgetting
    gate="scalar"    G_t = a_t I             one decay rate for the whole head
    gate="channel"   G_t = Diag(a_t)         one decay rate per memory channel

and ``delta`` decides whether the write is corrective:

    delta=False      S_t = G_t S_{t-1} + b_t k_t v_t^T
    delta=True       S_t = G_t S_{t-1} + b_t k_t (v_t - (G_t S_{t-1})^T k_t)^T

The delta form is worth staring at. ``(G_t S_{t-1})^T k_t`` is what the memory would
answer if you queried it with ``k_t`` right now; the write is the *error* between that
and the value you meant to store, scaled by ``b_t``. So ``b_t`` is a learning rate and
the whole update is one SGD step on ``L_t(S) = 1/2 ||S^T k_t - v_t||^2``. Re-writing a
key corrects it; in the non-delta form it just piles on.

Filling in the grid:

    gate       delta   model            b_t
    ---------  ------  ---------------  --------------------
    none       False   Linear Attention fixed to 1
    scalar     False   Mamba2           learned
    channel    False   GLA              fixed to 1
    none       True    DeltaNet         learned
    scalar     True    Gated DeltaNet   learned
    channel    True    KDA              learned      <- Eq. 1

Linear Attention and GLA write ``+ k_t v_t^T`` with no coefficient, so they are the
``b_t = 1`` corner of this family rather than a separate rule; ``VARIANTS`` records
that with ``fixed_beta``.

This module is the readable O(T)-loop reference. ``kda.chunk`` implements the same
maths in the chunkwise-parallel form the GPU actually wants; the two are asserted
equal in ``tests/``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = ["linear_attn", "VARIANTS", "variant"]

Gate = Literal["none", "scalar", "channel"]


def linear_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor | None = None,
    beta: Tensor | None = None,
    *,
    gate: Gate = "channel",
    delta: bool = True,
    log_alpha: Tensor | None = None,
    scale: float = 1.0,
    initial_state: Tensor | None = None,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sequential reference implementation of the gated (delta) linear attention family.

    Args:
        q, k: ``(B, H, T, Dk)``.
        v: ``(B, H, T, Dv)``.
        alpha: decay. ``(B, H, T)`` when ``gate="scalar"``, ``(B, H, T, Dk)`` when
            ``gate="channel"``, ignored when ``gate="none"``. Values in ``[0, 1]``.
        beta: ``(B, H, T)`` write strength / learning rate. Defaults to all ones.
        gate: granularity of the forget gate -- the axis KDA moves along.
        delta: whether the write corrects the existing memory (the delta rule).
        log_alpha: the decay in log space, same shape as ``alpha``. Trained models
            produce this directly (``g = -exp(A_log) * softplus(...)``, always
            negative), so taking it here avoids an exp/log round trip. Overrides
            ``alpha`` when given.
        scale: multiplier on the queries. The recurrence itself needs none; the
            neural layer uses ``1/sqrt(Dk)``, matching the reference kernel.
        initial_state: ``(B, H, Dk, Dv)`` carried in from a previous chunk.
        return_state: also return the final ``(B, H, Dk, Dv)`` state.

    Returns:
        ``(B, H, T, Dv)`` outputs, and the final state if ``return_state``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]

    if log_alpha is not None:
        alpha = log_alpha.exp()
    if scale != 1.0:
        q = q * scale

    if gate == "none":
        if alpha is not None:
            raise ValueError("gate='none' takes no alpha")
    elif gate == "scalar":
        if alpha is None or alpha.shape != (b, h, t):
            raise ValueError(f"gate='scalar' needs alpha of shape {(b, h, t)}, got {None if alpha is None else tuple(alpha.shape)}")
    elif gate == "channel":
        if alpha is None or alpha.shape != (b, h, t, dk):
            raise ValueError(f"gate='channel' needs alpha of shape {(b, h, t, dk)}, got {None if alpha is None else tuple(alpha.shape)}")
    else:
        raise ValueError(f"unknown gate {gate!r}")

    if beta is None:
        beta = q.new_ones(b, h, t)
    elif beta.shape != (b, h, t):
        raise ValueError(f"beta must have shape {(b, h, t)}, got {tuple(beta.shape)}")

    S = q.new_zeros(b, h, dk, dv) if initial_state is None else initial_state
    out = []
    for i in range(t):
        # 1. decay the memory: G_t S_{t-1}
        if gate == "scalar":
            S = alpha[:, :, i, None, None] * S
        elif gate == "channel":
            S = alpha[:, :, i, :, None] * S

        # 2. decide what to write at key k_t
        ki, vi = k[:, :, i], v[:, :, i]
        if delta:
            # the error between what the memory already answers and what we want
            target = vi - (S * ki[..., None]).sum(dim=-2)
        else:
            target = vi

        # 3. write it
        S = S + beta[:, :, i, None, None] * ki[..., None] * target[..., None, :]

        # 4. read with the *updated* state (Eq. 1: o_t = S_t^T q_t)
        out.append((S * q[:, :, i, :, None]).sum(dim=-2))

    o = torch.stack(out, dim=2)
    return (o, S) if return_state else o


#: How each named architecture instantiates :func:`linear_attn`.
#: ``fixed_beta`` records models whose write has no learned coefficient.
VARIANTS: dict[str, dict] = {
    "linear_attention": {"gate": "none", "delta": False, "fixed_beta": 1.0},
    "mamba2": {"gate": "scalar", "delta": False, "fixed_beta": None},
    "gla": {"gate": "channel", "delta": False, "fixed_beta": 1.0},
    "deltanet": {"gate": "none", "delta": True, "fixed_beta": None},
    "gated_deltanet": {"gate": "scalar", "delta": True, "fixed_beta": None},
    "kda": {"gate": "channel", "delta": True, "fixed_beta": None},
}


def variant(name: str, q: Tensor, k: Tensor, v: Tensor, alpha=None, beta=None, **kw):
    """Run a named architecture from :data:`VARIANTS` through :func:`linear_attn`."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; have {sorted(VARIANTS)}")
    spec = dict(VARIANTS[name])
    fixed = spec.pop("fixed_beta")
    if fixed is not None:
        beta = q.new_full(q.shape[:3], fixed)
    if spec["gate"] == "none":
        alpha = None
    return linear_attn(q, k, v, alpha, beta, **spec, **kw)
