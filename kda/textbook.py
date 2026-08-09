"""Literal transcriptions of Table 7 of the Kimi Linear paper (arXiv:2510.26692).

Each function here is a deliberately unoptimised, one-to-one translation of a single
row of that table: a Python loop that you can read side by side with the equation it
came from. Nothing is shared between them, on purpose -- shared helpers are exactly
what hides the differences we are trying to show.

``kda.recurrent.linear_attn`` is the *unified* kernel that subsumes all six. The test
suite asserts it reproduces every function below to floating-point tolerance, which is
what licenses the claim in the writeup that these six architectures differ only in two
switches: how the state decays, and whether it corrects itself before writing.

Shapes throughout::

    q, k : (B, H, T, Dk)      queries and keys
    v    : (B, H, T, Dv)      values
    S    : (B, H, Dk, Dv)     the recurrent state ("fast weights")
    o    : (B, H, T, Dv)      output

The state is read *after* the write (``o_t = S_t^T q_t``, Eq. 1), so a token can
retrieve what it just stored.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "linear_attention",
    "mamba2",
    "gla",
    "deltanet",
    "gated_deltanet",
    "kda",
    "softmax_attention",
]


def _empty_state(q: Tensor, v: Tensor) -> Tensor:
    b, h, _, dk = q.shape
    return q.new_zeros(b, h, dk, v.shape[-1])


def linear_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Katharopoulos et al. 2020.  ``S_t = S_{t-1} + k_t v_t^T``

    The state accumulates every key-value outer product and never forgets anything.
    Two tokens with non-orthogonal keys write on top of each other permanently: this
    is the interference that everything downstream is trying to fix.
    """
    S = _empty_state(q, v)
    out = []
    for t in range(q.shape[2]):
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]
        out.append((S * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def mamba2(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """Dao & Gu 2024.  ``S_t = a_t S_{t-1} + b_t k_t v_t^T``

    One scalar decay per head per step. The state now forgets, but uniformly: every
    channel of memory ages at exactly the same rate.

    alpha: (B, H, T) scalar decay in [0, 1];  beta: (B, H, T) write strength.
    """
    S = _empty_state(q, v)
    out = []
    for t in range(q.shape[2]):
        S = alpha[:, :, t, None, None] * S
        S = S + beta[:, :, t, None, None] * k[:, :, t, :, None] * v[:, :, t, None, :]
        out.append((S * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def gla(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor) -> Tensor:
    """Yang et al. 2024, Gated Linear Attention.  ``S_t = Diag(a_t) S_{t-1} + k_t v_t^T``

    The decay becomes a vector: one forgetting rate per key channel. This is the
    fine-grained gate KDA will keep -- but GLA has no delta rule, so it still writes
    blindly on top of whatever is already stored.

    alpha: (B, H, T, Dk) per-channel decay in [0, 1].
    """
    S = _empty_state(q, v)
    out = []
    for t in range(q.shape[2]):
        S = alpha[:, :, t, :, None] * S
        S = S + k[:, :, t, :, None] * v[:, :, t, None, :]
        out.append((S * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def deltanet(q: Tensor, k: Tensor, v: Tensor, beta: Tensor) -> Tensor:
    """Schlag et al. 2021 / Yang et al. 2024.

    ``S_t = (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T``

    The delta rule. Equivalently ``S_t = S_{t-1} + b_t k_t (v_t - S_{t-1}^T k_t)^T``:
    it writes the *prediction error*, not the value. That is one step of online
    gradient descent on ``L = 1/2 ||S^T k_t - v_t||^2`` with learning rate ``b_t``,
    so re-writing a key overwrites its old value instead of piling on top of it.

    beta: (B, H, T) learning rate in [0, 1]. No forgetting -- old keys live forever.
    """
    S = _empty_state(q, v)
    out = []
    for t in range(q.shape[2]):
        kt, vt = k[:, :, t], v[:, :, t]
        pred = (S * kt[..., None]).sum(dim=-2)          # S^T k_t, the current guess
        S = S + beta[:, :, t, None, None] * kt[..., None] * (vt - pred)[..., None, :]
        out.append((S * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def gated_deltanet(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """Yang et al. 2025, Gated DeltaNet.

    ``S_t = (I - b_t k_t k_t^T) a_t S_{t-1} + b_t k_t v_t^T``

    Delta rule *and* forgetting -- but the forget gate is a single scalar per head,
    so the whole state ages at one rate. This is the model KDA refines.

    alpha: (B, H, T) scalar decay;  beta: (B, H, T) learning rate.
    """
    S = _empty_state(q, v)
    out = []
    for t in range(q.shape[2]):
        kt, vt = k[:, :, t], v[:, :, t]
        S = alpha[:, :, t, None, None] * S              # decay first...
        pred = (S * kt[..., None]).sum(dim=-2)          # ...then correct the decayed state
        S = S + beta[:, :, t, None, None] * kt[..., None] * (vt - pred)[..., None, :]
        out.append((S * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def kda(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """Kimi Delta Attention, Eq. 1 of arXiv:2510.26692.

    ``S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T``

    GLA's per-channel gate meets DeltaNet's self-correcting write. The only change
    from Gated DeltaNet is that ``a_t`` is a vector rather than a scalar: each of the
    Dk memory channels gets its own forgetting rate, so the model can hold one fact
    while letting another decay. That single upgrade is what the paper's synthetic
    results and the per-channel decay figure in the writeup are about.

    alpha: (B, H, T, Dk) per-channel decay in [0, 1];  beta: (B, H, T) learning rate.
    """
    S = _empty_state(q, v)
    out = []
    for t in range(q.shape[2]):
        kt, vt = k[:, :, t], v[:, :, t]
        S = alpha[:, :, t, :, None] * S                 # Diag(a_t) S_{t-1}
        pred = (S * kt[..., None]).sum(dim=-2)          # what the decayed state predicts
        S = S + beta[:, :, t, None, None] * kt[..., None] * (vt - pred)[..., None, :]
        out.append((S * q[:, :, t, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def softmax_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal softmax attention, for reference.

    No state at all: every past key and value is kept verbatim and re-scanned. That
    is why it never interferes with itself -- and why its memory grows with T.
    """
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    t = q.shape[2]
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v
