"""Third validation leg: agree with the official KDA kernel.

Phase 0 showed our unified recurrence reproduces six textbook equations. Phase 1
showed the chunkwise rearrangement reproduces the recurrence. Neither rules out
having misread the paper in the same way twice. This file closes that loop against
``fla.ops.kda`` -- the kernel released by the Kimi team alongside the paper.

Requires a CUDA GPU and ``flash-linear-attention``; skipped otherwise, so the suite
still runs on a laptop.

Layout note: fla uses ``[B, T, H, K]`` and log-space gates; we use ``[B, H, T, K]``.
:func:`to_fla` is the only adapter.
"""

from __future__ import annotations

import pytest
import torch

from kda.chunk import chunk_linear_attn
from kda.recurrent import linear_attn

fla_kda = pytest.importorskip("fla.ops.kda", reason="flash-linear-attention not installed")
naive_mod = pytest.importorskip("fla.ops.kda.naive")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

B, H, T, DK, DV = 2, 4, 256, 64, 64


def to_fla(x):
    """(B, H, T, D) -> (B, T, H, D)."""
    return x.transpose(1, 2).contiguous()


def from_fla(x):
    """(B, T, H, D) -> (B, H, T, D)."""
    return x.transpose(1, 2).contiguous()


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(B, H, T, DK, device=dev, dtype=torch.float32)
    k = torch.nn.functional.normalize(
        torch.randn(B, H, T, DK, device=dev, dtype=torch.float32), dim=-1
    )
    v = torch.randn(B, H, T, DV, device=dev, dtype=torch.float32)
    # log-space decay, the form trained models actually produce:
    # g = -exp(A_log) * softplus(...)  -- always negative, so alpha = exp(g) in (0, 1)
    log_alpha = -torch.nn.functional.softplus(
        torch.randn(B, H, T, DK, device=dev, dtype=torch.float32)
    ) * 0.5
    beta = torch.rand(B, H, T, device=dev, dtype=torch.float32)
    return q, k, v, log_alpha, beta


def test_matches_fla_naive_recurrent(inputs):
    """Our O(T) loop against theirs, same scale convention."""
    q, k, v, log_alpha, beta = inputs
    scale = DK ** -0.5

    ref, _ = naive_mod.naive_recurrent_kda(
        to_fla(q), to_fla(k), to_fla(v), to_fla(log_alpha),
        beta.transpose(1, 2).contiguous(), scale=scale,
    )
    ours = linear_attn(
        q, k, v, beta=beta, log_alpha=log_alpha,
        gate="channel", delta=True, scale=scale,
    )
    torch.testing.assert_close(ours, from_fla(ref), rtol=2e-4, atol=2e-4)


def test_chunkwise_matches_fla_chunk_kernel(inputs):
    """Our Eq. 2-9 implementation against the released Triton chunk kernel.

    This is the one that matters: it says the algorithm in ``kda/chunk.py``, which the
    writeup walks through line by line, computes what the Kimi team's kernel computes.
    """
    q, k, v, log_alpha, beta = inputs
    scale = DK ** -0.5

    ref, _ = fla_kda.chunk_kda(
        q=to_fla(q), k=to_fla(k), v=to_fla(v), g=to_fla(log_alpha),
        beta=beta.transpose(1, 2).contiguous(), scale=scale,
    )
    ours = chunk_linear_attn(
        q, k, v, beta=beta, log_alpha=log_alpha,
        gate="channel", delta=True, scale=scale, chunk_size=64,
    )
    torch.testing.assert_close(ours, from_fla(ref), rtol=2e-3, atol=2e-3)


def test_chunkwise_matches_fla_fused_recurrent(inputs):
    """And against their decode-path kernel, which is what inference actually runs."""
    q, k, v, log_alpha, beta = inputs
    scale = DK ** -0.5

    ref, _ = fla_kda.fused_recurrent_kda(
        q=to_fla(q), k=to_fla(k), v=to_fla(v), g=to_fla(log_alpha),
        beta=beta.transpose(1, 2).contiguous(), scale=scale,
    )
    ours = chunk_linear_attn(
        q, k, v, beta=beta, log_alpha=log_alpha,
        gate="channel", delta=True, scale=scale, chunk_size=64,
    )
    torch.testing.assert_close(ours, from_fla(ref), rtol=2e-3, atol=2e-3)


def test_final_state_matches_fla(inputs):
    """The carried state too -- this is what a KV-cache-free decode depends on."""
    q, k, v, log_alpha, beta = inputs
    scale = DK ** -0.5

    _, ref_state = fla_kda.chunk_kda(
        q=to_fla(q), k=to_fla(k), v=to_fla(v), g=to_fla(log_alpha),
        beta=beta.transpose(1, 2).contiguous(), scale=scale,
        output_final_state=True,
    )
    _, ours_state = chunk_linear_attn(
        q, k, v, beta=beta, log_alpha=log_alpha,
        gate="channel", delta=True, scale=scale, chunk_size=64, return_state=True,
    )
    torch.testing.assert_close(ours_state, ref_state, rtol=2e-3, atol=2e-3)
