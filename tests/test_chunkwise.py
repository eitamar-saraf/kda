"""The fast form must compute exactly what the readable form computes.

``kda.recurrent.linear_attn`` is a transparent O(T) loop; ``kda.chunk`` is the
matmul-shaped rearrangement (Eq. 2-9) that a GPU can actually run. They are supposed
to be the same function. Everything downstream -- the benchmarks, the trained models,
the figures -- assumes that, so it gets asserted here across gate granularities, chunk
sizes, sequence lengths that do not divide evenly, and carried state.
"""

from __future__ import annotations

import pytest
import torch

from kda.chunk import chunk_linear_attn
from kda.recurrent import VARIANTS, linear_attn

B, H, DK, DV = 2, 2, 16, 12
TOL = dict(rtol=1e-8, atol=1e-9)


def make(t, dtype=torch.float64, seed=0, decay=(0.5, 1.0)):
    g = torch.Generator().manual_seed(seed)
    lo, hi = decay
    return {
        "q": torch.randn(B, H, t, DK, generator=g, dtype=dtype),
        "k": torch.nn.functional.normalize(
            torch.randn(B, H, t, DK, generator=g, dtype=dtype), dim=-1
        ),
        "v": torch.randn(B, H, t, DV, generator=g, dtype=dtype),
        "alpha_scalar": torch.rand(B, H, t, generator=g, dtype=dtype) * (hi - lo) + lo,
        "alpha_channel": torch.rand(B, H, t, DK, generator=g, dtype=dtype) * (hi - lo) + lo,
        "beta": torch.rand(B, H, t, generator=g, dtype=dtype),
    }


@pytest.mark.parametrize("name", sorted(VARIANTS))
@pytest.mark.parametrize("chunk_size", [1, 4, 16])
def test_chunkwise_matches_recurrent(name, chunk_size):
    """Every architecture, every chunk size, same answer."""
    t = 48
    d = make(t)
    spec = VARIANTS[name]
    alpha = d["alpha_scalar"] if spec["gate"] == "scalar" else d["alpha_channel"]
    if spec["gate"] == "none":
        alpha = None
    beta = (
        torch.ones(B, H, t, dtype=torch.float64)
        if spec["fixed_beta"] is not None
        else d["beta"]
    )
    kw = dict(gate=spec["gate"], delta=spec["delta"])

    ref = linear_attn(d["q"], d["k"], d["v"], alpha, beta, **kw)
    got = chunk_linear_attn(d["q"], d["k"], d["v"], alpha, beta, chunk_size=chunk_size, **kw)
    torch.testing.assert_close(got, ref, **TOL)


@pytest.mark.parametrize("t", [1, 7, 63, 64, 65, 130])
def test_ragged_sequence_lengths(t):
    """Padding to a whole number of chunks must not change the answer."""
    d = make(t)
    kw = dict(gate="channel", delta=True)
    ref = linear_attn(d["q"], d["k"], d["v"], d["alpha_channel"], d["beta"], **kw)
    got = chunk_linear_attn(
        d["q"], d["k"], d["v"], d["alpha_channel"], d["beta"], chunk_size=16, **kw
    )
    assert got.shape == (B, H, t, DV)
    torch.testing.assert_close(got, ref, **TOL)


def test_final_state_matches():
    """The carried state, not just the outputs -- Eq. 8 has to be right on its own."""
    t = 40
    d = make(t)
    kw = dict(gate="channel", delta=True)
    _, ref_s = linear_attn(
        d["q"], d["k"], d["v"], d["alpha_channel"], d["beta"], **kw, return_state=True
    )
    _, got_s = chunk_linear_attn(
        d["q"], d["k"], d["v"], d["alpha_channel"], d["beta"],
        chunk_size=8, return_state=True, **kw,
    )
    torch.testing.assert_close(got_s, ref_s, **TOL)


def test_initial_state_is_honoured():
    """Chunked prefill then continue: the state has to flow in as well as out."""
    t = 32
    d = make(t)
    kw = dict(gate="channel", delta=True, chunk_size=8)
    s0 = torch.randn(B, H, DK, DV, dtype=torch.float64) * 0.1

    ref = linear_attn(
        d["q"], d["k"], d["v"], d["alpha_channel"], d["beta"],
        gate="channel", delta=True, initial_state=s0,
    )
    got = chunk_linear_attn(
        d["q"], d["k"], d["v"], d["alpha_channel"], d["beta"], initial_state=s0, **kw
    )
    torch.testing.assert_close(got, ref, **TOL)


def test_stable_and_naive_agree_in_float64():
    """The two ways of getting K/G are algebraically identical...

    ...which is exactly why the *next* test is interesting: they only diverge once
    precision runs out.
    """
    d = make(64, decay=(0.85, 0.99))
    kw = dict(gate="channel", delta=True, chunk_size=32)
    a, b = d["alpha_channel"], d["beta"]
    stable = chunk_linear_attn(d["q"], d["k"], d["v"], a, b, stable=True, **kw)
    naive = chunk_linear_attn(d["q"], d["k"], d["v"], a, b, stable=False, **kw)
    torch.testing.assert_close(stable, naive, rtol=1e-6, atol=1e-8)


def test_naive_reciprocal_overflows_in_half():
    """Sec. 6.2's numerical claim, isolated to the quantity it is actually about.

    The decay-weighted Gram matrix is the one place ``1/G`` appears. Over a chunk of
    64 with a ~ 0.8 the cumulative decay reaches ~1e-8, so ``1/G`` reaches ~1e8 --
    well past fp16's 65504 ceiling. Forming that ratio destroys the result; asking
    for the same numbers as ``exp(gc_r - gc_i)`` keeps every factor in (0, 1] and
    lands within fp16's ordinary rounding error of the fp64 answer.

    This is the measurement behind the "making it fast" figure in the writeup.
    """
    from kda.chunk import decay_gram

    c, dk = 64, 16
    g = torch.Generator().manual_seed(0)
    k = torch.nn.functional.normalize(torch.randn(1, c, dk, generator=g, dtype=torch.float64), dim=-1)
    q = torch.randn(1, c, dk, generator=g, dtype=torch.float64)
    a = torch.rand(1, c, dk, generator=g, dtype=torch.float64) * 0.1 + 0.75
    gc = a.log().cumsum(dim=-2)

    assert (-gc).exp().max().item() > 65504, "test needs 1/G to exceed the fp16 ceiling"

    truth = decay_gram(q, k, gc, stable=True)

    def err(stable):
        got = decay_gram(q.half(), k.half(), gc.float(), stable=stable)
        return (got.double() - truth).abs().max().item()

    stable_err, naive_err = err(True), err(False)
    assert not torch.isfinite(torch.tensor(naive_err)) or naive_err > 100 * stable_err, (
        f"expected the 1/G route to blow up in fp16, "
        f"got stable={stable_err:.3e} naive={naive_err:.3e}"
    )
    assert stable_err < 1e-2, f"pairwise route should stay accurate, got {stable_err:.3e}"


def test_gradients_flow_and_match():
    """Backward through the chunkwise path must match backward through the loop."""
    t = 24
    d = make(t)
    a = d["alpha_channel"].clone().requires_grad_(True)
    b = d["beta"].clone().requires_grad_(True)
    a2 = d["alpha_channel"].clone().requires_grad_(True)
    b2 = d["beta"].clone().requires_grad_(True)
    kw = dict(gate="channel", delta=True)

    linear_attn(d["q"], d["k"], d["v"], a, b, **kw).square().sum().backward()
    chunk_linear_attn(d["q"], d["k"], d["v"], a2, b2, chunk_size=8, **kw).square().sum().backward()

    torch.testing.assert_close(a2.grad, a.grad, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(b2.grad, b.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_runs_end_to_end_in_low_precision(dtype):
    """The whole kernel, in the dtypes it is actually called with.

    Every other test here runs in float64, which hid a real bug: the gate maths is kept
    in fp32 on purpose, and one of those fp32 values multiplies the carried state. That
    silently promoted S on the first chunk, and the *second* chunk then failed with
    "expected scalar type BFloat16 but found Float" -- so it only reproduced at
    sequences longer than one chunk, which no float64 test would ever surface.
    """
    t = 96                                  # > chunk_size, so the state really is carried
    d = make(t, dtype=torch.float32, decay=(0.9, 0.999))
    cast = lambda x: x.to(dtype)
    ref = linear_attn(d["q"].double(), d["k"].double(), d["v"].double(),
                      d["alpha_channel"].double(), d["beta"].double(),
                      gate="channel", delta=True)

    got = chunk_linear_attn(
        cast(d["q"]), cast(d["k"]), cast(d["v"]),
        cast(d["alpha_channel"]), cast(d["beta"]),
        gate="channel", delta=True, chunk_size=32,
    )
    assert got.dtype == dtype, f"output dtype changed to {got.dtype}"
    assert torch.isfinite(got).all(), "non-finite output"

    # relative Frobenius error -- the right metric across precisions
    rel = ((got.double() - ref).norm() / ref.norm()).item()
    budget = {torch.float32: 1e-5, torch.bfloat16: 5e-2, torch.float16: 5e-2}[dtype]
    assert rel < budget, f"{dtype} relative error {rel:.3e} exceeds {budget:.0e}"
