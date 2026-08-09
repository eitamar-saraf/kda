"""Every production kernel must compute what our readable reference computes.

This is the load-bearing test of the whole repo. The writeup explains ``kda/chunk.py``;
the experiments run ``kda/fast.py``. Without this file those are two unrelated pieces
of software that happen to live in the same directory.

Requires CUDA and flash-linear-attention.
"""

from __future__ import annotations

import pytest
import torch

from kda.fast import FLA_KERNELS, fast_linear_attn
from kda.recurrent import VARIANTS, linear_attn

pytest.importorskip("fla.ops", reason="flash-linear-attention not installed")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

B, H, T, DK, DV = 2, 4, 256, 64, 64

#: Kernels run in bfloat16 -- the dtype the experiments actually train in -- and are
#: compared against an fp32 reference. Elementwise relative error is the wrong metric
#: for that: near-zero reference entries blow it up while contributing nothing. The
#: standard kernel-validation metric is the relative Frobenius norm of the residual.
REL_TOL = 2e-2


def rel_err(got, ref):
    return ((got.float() - ref.float()).norm() / ref.float().norm()).item()


def assert_kernel_close(got, ref, tol=REL_TOL, label=""):
    e = rel_err(got, ref)
    assert e < tol, f"{label} relative error {e:.3e} exceeds {tol:.1e}"


def make(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.float32
    q = torch.randn(B, H, T, DK, generator=g, device=dev, dtype=dt)
    k = torch.nn.functional.normalize(
        torch.randn(B, H, T, DK, generator=g, device=dev, dtype=dt), dim=-1
    )
    v = torch.randn(B, H, T, DV, generator=g, device=dev, dtype=dt)
    # log-space decay in the range a trained gate actually produces
    la_ch = -torch.nn.functional.softplus(
        torch.randn(B, H, T, DK, generator=g, device=dev, dtype=dt)
    ) * 0.3
    la_sc = -torch.nn.functional.softplus(
        torch.randn(B, H, T, generator=g, device=dev, dtype=dt)
    ) * 0.3
    beta = torch.rand(B, H, T, generator=g, device=dev, dtype=dt)
    return q, k, v, la_ch, la_sc, beta


@pytest.mark.parametrize("name", sorted(FLA_KERNELS))
def test_fla_kernel_matches_our_reference(name):
    """Each production kernel against our reference, in the dtype we train in."""
    q, k, v, la_ch, la_sc, beta = make()
    spec = VARIANTS[name]
    scale = DK ** -0.5

    log_alpha = {"channel": la_ch, "scalar": la_sc, "none": None}[spec["gate"]]
    b = torch.ones_like(beta) if spec["fixed_beta"] is not None else beta

    ref = linear_attn(
        q, k, v, beta=b, log_alpha=log_alpha,
        gate=spec["gate"], delta=spec["delta"], scale=scale,
    )
    # some fla kernels require half precision; bf16 is what training uses anyway
    cast = lambda x: None if x is None else x.bfloat16()
    got = fast_linear_attn(
        cast(q), cast(k), cast(v), beta=cast(b), log_alpha=cast(log_alpha),
        gate=spec["gate"], delta=spec["delta"], scale=scale,
    )
    assert_kernel_close(got, ref, label=name)


@pytest.mark.parametrize("name", sorted(FLA_KERNELS))
def test_fla_kernel_gradients_match(name):
    """Forward agreement is not enough -- we train with these."""
    q, k, v, la_ch, la_sc, beta = make(seed=1)
    spec = VARIANTS[name]
    scale = DK ** -0.5
    log_alpha = {"channel": la_ch, "scalar": la_sc, "none": None}[spec["gate"]]
    b0 = torch.ones_like(beta) if spec["fixed_beta"] is not None else beta

    def grads(fn, dtype):
        cast = lambda x: x.to(dtype)
        qq = cast(q).clone().requires_grad_(True)
        vv = cast(v).clone().requires_grad_(True)
        la = None if log_alpha is None else cast(log_alpha).clone().requires_grad_(True)
        bb = cast(b0).clone().requires_grad_(True)
        out = fn(qq, cast(k), vv, beta=bb, log_alpha=la,
                 gate=spec["gate"], delta=spec["delta"], scale=scale)
        out.float().square().mean().backward()
        return qq.grad, vv.grad, (None if la is None else la.grad)

    ref = grads(linear_attn, torch.float32)
    got = grads(fast_linear_attn, torch.bfloat16)
    for a, c, label in zip(ref, got, ("dq", "dv", "dlog_alpha")):
        if a is None:
            continue
        # gradients tolerate more drift than activations, especially ungated where
        # the state grows without bound over the sequence
        assert_kernel_close(c, a, tol=5e-2, label=f"{name}/{label}")


def test_kda_final_state_matches():
    q, k, v, la_ch, _, beta = make(seed=2)
    scale = DK ** -0.5
    _, ref = linear_attn(q, k, v, beta=beta, log_alpha=la_ch, gate="channel",
                         delta=True, scale=scale, return_state=True)
    _, got = fast_linear_attn(q.bfloat16(), k.bfloat16(), v.bfloat16(),
                              beta=beta.bfloat16(), log_alpha=la_ch.bfloat16(),
                              gate="channel", delta=True, scale=scale, return_state=True)
    assert_kernel_close(got, ref, label="kda final state")
