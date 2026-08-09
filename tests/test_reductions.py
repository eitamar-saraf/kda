"""The unified kernel must reproduce every architecture it claims to subsume.

``kda.textbook`` holds independent, literal transcriptions of Table 7 of the paper --
six separate loops that share no code. ``kda.recurrent.linear_attn`` is one loop with
two switches. If the two agree for every setting of those switches, then the claim the
writeup makes ("these six models differ only in gate granularity and whether the write
is corrective") is a tested fact rather than a nice story.
"""

from __future__ import annotations

import pytest
import torch

from kda import textbook
from kda.recurrent import VARIANTS, linear_attn, variant

B, H, T, DK, DV = 2, 3, 24, 8, 6
TOL = dict(rtol=1e-5, atol=1e-6)


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    return {
        "q": torch.randn(B, H, T, DK, generator=g, dtype=torch.float64),
        "k": torch.nn.functional.normalize(
            torch.randn(B, H, T, DK, generator=g, dtype=torch.float64), dim=-1
        ),
        "v": torch.randn(B, H, T, DV, generator=g, dtype=torch.float64),
        # decays in (0, 1), the range the sigmoid/exp parameterisations produce
        "alpha_scalar": torch.rand(B, H, T, generator=g, dtype=torch.float64) * 0.5 + 0.5,
        "alpha_channel": torch.rand(B, H, T, DK, generator=g, dtype=torch.float64) * 0.5 + 0.5,
        "beta": torch.rand(B, H, T, generator=g, dtype=torch.float64),
    }


def test_linear_attention(inputs):
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    ours = linear_attn(q, k, v, beta=torch.ones(B, H, T, dtype=torch.float64),
                       gate="none", delta=False)
    torch.testing.assert_close(ours, textbook.linear_attention(q, k, v), **TOL)


def test_mamba2(inputs):
    q, k, v, a, b = (inputs[x] for x in ("q", "k", "v", "alpha_scalar", "beta"))
    ours = linear_attn(q, k, v, a, b, gate="scalar", delta=False)
    torch.testing.assert_close(ours, textbook.mamba2(q, k, v, a, b), **TOL)


def test_gla(inputs):
    q, k, v, a = (inputs[x] for x in ("q", "k", "v", "alpha_channel"))
    ours = linear_attn(q, k, v, a, torch.ones(B, H, T, dtype=torch.float64),
                       gate="channel", delta=False)
    torch.testing.assert_close(ours, textbook.gla(q, k, v, a), **TOL)


def test_deltanet(inputs):
    q, k, v, b = (inputs[x] for x in ("q", "k", "v", "beta"))
    ours = linear_attn(q, k, v, beta=b, gate="none", delta=True)
    torch.testing.assert_close(ours, textbook.deltanet(q, k, v, b), **TOL)


def test_gated_deltanet(inputs):
    q, k, v, a, b = (inputs[x] for x in ("q", "k", "v", "alpha_scalar", "beta"))
    ours = linear_attn(q, k, v, a, b, gate="scalar", delta=True)
    torch.testing.assert_close(ours, textbook.gated_deltanet(q, k, v, a, b), **TOL)


def test_kda(inputs):
    """Eq. 1 -- the one the whole writeup is about."""
    q, k, v, a, b = (inputs[x] for x in ("q", "k", "v", "alpha_channel", "beta"))
    ours = linear_attn(q, k, v, a, b, gate="channel", delta=True)
    torch.testing.assert_close(ours, textbook.kda(q, k, v, a, b), **TOL)


def test_kda_with_broadcast_scalar_gate_is_gated_deltanet(inputs):
    """KDA's *only* structural change over GDN is that the gate became a vector.

    Broadcasting a scalar decay across all Dk channels must therefore turn KDA back
    into Gated DeltaNet exactly -- this is the reduction the writeup leans on hardest.
    """
    q, k, v, a_s, b = (inputs[x] for x in ("q", "k", "v", "alpha_scalar", "beta"))
    a_broadcast = a_s[..., None].expand(B, H, T, DK).contiguous()
    kda_out = linear_attn(q, k, v, a_broadcast, b, gate="channel", delta=True)
    torch.testing.assert_close(kda_out, textbook.gated_deltanet(q, k, v, a_s, b), **TOL)


def test_kda_with_unit_gate_is_deltanet(inputs):
    """alpha = 1 means "never forget", which is precisely DeltaNet."""
    q, k, v, b = (inputs[x] for x in ("q", "k", "v", "beta"))
    ones = torch.ones(B, H, T, DK, dtype=torch.float64)
    kda_out = linear_attn(q, k, v, ones, b, gate="channel", delta=True)
    torch.testing.assert_close(kda_out, textbook.deltanet(q, k, v, b), **TOL)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_dispatch_matches_textbook(inputs, name):
    """The VARIANTS registry the writeup quotes must actually dispatch correctly."""
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    spec = VARIANTS[name]
    alpha = inputs["alpha_scalar"] if spec["gate"] == "scalar" else inputs["alpha_channel"]
    ours = variant(name, q, k, v, alpha=alpha, beta=inputs["beta"])

    fn = getattr(textbook, name)
    args = [q, k, v]
    if spec["gate"] != "none":
        args.append(alpha)
    if spec["fixed_beta"] is None:
        args.append(inputs["beta"])
    torch.testing.assert_close(ours, fn(*args), **TOL)


def test_delta_rule_overwrites_rather_than_accumulates():
    """The behavioural claim behind the delta rule, stated as a test.

    Write (k, v1) then (k, v2) at the *same* key with beta=1 and no decay. The delta
    rule should leave the memory answering v2. Plain linear attention answers v1 + v2,
    because it never erased anything -- that is the interference the writeup's
    associative-memory widget visualises.
    """
    torch.manual_seed(0)
    k1 = torch.nn.functional.normalize(torch.randn(1, 1, 1, 8, dtype=torch.float64), dim=-1)
    k = k1.repeat(1, 1, 2, 1)                      # same key twice
    v = torch.randn(1, 1, 2, 4, dtype=torch.float64)
    q = k1                                          # query that key

    ones = torch.ones(1, 1, 2, dtype=torch.float64)
    delta_out = linear_attn(k, k, v, beta=ones, gate="none", delta=True)
    plain_out = linear_attn(k, k, v, beta=ones, gate="none", delta=False)

    torch.testing.assert_close(delta_out[:, :, 1], v[:, :, 1], **TOL)
    torch.testing.assert_close(plain_out[:, :, 1], v[:, :, 0] + v[:, :, 1], **TOL)
    assert q.shape[-1] == 8


def test_state_carry_is_equivalent_to_one_pass(inputs):
    """Splitting a sequence and carrying the state must equal running it whole.

    This is the property the chunkwise algorithm in Phase 1 depends on.
    """
    q, k, v, a, b = (inputs[x] for x in ("q", "k", "v", "alpha_channel", "beta"))
    whole = linear_attn(q, k, v, a, b, gate="channel", delta=True)

    split = T // 2
    first, state = linear_attn(
        q[:, :, :split], k[:, :, :split], v[:, :, :split],
        a[:, :, :split], b[:, :, :split],
        gate="channel", delta=True, return_state=True,
    )
    second = linear_attn(
        q[:, :, split:], k[:, :, split:], v[:, :, split:],
        a[:, :, split:], b[:, :, split:],
        gate="channel", delta=True, initial_state=state,
    )
    torch.testing.assert_close(torch.cat([first, second], dim=2), whole, **TOL)
