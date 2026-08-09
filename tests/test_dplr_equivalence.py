"""KDA is a constrained DPLR recurrence -- Sec. 6.2, checked rather than asserted.

The paper's efficiency argument is that KDA's transition matrix is a *special case* of
the general Diagonal-Plus-Low-Rank form, and that the specialisation is what removes
work. The benchmark in ``experiments/bench_kernel.py`` leans on this: it times the KDA
kernel against a general DPLR kernel and reports the ratio. That comparison only means
something if the two kernels are computing the same function -- otherwise it is timing
a fast thing against a different, slower thing and calling the difference a speedup.

So the mapping gets its own test, on CPU, against a naive DPLR loop:

    general DPLR (fla convention):
        S_t = (Diag(exp(gk_t)) + a_t b_t^T) S_{t-1} + k_t v_t^T

    KDA (Eq. 1):
        S_t = (I - b_t k_t k_t^T) Diag(a_t) S_{t-1} + b_t k_t v_t^T
            = (Diag(a_t) - b_t k_t (a_t * k_t)^T) S_{t-1} + b_t k_t v_t^T

    so       gk = log a,   a_vec = -beta * k,   b_vec = a * k,   v' = beta * v

using ``k^T Diag(a) = (a * k)^T``. The write term carries no coefficient in the DPLR
form, so beta folds into v.
"""

from __future__ import annotations

import torch

from kda.recurrent import linear_attn

B, H, T, DK, DV = 2, 2, 32, 8, 6
TOL = dict(rtol=1e-9, atol=1e-11)


def naive_dplr(q, k, v, a_vec, b_vec, gk):
    """S_t = (Diag(exp(gk_t)) + a_t b_t^T) S_{t-1} + k_t v_t^T ;  o_t = S_t^T q_t.

    A literal transcription of the general form, deliberately independent of anything
    in ``kda/`` so it cannot inherit the same mistake.
    """
    b, h, t, dk = q.shape
    S = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for i in range(t):
        # Both the diagonal and the rank-1 term act on S_{t-1}, simultaneously -- not
        # the rank-1 term acting on an already-decayed state. Getting this wrong makes
        # the transition (Diag + ab^T applied in sequence) a different operator, and
        # KDA then looks like it is *not* a DPLR special case when it is.
        prev = S
        bS = (prev * b_vec[:, :, i, :, None]).sum(dim=-2)        # b_i^T S_{t-1}, (B,H,Dv)
        S = (gk[:, :, i, :, None].exp() * prev
             + a_vec[:, :, i, :, None] * bS[..., None, :])
        # write
        S = S + k[:, :, i, :, None] * v[:, :, i, None, :]
        out.append((S * q[:, :, i, :, None]).sum(dim=-2))
    return torch.stack(out, dim=2)


def test_kda_is_a_constrained_dplr():
    """The mapping the benchmark relies on, verified exactly."""
    g = torch.Generator().manual_seed(0)
    dt = torch.float64
    q = torch.randn(B, H, T, DK, generator=g, dtype=dt)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, DK, generator=g, dtype=dt), dim=-1)
    v = torch.randn(B, H, T, DV, generator=g, dtype=dt)
    log_alpha = -torch.nn.functional.softplus(
        torch.randn(B, H, T, DK, generator=g, dtype=dt)) * 0.3
    beta = torch.rand(B, H, T, generator=g, dtype=dt)
    alpha = log_alpha.exp()

    kda_out = linear_attn(q, k, v, beta=beta, log_alpha=log_alpha,
                          gate="channel", delta=True)

    dplr_out = naive_dplr(
        q, k,
        v * beta[..., None],                 # the DPLR write has no coefficient
        a_vec=-(beta[..., None] * k),
        b_vec=alpha * k,
        gk=log_alpha,
    )
    torch.testing.assert_close(dplr_out, kda_out, **TOL)


def test_bench_mapping_matches_the_helper():
    """``experiments.bench_kernel.dplr_equivalent`` must produce that same mapping.

    The helper works in fla's ``[B, T, H, D]`` layout, so this also pins the transpose.
    """
    from experiments.bench_kernel import dplr_equivalent

    g = torch.Generator().manual_seed(1)
    dt = torch.float64
    # fla layout: (B, T, H, D)
    q = torch.randn(B, T, H, DK, generator=g, dtype=dt)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, DK, generator=g, dtype=dt), dim=-1)
    v = torch.randn(B, T, H, DV, generator=g, dtype=dt)
    gl = -torch.nn.functional.softplus(torch.randn(B, T, H, DK, generator=g, dtype=dt)) * 0.3
    beta = torch.rand(B, T, H, generator=g, dtype=dt)

    qd, kd, vd, ad, bd, gd = dplr_equivalent(q, k, v, gl, beta)

    tr = lambda x: x.transpose(1, 2).contiguous()
    kda_out = linear_attn(tr(q), tr(k), tr(v), beta=tr(beta.unsqueeze(-1)).squeeze(-1),
                          log_alpha=tr(gl), gate="channel", delta=True)
    dplr_out = naive_dplr(tr(qd), tr(kd), tr(vd), tr(ad), tr(bd), tr(gd))
    torch.testing.assert_close(dplr_out, kda_out, **TOL)


def test_dplr_is_strictly_more_general():
    """Sanity: an unconstrained DPLR update is *not* expressible as KDA.

    If it were, the paper's efficiency claim would be vacuous -- there would be nothing
    special about binding a = b = k. Picking independent a and b should give a different
    answer from any KDA run with the same diagonal.
    """
    g = torch.Generator().manual_seed(2)
    dt = torch.float64
    q = torch.randn(B, H, T, DK, generator=g, dtype=dt)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, DK, generator=g, dtype=dt), dim=-1)
    v = torch.randn(B, H, T, DV, generator=g, dtype=dt)
    log_alpha = -torch.nn.functional.softplus(
        torch.randn(B, H, T, DK, generator=g, dtype=dt)) * 0.3
    beta = torch.rand(B, H, T, generator=g, dtype=dt)

    # a and b unrelated to k
    a_vec = torch.randn(B, H, T, DK, generator=g, dtype=dt) * 0.1
    b_vec = torch.randn(B, H, T, DK, generator=g, dtype=dt) * 0.1
    free = naive_dplr(q, k, v * beta[..., None], a_vec, b_vec, log_alpha)
    kda = linear_attn(q, k, v, beta=beta, log_alpha=log_alpha, gate="channel", delta=True)
    assert (free - kda).abs().max() > 1e-3, "a free DPLR update should not coincide with KDA"
