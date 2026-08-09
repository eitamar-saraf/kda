"""The chunkwise-parallel form of KDA -- Eq. 2-9 of arXiv:2510.26692.

The sequential recurrence in :mod:`kda.recurrent` is correct but useless on a GPU: it
is T dependent matrix updates, each too small to fill a tensor core. The chunkwise form
splits the sequence into chunks of length C and makes everything *inside* a chunk a
handful of dense matmuls, leaving only chunk-to-chunk state passing sequential. Cost
goes from T tiny steps to T/C fat ones.

Derivation
----------
Write the recurrence in error form, with ``D_r = Diag(a_r)``::

    S^r = D_r S^{r-1} + b_r k_r (v_r - (D_r S^{r-1})^T k_r)^T

Every term ever added to the state is ``k_i`` times some row vector, subsequently
decayed by the gates that come after step ``i``. So with ``g_r = prod_{j<=r} a_j`` the
cumulative decay from the start of the chunk, the state has the closed form

    (*)   S^r = Diag(g_r) S^0 + sum_{i<=r} Diag(g_r / g_i) k_i c_i^T

for some vectors ``c_i`` we now solve for. Note ``a_i`` decays the state *before* the
write at step ``i``, so it does not act on the term written at ``i`` -- hence
``g_r / g_i`` and not ``g_r / g_{i-1}``.

Substituting (*) into the update and matching the ``r``-th term gives

    c_r = b_r ( v_r - S^{0T}(g_r * k_r) - sum_{i<r} A_{r,i} c_i )
    A_{r,i} = <k_i / g_i, g_r * k_r>

which is linear in the ``c``'s and lower-triangular, so it inverts in one shot. Stack
``c_r`` into ``C``, let ``K~ = G * K`` (row r is ``g_r * k_r``) and ``K^ = K / G``::

    A = K~ K^T                                    (only StrictTril(A) is used)
    M = (I + Diag(b) StrictTril(A))^{-1} Diag(b)  Eq. 6, the "UT transform"
    W = M K~ ,  U = M V                           Eq. 7
    C = U - W S^0

``I + Diag(b) StrictTril(A)`` is unit lower triangular, so the inverse is a forward
substitution -- cheap and exact. From there the state and outputs are two more matmuls::

    S^C = Diag(g_C) S^0 + K_bar^T C               Eq. 8,  K_bar row i = (g_C/g_i) * k_i
    O   = Q~ S^0 + Tril(Q~ K^T) C                 Eq. 9,  Q~ = G * Q
          '------'   '-------------'
          inter-chunk  intra-chunk

The ``W``/``U`` pair is the WY representation (Eq. 3-5): it compresses C rank-1
Householder-style updates into two dense C-by-d matrices, which is the whole reason a
sequence of delta-rule corrections can run on tensor cores at all.

The 1/G problem
---------------
``K^ = K / G`` divides by a cumulative product of numbers in (0, 1). Over a chunk of
64 with a ~ 0.9 that is a factor of ~1e3, and in bf16 it is a precision disaster. The
paper (Sec. 6.2) notes GLA works around it by moving to log space with a second level
of chunking, at a real cost in speed.

The fix used here -- and in the official kernel -- is to never form ``1/G`` at all.
Both places it appears are *pairwise*: ``A_{r,i}`` and ``(Q~ K^T)_{r,i}`` only ever
need ``exp(gc_r - gc_i)`` for ``i <= r``, where ``gc = cumsum(log a)``. Since ``log a``
is negative, ``gc`` decreases, so every one of those exponents is ``<= 0`` and every
factor is in ``(0, 1]``. The blow-up is an artefact of factorising the ratio, not of
the maths. ``stable=True`` (the default) computes the differences; ``stable=False``
takes the naive route so the writeup can show exactly where it falls apart.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["chunk_linear_attn", "chunk_kda", "expand_gate", "decay_gram"]


def decay_gram(x: Tensor, y: Tensor, gc: Tensor, *, stable: bool = True) -> Tensor:
    """Decay-weighted Gram matrix ``R[r, i] = <y_i / g_i, g_r * x_r>`` for ``i <= r``.

    Both places the cumulative decay enters the chunkwise algorithm have this shape:
    ``A`` (built from K and K) and the intra-chunk read (built from Q and K). It is
    the one quantity where the ``1/G`` blow-up of Sec. 6.2 actually bites, so both
    routes to it live here side by side.

    ``stable=True``  -- rewrite the ratio as ``exp(gc_r - gc_i)``. Because ``gc`` is a
    cumulative sum of ``log a <= 0`` it decreases, so for ``i <= r`` every exponent is
    ``<= 0`` and every factor lands in ``(0, 1]``. Nothing can overflow.

    ``stable=False`` -- form ``y / G`` and ``G * x`` and take one matmul. Algebraically
    identical, and much friendlier to tensor cores, but ``1/G`` grows like
    ``a^-C``: at ``a = 0.8, C = 64`` that is ~1e6, past the fp16 range of 65504.

    Args:
        x, y: ``(..., C, D)``.
        gc: ``(..., C, D)`` inclusive cumulative ``log a``. Always float32 or better.
        stable: which route to take.

    Returns:
        ``(..., C, C)``, masked to the lower triangle (inclusive).
    """
    c = x.shape[-2]
    idx = torch.arange(c, device=x.device)
    lower = (idx[:, None] >= idx[None, :])

    if stable:
        rows = []
        for r in range(c):
            # For i <= r this exponent is <= 0 because gc is non-increasing. For
            # i > r it is positive and can overflow to inf -- and inf * 0 is NaN, not
            # 0, so the mask alone will not save us. Clamp first, then mask: the
            # clamp is inert on every entry we actually keep.
            diff = (gc[..., r : r + 1, :] - gc).clamp_max(0.0)
            d = diff.exp() * (idx <= r)[:, None]
            rows.append((x[..., r : r + 1, :] * y * d.to(x.dtype)).sum(-1))
        return torch.stack(rows, dim=-2)

    g = gc.exp()
    return ((x * g.to(x.dtype)) @ (y / g.to(y.dtype)).transpose(-1, -2)) * lower


def expand_gate(alpha: Tensor | None, gate: str, like: Tensor, identity: float = 1.0) -> Tensor:
    """Put any gate granularity into the common ``(B, H, T, Dk)`` channel form.

    This is what lets one chunkwise kernel serve all six architectures: a scalar gate
    is just a channel gate whose channels happen to agree, and no gate is a channel
    gate of ones. ``like`` supplies shape, dtype and device; ``identity`` is the
    no-decay value (1 for a plain gate, 0 for a log-space one).
    """
    if gate == "none":
        return torch.full_like(like, identity)
    if gate == "scalar":
        return alpha[..., None].expand_as(like)
    if gate == "channel":
        return alpha
    raise ValueError(f"unknown gate {gate!r}")


def chunk_linear_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor | None = None,
    beta: Tensor | None = None,
    *,
    gate: str = "channel",
    delta: bool = True,
    log_alpha: Tensor | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    return_state: bool = False,
    stable: bool = True,
) -> Tensor | tuple[Tensor, Tensor]:
    """Chunkwise-parallel evaluation of the same recurrence as :func:`kda.linear_attn`.

    Args mirror :func:`kda.recurrent.linear_attn`, plus:
        chunk_size: C. The paper uses 64.
        stable: compute pairwise ``exp(gc_r - gc_i)`` instead of materialising ``1/G``.

    ``log_alpha`` is the preferred input: this kernel wants ``cumsum(log a)`` anyway,
    so handing it the log directly skips a lossy exp/log round trip.

    Returns ``(B, H, T, Dv)`` outputs, and the final state if ``return_state``.
    """
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    cs = chunk_size

    if scale != 1.0:
        q = q * scale

    if beta is None:
        beta = q.new_ones(b, h, t)

    # The kernel only ever wants log a, so build it in log space and stay there.
    # Kept at float32 or better whatever the compute dtype: over a chunk of 64 the
    # cumulative decay can reach ~1e-8, already outside fp16's normal range.
    gate_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    if log_alpha is not None:
        la = expand_gate(log_alpha, gate, q, identity=0.0).to(gate_dtype)
    else:
        la = expand_gate(alpha, gate, q, identity=1.0).to(gate_dtype).clamp_min(1e-12).log()

    if not delta:
        # Without the delta rule the write is unconditional. Setting the corrective
        # term's key to zero removes it while leaving the b_t k_t v_t^T write intact.
        k_delta = torch.zeros_like(k)
    else:
        k_delta = k

    # ---- pad to a whole number of chunks -------------------------------------
    pad = (cs - t % cs) % cs
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        k_delta = F.pad(k_delta, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        la = F.pad(la, (0, 0, 0, pad), value=0.0)  # log 1 = 0: no decay in the padding
        beta = F.pad(beta, (0, pad))               # no write in the padding
    tp = t + pad
    n = tp // cs

    def chunks(x, last):
        return x.reshape(b, h, n, cs, last)

    q, k, k_delta, v = (chunks(x, x.shape[-1]) for x in (q, k, k_delta, v))
    la = chunks(la, dk)
    beta = beta.reshape(b, h, n, cs)

    # ---- cumulative decay ----------------------------------------------------
    # gc[..., r, :] = sum_{j<=r} log a_j   (inclusive), so exp(gc) = g_r.
    gc = la.cumsum(dim=-2)
    g = gc.exp()                       # G, all entries in (0, 1]
    g_last = g[..., -1:, :]            # g_C

    q_t = (q * g.to(q.dtype))          # Q~ = G * Q
    k_t = (k_delta * g.to(q.dtype))    # K~ = G * K   (delta path only)
    # K_bar row i = (g_C/g_i) * k_i. Computed as a difference of logs so the ratio is
    # never formed from a near-zero denominator; every entry is in (0, 1].
    k_bar = k * (gc[..., -1:, :] - gc).exp().to(q.dtype)

    # ---- the two decay-weighted Gram matrices --------------------------------
    # A[r, i]  = <k_i / g_i, g_r * k_r>   strictly lower -- the WY/UT solve
    # QK[r, i] = <k_i / g_i, g_r * q_r>   lower inclusive -- the intra-chunk read
    A = decay_gram(k_delta, k_delta, gc, stable=stable)
    QK = decay_gram(q, k, gc, stable=stable)

    tril = torch.ones(cs, cs, dtype=torch.bool, device=q.device).tril()
    A = A * tril.tril(-1)              # StrictTril: a token does not correct itself
    QK = QK * tril                     # Tril: a token *can* read what it just wrote

    # ---- UT transform: one triangular solve replaces C rank-1 updates --------
    # M = (I + Diag(b) StrictTril(A))^{-1} Diag(b);  W = M K~,  U = M V.
    # Solving against the stacked right-hand side avoids ever forming M.
    #
    # The solve runs in at least fp32 even when the rest of the layer is half. It is
    # a forward substitution over C steps, so rounding compounds down the chunk, and
    # it is a negligible share of the FLOPs -- the official kernels make the same
    # trade. Precision differences that survive this therefore come from the inputs,
    # which is what the `stable` flag is about.
    solve_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    eye = torch.eye(cs, dtype=solve_dtype, device=q.device)
    L = eye + (beta[..., None] * A).to(solve_dtype)   # unit lower triangular
    rhs = (beta[..., None] * torch.cat([k_t, v], dim=-1)).to(solve_dtype)
    wu = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True).to(q.dtype)
    w, u = wu.split([dk, dv], dim=-1)

    # ---- sequential over chunks, parallel within ----------------------------
    # Match the compute dtype: an fp32 state carried into a bf16 layer promotes S on
    # the first chunk and the next chunk's q_t @ S then fails on mixed dtypes.
    S = (q.new_zeros(b, h, dk, dv) if initial_state is None
         else initial_state.to(q.dtype).clone())
    out = []
    for i in range(n):
        c = u[:, :, i] - w[:, :, i] @ S               # C = U - W S^0
        out.append(q_t[:, :, i] @ S + QK[:, :, i] @ c)          # Eq. 9
        S = g_last[:, :, i].transpose(-1, -2) * S + k_bar[:, :, i].transpose(-1, -2) @ c  # Eq. 8

    o = torch.stack(out, dim=2).reshape(b, h, tp, dv)[:, :, :t]
    return (o, S) if return_state else o


def chunk_kda(q, k, v, alpha, beta, **kw):
    """KDA proper (Eq. 1) in chunkwise form: channel gate, delta rule on."""
    return chunk_linear_attn(q, k, v, alpha, beta, gate="channel", delta=True, **kw)
