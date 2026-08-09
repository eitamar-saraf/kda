"""The KDA token-mixing layer, and its five ancestors sharing one body.

Section 4 of the paper specifies the neural parameterisation around Eq. 1::

    q_t, k_t = L2Norm(Swish(ShortConv(W_qk x_t)))
    v_t      = Swish(ShortConv(W_v x_t))
    a_t      = f(W_a_up W_a_down x_t)  in [0, 1]^Dk     (low-rank, rank = head dim)
    b_t      = Sigmoid(W_b x_t)        in [0, 1]
    o_t      = W_o( Sigmoid(W_g_up W_g_down x_t) * RMSNorm(KDA(q, k, v, a, b)) )   Eq. 10

with ``f`` the Mamba/GDN decay, which the released kernel writes in log space as
``g = -exp(A_log) * softplus(proj(x) + dt_bias)``. That is always negative, so
``a = exp(g)`` lands in (0, 1) without a saturating sigmoid, and the chunk kernel gets
the log it wanted anyway.

Everything except the gate shape and the delta switch is shared by all six
architectures in :data:`kda.recurrent.VARIANTS`. That is deliberate: when the
synthetic-task curves separate, the cause is the memory mechanism, not a different
convolution or a missing output gate. The only parameter-count difference is the decay
projection -- ``Dk`` outputs per head for a channel gate, 1 for a scalar gate, 0 for
none -- which :meth:`LinearAttentionLayer.gate_params` reports so the writeup can
state it rather than hand-wave it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kda.chunk import chunk_linear_attn
from kda.fast import fast_linear_attn, fla_available
from kda.recurrent import VARIANTS, linear_attn

__all__ = ["ShortConv", "RMSNorm", "LinearAttentionLayer", "KDALayer", "timescale_dt_range"]


def _inv_softplus_init(n: int, dt_min: float = 1e-3, dt_max: float = 1e-1) -> Tensor:
    """Bias init such that ``softplus(bias)`` is log-uniform in ``[dt_min, dt_max]``.

    Straight from Mamba. ``softplus^-1(dt) = dt + log(-expm1(-dt))``, computed in that
    form because it stays accurate for small ``dt``.
    """
    dt = torch.empty(n).uniform_(math.log(dt_min), math.log(dt_max)).exp()
    return dt + torch.log(-torch.expm1(-dt))


def timescale_dt_range(max_timescale: int) -> tuple[float, float]:
    """Pick ``[dt_min, dt_max]`` so memory timescales span ~10 to ``max_timescale``.

    The decay rate is ``r = -log a = exp(A_log) * softplus(.)``, so a channel forgets
    on a timescale of ``1/r`` tokens. Whether the model can learn a task therefore
    depends on whether *any* channel starts with a timescale comparable to the distance
    the task requires.

    This matters more than it sounds. With Mamba's stock ``dt in [1e-3, 1e-1]`` and
    ``exp(A_log)`` up to 16, the slowest channel at initialisation forgets over ~1000
    tokens and most forget far sooner. Measured on MQAR with 8 key-value pairs, that
    model scores 1.00 at T=512 and 0.37 at T=1024 -- a cliff caused entirely by the
    initialisation, not by memory capacity (at T=1024 it still solves 4 pairs
    perfectly). Once the state has decayed before the query arrives there is no
    gradient left to tell the gate to decay more slowly, and the loss sits at exactly
    ln(num_values) forever.

    Scaling ``dt_min`` with the context the model is meant to handle is standard Mamba
    practice, and it is applied identically to every gated variant here so the
    comparison stays about gate *granularity* rather than about who got a luckier init.
    """
    return 1.0 / max(16, max_timescale), 1e-1


class RMSNorm(nn.Module):
    """Root-mean-square norm, applied per head to the attention output (Eq. 10)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class ShortConv(nn.Module):
    """Depthwise causal 1-D convolution, kernel 4.

    A cheap local mixer that every recent linear-attention model carries. Table 1 of
    the paper ablates it: removing it costs real perplexity, because a fixed-size
    recurrent state is a poor place to keep "what were the last three tokens".
    """

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D). Left-pad so position t never sees t+1.
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        return self.conv(x).transpose(1, 2)


class LinearAttentionLayer(nn.Module):
    """One token-mixing layer, instantiable as any member of the family.

    Args:
        d_model: model width.
        num_heads: heads. Head dims are ``d_model // num_heads``.
        variant: a key of :data:`kda.recurrent.VARIANTS`.
        use_conv: include the short convolution (Table 1 ablation).
        output_gate: "sigmoid" (paper default), "swish", or "none" (Table 1 ablation).
        backend: which implementation of the recurrence to call.
            ``"fla"``    production Triton kernels -- what the experiments use.
            ``"chunk"``  our readable Eq. 2-9 implementation -- what the writeup explains.
            ``"loop"``   the O(T) reference -- slowest, clearest.
            ``"auto"``   fla when a GPU and the package are present, else chunk.
            All four are asserted equal in the test suite.
        chunkwise: deprecated alias; ``False`` selects the ``"loop"`` backend.
        a_log_init: initial decay range; heads are spread over it so that at init the
            model holds a spread of memory timescales rather than one.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        variant: str = "kda",
        *,
        use_conv: bool = True,
        output_gate: str = "sigmoid",
        chunkwise: bool = True,
        backend: str = "auto",
        conv_size: int = 4,
        a_log_init: tuple[float, float] = (1.0, 4.0),
        max_timescale: int = 1024,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise KeyError(f"unknown variant {variant!r}; have {sorted(VARIANTS)}")
        spec = VARIANTS[variant]
        self.variant = variant
        self.gate = spec["gate"]
        self.delta = spec["delta"]
        self.fixed_beta = spec["fixed_beta"]
        if not chunkwise:
            backend = "loop"
        self.backend = backend
        self.output_gate = output_gate

        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.use_conv = use_conv
        if use_conv:
            self.q_conv = ShortConv(d_model, conv_size)
            self.k_conv = ShortConv(d_model, conv_size)
            self.v_conv = ShortConv(d_model, conv_size)

        # --- the decay gate: the one thing that differs across the family ---
        # Low-rank projection, rank = head_dim, as specified in Sec. 4.
        if self.gate == "channel":
            gate_out = d_model                       # Dk values per head
        elif self.gate == "scalar":
            gate_out = num_heads                     # 1 value per head
        else:
            gate_out = 0
        if gate_out:
            rank = self.head_dim
            self.a_down = nn.Linear(d_model, rank, bias=False)
            self.a_up = nn.Linear(rank, gate_out, bias=True)
            # A_log per head: exp(A_log) sets how fast this head *can* forget.
            lo, hi = a_log_init
            self.A_log = nn.Parameter(
                torch.linspace(math.log(lo), math.log(hi), num_heads)
            )
            # The bias matters more than it looks. log a = -exp(A_log) * softplus(.),
            # so a zero bias gives softplus(0) = 0.69 and, with exp(A_log) up to 16,
            # a decay of e^-11 per token: the state is erased before it is read and
            # nothing trains. Mamba's initialisation instead picks a target step size
            # dt in [dt_min, dt_max] and inverts softplus to get the bias, which puts
            # a in roughly [0.85, 1.0) at init -- long memory first, forgetting learned.
            with torch.no_grad():
                dt_min, dt_max = timescale_dt_range(max_timescale)
                self.a_up.bias.copy_(_inv_softplus_init(gate_out, dt_min, dt_max))
                # Keep the projection small at init so the bias above actually sets
                # the decay. With PyTorch's default Linear init the projection swamps
                # it and alpha collapses again -- the layer must not depend on its
                # parent model calling apply(init) to be usable.
                nn.init.normal_(self.a_down.weight, std=0.02)
                nn.init.normal_(self.a_up.weight, std=0.02)
            # Flag so a model-wide `apply(init)` does not zero the bias back out.
            self.a_up._skip_default_init = True

        if self.fixed_beta is None:
            self.b_proj = nn.Linear(d_model, num_heads, bias=True)

        self.norm = RMSNorm(self.head_dim)
        if output_gate != "none":
            rank = self.head_dim
            self.g_down = nn.Linear(d_model, rank, bias=False)
            self.g_up = nn.Linear(rank, d_model, bias=True)

    def gate_params(self) -> int:
        """Parameters spent on the decay gate -- what a channel gate actually costs."""
        if self.gate == "none":
            return 0
        return sum(p.numel() for p in (self.a_down.weight, self.a_up.weight,
                                       self.a_up.bias, self.A_log))

    def _heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, initial_state: Tensor | None = None,
                return_state: bool = False, return_gates: bool = False):
        """x: (B, T, d_model) -> (B, T, d_model)."""
        b, t, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if self.use_conv:
            q, k, v = self.q_conv(q), self.k_conv(k), self.v_conv(v)
        q, k, v = F.silu(q), F.silu(k), F.silu(v)

        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        # L2 normalising q and k bounds the eigenvalues of (I - b k k^T), which is
        # what keeps the delta-rule recurrence from blowing up over long sequences.
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        log_alpha = None
        if self.gate != "none":
            a = self.a_up(self.a_down(x))                       # (B, T, gate_out)
            if self.gate == "channel":
                a = self._heads(a)                              # (B, H, T, Dk)
                a_log = self.A_log.view(1, -1, 1, 1)
            else:
                a = a.transpose(1, 2)                           # (B, H, T)
                a_log = self.A_log.view(1, -1, 1)
            # g = -exp(A_log) * softplus(.)  <= 0, so alpha = exp(g) in (0, 1)
            log_alpha = -a_log.exp() * F.softplus(a.float())

        if self.fixed_beta is None:
            beta = self.b_proj(x).transpose(1, 2).sigmoid()      # (B, H, T)
        else:
            beta = torch.full((b, self.num_heads, t), self.fixed_beta,
                              device=x.device, dtype=x.dtype)

        backend = self.backend
        if backend == "auto":
            backend = "fla" if (fla_available() and q.is_cuda) else "chunk"
        fn = {"fla": fast_linear_attn, "chunk": chunk_linear_attn, "loop": linear_attn}[backend]
        out = fn(
            q, k, v, beta=beta, log_alpha=log_alpha,
            gate=self.gate, delta=self.delta, scale=self.scale,
            initial_state=initial_state, return_state=return_state,
        )
        o, state = out if return_state else (out, None)

        o = self.norm(o)                                         # per-head RMSNorm
        o = o.transpose(1, 2).reshape(b, t, self.d_model)
        if self.output_gate != "none":
            g = self.g_up(self.g_down(x))
            o = o * (g.sigmoid() if self.output_gate == "sigmoid" else F.silu(g))
        o = self.o_proj(o)

        if return_gates:
            gates = {"log_alpha": log_alpha, "beta": beta}
            return (o, state, gates) if return_state else (o, gates)
        return (o, state) if return_state else o


def KDALayer(d_model: int, num_heads: int = 4, **kw) -> LinearAttentionLayer:
    """Kimi Delta Attention proper: channel gate + delta rule."""
    return LinearAttentionLayer(d_model, num_heads, variant="kda", **kw)
