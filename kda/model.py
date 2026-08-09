"""A small language model whose token mixer is swappable.

Used two ways:

* Sec. 5.1 (synthetic tasks) -- a pure stack of one linear-attention variant, so the
  only difference between runs is the memory mechanism.
* Sec. 5.2 (hybrid ratio) -- ``hybrid_ratio=N`` interleaves N linear layers with one
  full-attention layer, the structure the paper settles on at N=3.

Everything else (embeddings, SwiGLU MLP, pre-norm residuals, tied head) is held fixed
across every configuration so the comparisons stay honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kda.layers import RMSNorm, LinearAttentionLayer, ShortConv

__all__ = ["ModelConfig", "LanguageModel", "layer_plan"]


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 2
    variant: str = "kda"
    #: None -> every layer linear; 0 -> every layer full attention;
    #: N -> N linear layers per full-attention layer (the paper uses 3).
    hybrid_ratio: int | None = None
    use_conv: bool = True
    output_gate: str = "sigmoid"
    chunkwise: bool = True
    mlp_ratio: float = 4.0
    rope: bool = False           # full-attention layers use NoPE by default, per Sec. 4
    attn_conv: bool = True       # give attention the same short conv the linear layers have
    #: Longest dependency the gated variants are initialised to be able to hold.
    #: See kda.layers.timescale_dt_range -- set this to the sequence length or more.
    max_timescale: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True


def layer_plan(n_layers: int, hybrid_ratio: int | None) -> list[str]:
    """Which layers are linear and which are full attention.

    ``hybrid_ratio=3`` gives ``[linear, linear, linear, full, linear, ...]`` -- the
    full-attention layer goes last in each group so every block of linear layers gets
    its state refreshed by a global view immediately afterwards.
    """
    if hybrid_ratio is None:
        return ["linear"] * n_layers
    if hybrid_ratio == 0:
        return ["full"] * n_layers
    plan = []
    for i in range(n_layers):
        plan.append("full" if (i + 1) % (hybrid_ratio + 1) == 0 else "linear")
    return plan


def _rope(x: Tensor, theta: float) -> Tensor:
    """Rotary embedding. Off by default -- the paper uses NoPE on full-attention layers
    and lets the linear layers carry position information (Sec. 6.1)."""
    b, h, t, d = x.shape
    half = d // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * freqs[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class SoftmaxAttention(nn.Module):
    """Causal multi-head softmax attention -- the thing being replaced.

    Keeps every past key and value, so its cache grows linearly with context and its
    prefill cost grows quadratically. That is the bill the whole paper is trying to
    avoid paying, and here it is the reference for what perfect recall looks like.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # The linear-attention layers all carry a short convolution, which hands them
        # a previous-token operation for free -- exactly the primitive an induction
        # head has to learn from scratch. Giving attention the same convolution keeps
        # the comparison about the memory mechanism instead of about who got a free
        # local mixer. Table 1 of the paper ablates the convolution for the same reason.
        self.use_conv = cfg.attn_conv
        if self.use_conv:
            self.q_conv = ShortConv(cfg.d_model, 4)
            self.k_conv = ShortConv(cfg.d_model, 4)
            self.v_conv = ShortConv(cfg.d_model, 4)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if self.use_conv:
            q, k, v = self.q_conv(q), self.k_conv(k), self.v_conv(v)
        shape = lambda z: z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if self.cfg.rope:
            q, k = _rope(q, self.cfg.rope_theta), _rope(k, self.cfg.rope_theta)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(b, t, d))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ratio: float):
        super().__init__()
        hidden = int(d_model * ratio)
        self.w1 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model)
        if kind == "linear":
            self.mixer = LinearAttentionLayer(
                cfg.d_model, cfg.n_heads, variant=cfg.variant,
                use_conv=cfg.use_conv, output_gate=cfg.output_gate,
                chunkwise=cfg.chunkwise, max_timescale=cfg.max_timescale,
            )
        else:
            self.mixer = SoftmaxAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.plan = layer_plan(cfg.n_layers, cfg.hybrid_ratio)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, kind) for kind in self.plan)
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        # The decay gate's bias carries Mamba's dt initialisation; zeroing it here
        # would set the decay to e^-11 per token and the model would not train.
        if getattr(m, "_skip_default_init", False):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            return
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.head.weight.numel()
        return n

    def loss(self, tokens: Tensor, targets: Tensor, mask: Tensor | None = None):
        """Cross-entropy, restricted to ``mask`` when given.

        Returns ``(loss, accuracy)`` where accuracy is over the masked positions --
        the only number that means anything on these tasks.
        """
        logits = self(tokens)
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        sel = mask.reshape(-1)
        lg = logits.reshape(-1, logits.shape[-1])[sel]
        tg = targets.reshape(-1)[sel]
        loss = F.cross_entropy(lg, tg)
        acc = (lg.argmax(-1) == tg).float().mean()
        return loss, acc
