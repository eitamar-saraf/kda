"""Generate ``notebooks/annotated_kda.ipynb``.

The notebook is generated rather than hand-edited so its prose and its code cannot
drift apart from the package: every claim in it is a cell that runs, and the whole
thing is executed in CI-style by ``--execute`` before being written.

Usage::

    python -m scripts.build_notebook --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

MD, CODE = "markdown", "code"

CELLS: list[tuple[str, str]] = [
(MD, r"""# The Annotated Kimi Delta Attention

A runnable walk through **Kimi Delta Attention** ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)),
building its one equation from scratch and checking every claim as we go.

The destination:

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top,
\qquad o_t = S_t^\top q_t$$

Nothing here needs a GPU. Everything runs in float64 in a few seconds, because the point
is to see *what* is computed, not how fast.

Companion article: [Kimi Delta Attention](https://eitamar-saraf.github.io/blog/kimi-linear/)."""),

(CODE, """import torch
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False, linewidth=110)
torch.manual_seed(0)
DT = torch.float64          # exact enough that any discrepancy below is real, not rounding"""),

(MD, r"""## 1. Attention's memory grows; a recurrent state's does not

Drop the softmax from attention and you are left with three matrices multiplied together,
$O = (QK^\top)V$. Matrix multiplication is associative, so you may instead compute
$Q(K^\top V)$ — the same answer, but the intermediate is $d \times d$ instead of
$T \times T$.

That regrouping is the whole idea. Let's verify it is really the same answer, and count
what each order costs."""),

(CODE, """T, d = 64, 8
Q, K, V = (torch.randn(T, d, dtype=DT) for _ in range(3))

left  = (Q @ K.T) @ V          # T x T intermediate
right = Q @ (K.T @ V)          # d x d intermediate
print("same answer:", torch.allclose(left, right))
print(f"intermediate sizes: left {tuple((Q @ K.T).shape)}, right {tuple((K.T @ V).shape)}")

flops_left  = 2 * T * T * d + 2 * T * T * d
flops_right = 2 * T * d * d + 2 * T * d * d
print(f"FLOPs: left {flops_left:,}  right {flops_right:,}   ratio {flops_left/flops_right:.2f}x")"""),

(MD, r"""Note the crossover: at $T = 64, d = 8$ the "efficient" order is already 8x cheaper, but
for $T < d$ it would be *worse*. Linear attention is not free — it is a different
asymptote, and you only collect on it at long context.

Made causal, the right-hand order becomes a state you carry forward:

$$S_t = S_{t-1} + k_t v_t^\top, \qquad o_t = S_t^\top q_t$$"""),

(CODE, """from kda.textbook import linear_attention, softmax_attention

# our package uses (B, H, T, D)
q, k, v = (x[None, None] for x in (Q, K, V))
o_lin = linear_attention(q, k, v)
print("linear attention output:", tuple(o_lin.shape))

# the state is d x d no matter how long the sequence is
for t in (64, 4096, 1_000_000):
    kv_cache_floats = 2 * t * d       # attention keeps every k and v
    state_floats = d * d              # the recurrence keeps one matrix
    print(f"T={t:>9,}:  KV cache {kv_cache_floats:>12,} floats   recurrent state {state_floats:>6,}")"""),

(MD, r"""## 2. That state is an associative memory, and it interferes with itself

Writing $S = \sum_i k_i v_i^\top$ and reading with $k_j$ gives

$$S^\top k_j = \underbrace{\lVert k_j \rVert^2 v_j}_{\text{signal}} \;+\; \underbrace{\sum_{i \neq j} (k_i \cdot k_j)\, v_i}_{\text{interference}}$$

The second term vanishes only if the keys are orthogonal — and $d$ dimensions hold at
most $d$ orthogonal directions. Let's watch it break."""),

(CODE, """def signal_and_interference(keys, values, probe):
    \"\"\"Split S^T k_probe into the term we wanted and the sum of everything else.\"\"\"
    kq = keys[probe]
    signal = (kq @ kq) * values[probe]
    noise = sum((keys[i] @ kq) * values[i]
                for i in range(len(keys)) if i != probe)
    return signal, noise

dk, dv = 8, 4
print(f"{'keys':>5}  {'signal':>8}  {'interference':>13}  {'error / |v|':>12}")
for n in (2, 4, 8, 16, 32):
    # average over many random draws: a single draw is dominated by how (un)lucky
    # that particular set of directions was
    sig, noi, rel = 0.0, 0.0, 0.0
    trials = 200
    for t in range(trials):
        gg = torch.Generator().manual_seed(1000 * n + t)
        keys = F.normalize(torch.randn(n, dk, generator=gg, dtype=DT), dim=-1)
        values = F.normalize(torch.randn(n, dv, generator=gg, dtype=DT), dim=-1)
        s, e = signal_and_interference(keys, values, 0)
        sig += s.norm().item() / trials
        noi += e.norm().item() / trials
        rel += ((s + e) - values[0]).norm().item() / trials
    print(f"{n:>5}  {sig:>8.3f}  {noi:>13.3f}  {rel:>12.3f}")
print(f"\\nThe keys are unit vectors, so the signal is always 1. Only the leak grows —")
print(f"and past {dk} keys it cannot be reduced, because {dk} dimensions hold at most")
print(f"{dk} mutually orthogonal directions.")"""),

(MD, r"""## 3. The delta rule: descend a loss that has a floor

Reframe the state as a model being trained during the forward pass. Plain linear
attention descends $\mathcal{L} = -\langle S^\top k_t, v_t\rangle$, which is unbounded —
it just keeps reinforcing, with no criterion for what to erase.

DeltaNet descends a reconstruction loss instead:

$$\mathcal{L}_t(S) = \tfrac{1}{2}\lVert S^\top k_t - v_t \rVert^2$$

One gradient step with learning rate $\beta_t$:

$$S_t = S_{t-1} + \beta_t k_t\,(v_t - S_{t-1}^\top k_t)^\top$$

The parenthesis is the *error* between what the memory already answers and what we meant
to store. So writing a key twice corrects it rather than stacking on it."""),

(CODE, """from kda.recurrent import linear_attn

# write the SAME key twice with different values, beta = 1, no decay
kk = F.normalize(torch.randn(1, 1, 1, dk, dtype=DT), dim=-1)
keys = kk.repeat(1, 1, 2, 1)
vals = torch.randn(1, 1, 2, dv, dtype=DT)
ones = torch.ones(1, 1, 2, dtype=DT)

delta = linear_attn(keys, keys, vals, beta=ones, gate="none", delta=True)
plain = linear_attn(keys, keys, vals, beta=ones, gate="none", delta=False)

print("stored second value :", vals[0, 0, 1])
print("delta rule returns  :", delta[0, 0, 1], " <- the second value, exactly")
print("plain rule returns  :", plain[0, 0, 1], " <- v1 + v2, both still in there")"""),

(CODE, """# and beta really is a learning rate: it closes that fraction of the gap each write
v_target = torch.tensor([1.0, -0.5, 0.8, 0.2], dtype=DT)
for beta in (0.0, 0.25, 0.5, 1.0):
    reps = 6
    ks = kk.repeat(1, 1, reps, 1)
    vs = v_target.repeat(1, 1, reps, 1)
    out = linear_attn(ks, ks, vs, beta=torch.full((1, 1, reps), beta, dtype=DT),
                      gate="none", delta=True)
    errs = [(out[0, 0, i] - v_target).norm().item() for i in range(reps)]
    print(f"beta={beta:.2f}:  " + "  ".join(f"{e:.3f}" for e in errs))"""),

(MD, r"""## 4. Forgetting, and the one change KDA makes

The delta rule stops keys corrupting each other but never forgets: a fact written at
token 5 still occupies capacity at token 500,000. Gated DeltaNet multiplies the state by
a decay $\alpha_t \in [0,1]$ first — but $\alpha_t$ is **one scalar per head**, so
everything ages together.

KDA makes it a vector. That is the entire contribution:

$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top$$

Two switches — how fine-grained the decay is, and whether the write is corrective —
generate six published architectures."""),

(CODE, """from kda import textbook
from kda.recurrent import VARIANTS, variant

B, H, T2 = 1, 2, 24
g = torch.Generator().manual_seed(7)
qq = torch.randn(B, H, T2, dk, generator=g, dtype=DT)
kk2 = F.normalize(torch.randn(B, H, T2, dk, generator=g, dtype=DT), dim=-1)
vv = torch.randn(B, H, T2, dv, generator=g, dtype=DT)
a_scalar = torch.rand(B, H, T2, generator=g, dtype=DT) * 0.4 + 0.6
a_channel = torch.rand(B, H, T2, dk, generator=g, dtype=DT) * 0.4 + 0.6
bb = torch.rand(B, H, T2, generator=g, dtype=DT)

print(f"{'model':<18}{'gate':<10}{'delta':<7}matches its own textbook transcription")
print("-" * 74)
for name, spec in VARIANTS.items():
    alpha = a_scalar if spec["gate"] == "scalar" else a_channel
    ours = variant(name, qq, kk2, vv, alpha=alpha, beta=bb)
    args = [qq, kk2, vv]
    if spec["gate"] != "none":
        args.append(alpha)
    if spec["fixed_beta"] is None:
        args.append(bb)
    ref = getattr(textbook, name)(*args)
    ok = torch.allclose(ours, ref, rtol=1e-9, atol=1e-11)
    print(f"{name:<18}{spec['gate']:<10}{str(spec['delta']):<7}{ok}")"""),

(MD, r"""Two of those are worth stating as reductions, because they pin down exactly what KDA
changed. Broadcast a scalar decay across every channel and KDA *is* Gated DeltaNet; set
the decay to 1 and it *is* DeltaNet."""),

(CODE, """kda_broadcast = linear_attn(qq, kk2, vv, a_scalar[..., None].expand(B, H, T2, dk).contiguous(),
                            bb, gate="channel", delta=True)
gdn = textbook.gated_deltanet(qq, kk2, vv, a_scalar, bb)
print("KDA with a broadcast scalar gate == Gated DeltaNet:",
      torch.allclose(kda_broadcast, gdn, rtol=1e-9, atol=1e-11))

kda_ungated = linear_attn(qq, kk2, vv, torch.ones(B, H, T2, dk, dtype=DT), bb,
                          gate="channel", delta=True)
dn = textbook.deltanet(qq, kk2, vv, bb)
print("KDA with alpha = 1                == DeltaNet:        ",
      torch.allclose(kda_ungated, dn, rtol=1e-9, atol=1e-11))"""),

(MD, r"""## 5. Why per-channel decay buys anything

A scalar gate forces one memory half-life on the whole head. A channel gate lets the
same head hold one thing while flushing another. Here is that difference as retention
after 30 tokens."""),

(CODE, """def retention(half_life, steps=30):
    return 0.5 ** (steps / half_life)

print("scalar gate  (one dial for the whole head)")
for hl in (4, 12, 40):
    print(f"   half-life {hl:>3} tokens -> {retention(hl)*100:8.3f}% of a 30-token-old fact survives")
print("\\nchannel gate (KDA: 128 dials per head, these are four of them)")
for hl in (160, 56, 12, 4):
    print(f"   half-life {hl:>3} tokens -> {retention(hl)*100:8.3f}%")
print("\\nSame head, same step: some channels keep almost everything, others almost nothing.")"""),

(MD, r"""## 6. The chunkwise algorithm (Eq. 2–9)

The recurrence above is $T$ dependent updates, each too small to fill a tensor core.
The chunkwise form splits the sequence into blocks of $C$ and turns everything *inside*
a block into dense matmuls, leaving only block-to-block state passing sequential.

`kda/chunk.py` implements it and its module docstring carries the full derivation. The
part worth doing here is confirming it computes the same function — across gate types,
chunk sizes, and lengths that do not divide evenly."""),

(CODE, """from kda.chunk import chunk_linear_attn

print(f"{'model':<18}{'C=1':<8}{'C=4':<8}{'C=16':<8}")
print("-" * 42)
for name, spec in VARIANTS.items():
    alpha = a_scalar if spec["gate"] == "scalar" else a_channel
    if spec["gate"] == "none":
        alpha = None
    beta = torch.ones(B, H, T2, dtype=DT) if spec["fixed_beta"] is not None else bb
    kw = dict(gate=spec["gate"], delta=spec["delta"])
    ref = linear_attn(qq, kk2, vv, alpha, beta, **kw)
    cells = ""
    for C in (1, 4, 16):
        got = chunk_linear_attn(qq, kk2, vv, alpha, beta, chunk_size=C, **kw)
        cells += f"{str(torch.allclose(got, ref, rtol=1e-9, atol=1e-11)):<8}"
    print(f"{name:<18}{cells}")"""),

(CODE, """# ragged lengths must work too -- padding is internal and must not leak
for t in (1, 7, 63, 64, 65, 130):
    gg = torch.Generator().manual_seed(t)
    Qr = torch.randn(1, 1, t, dk, generator=gg, dtype=DT)
    Kr = F.normalize(torch.randn(1, 1, t, dk, generator=gg, dtype=DT), dim=-1)
    Vr = torch.randn(1, 1, t, dv, generator=gg, dtype=DT)
    Ar = torch.rand(1, 1, t, dk, generator=gg, dtype=DT) * 0.4 + 0.6
    Br = torch.rand(1, 1, t, generator=gg, dtype=DT)
    kw = dict(gate="channel", delta=True)
    r = linear_attn(Qr, Kr, Vr, Ar, Br, **kw)
    c = chunk_linear_attn(Qr, Kr, Vr, Ar, Br, chunk_size=16, **kw)
    print(f"T={t:>4}: shape {tuple(c.shape)}  matches {torch.allclose(c, r, rtol=1e-9, atol=1e-11)}")"""),

(MD, r"""## 7. The $1/\Gamma$ trap

The chunkwise algebra divides by the cumulative decay $\Gamma$. Over 64 tokens with
$\alpha \approx 0.8$ that reciprocal reaches $10^6$ — past float16's largest finite value
of 65504. The fix is not more precision, it is refactoring: every occurrence is a
*pairwise* ratio $\gamma_r/\gamma_i$ with $i \le r$, which is
$\exp(gc_r - gc_i)$ with a non-positive exponent, so every factor lands in $(0, 1]$.

Same numbers. One overflows, one doesn't."""),

(CODE, """from kda.chunk import decay_gram

C_, D_ = 64, 16
gg = torch.Generator().manual_seed(0)
x = torch.randn(1, C_, D_, generator=gg, dtype=DT)
y = F.normalize(torch.randn(1, C_, D_, generator=gg, dtype=DT), dim=-1)
alpha = torch.rand(1, C_, D_, generator=gg, dtype=DT) * 0.1 + 0.75
gc = alpha.log().cumsum(dim=-2)

print(f"largest 1/Gamma in this chunk : {(-gc).exp().max():.3e}")
print(f"float16 can represent up to   : 65504")

truth = decay_gram(x, y, gc, stable=True)
for stable in (True, False):
    got = decay_gram(x.half(), y.half(), gc.float(), stable=stable)
    bad = (~torch.isfinite(got)).sum().item()
    err = (got.double() - truth).abs().max().item()
    label = "pairwise exp(gc_r - gc_i)" if stable else "form 1/Gamma explicitly  "
    print(f"{label}:  non-finite entries {bad:>5} / {C_*C_}   max error {err:.3e}")"""),

(MD, r"""## 8. The hybrid

Pure linear attention still cannot copy a long span verbatim — a fixed state has nowhere
to put it. So Kimi Linear interleaves three KDA layers with one full-attention layer, and
pays a KV cache on only a quarter of its layers."""),

(CODE, """from kda.model import LanguageModel, ModelConfig, layer_plan

for ratio, name in [(0, "all attention"), (1, "1:1"), (3, "3:1 (the paper)"), (None, "all KDA")]:
    plan = layer_plan(16, ratio)
    n_attn = plan.count("full")
    seq_len, heads, head_dim = 1_000_000, 32, 128
    d_model = heads * head_dim
    kv = n_attn * 2 * seq_len * d_model * 2                     # bf16 K and V
    state = (16 - n_attn) * heads * head_dim * head_dim * 2     # one Dk x Dv per head
    print(f"{name:<18}{''.join('A' if p == 'full' else 'K' for p in plan)}"
          f"   {(kv + state) / 2**30:7.2f} GiB at 1M tokens")"""),

(CODE, """# and it builds and trains, for every variant
cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=4, n_heads=2,
                  variant="kda", hybrid_ratio=3)
model = LanguageModel(cfg)
tok = torch.randint(0, 256, (2, 32))
loss, acc = model.loss(tok, torch.randint(0, 256, (2, 32)))
loss.backward()
print(f"plan={''.join('A' if p=='full' else 'K' for p in model.plan)}  "
      f"params={model.num_params():,}  loss={loss.item():.3f}  finite grads="
      f"{all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)}")"""),

(MD, r"""## Where to go next

- `kda/textbook.py` — one literal loop per row of the paper's Table 7.
- `kda/chunk.py` — the chunkwise derivation, written to be read against Eq. 2–9.
- `kda/fast.py` — the production Triton kernels, pinned to the reference by tests.
- `tests/` — the reduction identities, the chunkwise equivalence, the DPLR specialisation,
  and comparisons against the kernels the Kimi team released.

Run `pytest` for the full argument; it takes a few seconds on a laptop."""),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == MD else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    # Deterministic ids. nbformat assigns random ones, so without this a rebuild shows
    # up as a large diff even when nothing changed -- unhelpful for a generated file
    # that is meant to be regenerated often.
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:02d}"
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def strip_volatile(nb: nbf.NotebookNode) -> None:
    """Remove wall-clock timestamps so the same content produces the same bytes."""
    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)
        for out in cell.get("outputs", []):
            out.get("metadata", {}).pop("execution", None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notebooks/annotated_kda.ipynb")
    p.add_argument("--execute", action="store_true",
                   help="run every cell and store the outputs; fails loudly on error")
    a = p.parse_args()

    nb = build()
    if a.execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=600, kernel_name="python3",
                               resources={"metadata": {"path": str(Path.cwd())}})
        client.execute()
        strip_volatile(nb)
        print("all cells executed without error")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    n_code = sum(1 for k, _ in CELLS if k == CODE)
    print(f"wrote {out}  ({len(CELLS)} cells, {n_code} of them code)")


if __name__ == "__main__":
    main()
