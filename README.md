# KDA — Kimi Delta Attention, implemented and illustrated

A readable, tested implementation of **Kimi Delta Attention** and the five architectures
it descends from, plus the experiments behind the writeup
*[Kimi Delta Attention, Illustrated](https://eitamar-saraf.github.io/blog/kimi-linear/)*.

Reference: Kimi Team, *[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)*
(arXiv:2510.26692).

---

## The idea in one equation

```
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ        o_t = S_tᵀ q_t
```

A fixed-size matrix `S` is an associative memory. `Diag(α_t)` decays it — one rate per
channel. `(I − β_t k_t k_tᵀ)` is the delta rule: it subtracts what the memory already
answers for `k_t` before writing, so a repeated key is corrected rather than piled on.

Gated DeltaNet is the same equation with `α_t` a **scalar**. That is the whole of KDA's
change, and everything else in the paper follows from it.

## One kernel, six architectures

`kda.recurrent.linear_attn` takes two switches — gate granularity and whether the write
is corrective — and reproduces six published models exactly:

| gate | delta | update | is |
|---|---|---|---|
| none | off | `S + k vᵀ` | Linear Attention |
| scalar | off | `α S + β k vᵀ` | Mamba2 |
| channel | off | `Diag(α) S + k vᵀ` | GLA |
| none | on | `(I − βkkᵀ) S + β k vᵀ` | DeltaNet |
| scalar | on | `(I − βkkᵀ) α S + β k vᵀ` | Gated DeltaNet |
| **channel** | **on** | `(I − βkkᵀ) Diag(α) S + β k vᵀ` | **KDA** |

This is enforced, not asserted in prose. `kda/textbook.py` holds six independent,
deliberately unoptimised transcriptions of the paper's Table 7, and the test suite
requires the unified kernel to match all six to floating-point tolerance.

## The correctness chain

Every link is a test, so the code the writeup explains is provably the code that runs:

```
six transcriptions of Table 7   ==   kda/recurrent.py   ==   kda/chunk.py
                                                              ||
     js/kda-math.js (the figures)   ==   fla.ops.kda (the Kimi team's Triton kernels)
```

- `tests/test_reductions.py` — the six reductions, plus the behavioural claim that the
  delta rule overwrites a repeated key while plain linear attention accumulates.
- `tests/test_chunkwise.py` — Eq. 2–9 against the sequential loop, across gate types,
  chunk sizes, ragged lengths, carried state, and gradients.
- `tests/test_against_fla.py` — against the released `chunk_kda`, `fused_recurrent_kda`
  and the naive reference (GPU).
- `tests/test_fast_backend.py` — all six production kernels against our reference,
  forward and backward (GPU).
- `js/kda-math.test.js` — the browser code in the article's figures against fixtures
  exported from the Python reference.
- `tests/test_dplr_equivalence.py` — KDA as a constrained diagonal-plus-low-rank
  recurrence, against an independent DPLR transcription. This is what makes the
  KDA-vs-DPLR speedup in the benchmark a comparison rather than a coincidence.

## Layout

```
kda/
  textbook.py    one literal loop per row of Table 7, sharing no code on purpose
  recurrent.py   the unified O(T) reference — the readable one
  chunk.py       Eq. 2–9: WY representation, UT transform, chunkwise parallel
  fast.py        dispatch to flash-linear-attention (≈12× faster; used for experiments)
  layers.py      the KDA module of Sec. 4: short conv, L2-normed q/k, low-rank gate,
                 sigmoid output gate + RMSNorm
  model.py       hybrid LM with a configurable KDA:attention ratio
  tasks/         MQAR, palindrome, interleaved LIFO stacks
experiments/     synthetic sweep, kernel benchmarks, LM pretraining, data prep
scripts/         figure generation, widget fixtures, notebook builder
js/              the browser implementation used by the article's interactive figures
notebooks/       annotated_kda.ipynb -- the whole derivation, runnable, CPU only
```

The notebook is *generated* by `python -m scripts.build_notebook --execute`, which runs
every cell before writing it. That way its prose and the package cannot drift apart:
every claim in it is a cell that executed.

## Running it

```bash
uv venv && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev,figures]"
pytest                      # 64 tests, CPU only, seconds
node --test js/             # the browser maths against the Python fixtures
```

The GPU tests and every experiment additionally need
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install flash-linear-attention
pytest                                            # now includes the fla comparisons
python -m experiments.synthetic --sweep --out runs/synthetic
python -m experiments.bench_kernel --out runs/bench/kernels.json
python -m experiments.prepare_data --tokens 1_000_000_000 --out data/
python -m experiments.pretrain --sweep --out runs/pretrain
python -m scripts.make_figures --runs runs/ --out figures/
```

## Four failure modes worth knowing about

Each of these produced healthy-looking curves while measuring nothing. All four are
now regression tests.

1. **`inf × 0 = NaN`.** In the decay-weighted Gram matrix, exponentiating before masking
   overflows above the diagonal, and `inf` times a zero mask is NaN. Clamp the exponent
   first; the clamp is inert on every entry that survives the mask.

2. **A zero bias erases the memory.** `log α = −exp(A)·softplus(·)`, so a zero bias gives
   `softplus(0) = 0.69`, and with `exp(A)` up to 16 that is `e⁻¹¹` decay *per token*. The
   loss pins at exactly `ln(vocab)` forever — once the state is wiped between writing a
   fact and reading it, no gradient survives to fix the gate. Mamba's `dt` initialisation
   is load-bearing.

3. **The decay init silently sets the result.** With Mamba's stock `dt ∈ [10⁻³, 10⁻¹]`,
   the same model scores **1.00 on MQAR at T=512 and 0.37 at T=1024** — not capacity (at
   T=1024 it still solves 4 pairs perfectly), purely initialisation. `max_timescale` must
   scale with the context being tested, or a comparison of gated architectures is partly
   a comparison of who got a luckier init.

4. **An unfair baseline flatters the result.** Every linear variant carries a short
   convolution, which hands it a previous-token operation for free. Without one, softmax
   attention stalls at 0.32 on MQAR while KDA scores 1.00 — which reads as "linear
   attention beats attention at recall", the opposite of the truth. Measured:
   rope+conv 1.00, conv only 0.92, rope only 0.32, neither 0.34. The convolution matters
   far more than the positional encoding.

## Also worth knowing

- **Never form `1/Γ`.** The chunkwise algebra divides by the cumulative decay, which over
  64 tokens at α ≈ 0.8 reaches 10⁶ — past fp16's 65504 ceiling. Computed naively in fp16
  that matrix came out with **834 non-finite entries of 4096**. Every occurrence is a
  *pairwise* ratio `exp(gc_r − gc_i)` with `i ≤ r`, and since `gc` only decreases, that
  exponent is never positive. Same numbers, no overflow, ordinary fp16 rounding error.
- **`kda/chunk.py` is written to be read, not raced.** At the kernel level it is ~82×
  slower than the fused Triton kernels (37 ms vs 0.46 ms at T=2048 on a 3090); inside a
  small training step, where projections dominate, the gap narrows to ~12×. That is why
  `kda/fast.py` exists, and why the two are pinned to each other by tests.

## Publishing the article

The writeup lives in a separate repository (the site), which serves its own copy of
`js/kda-math.js` and `js/widgets.js`. Those copies are what make the "the figures run
tested code" claim true, so they are checked rather than trusted:

```bash
python -m scripts.sync_site --check   # exits 1 if the deployed copy has drifted
python -m scripts.sync_site           # copy across, then rebuild the site
```

Run the check before publishing. A stale copy would leave the article's figures quietly
computing something no test has seen — the exact failure the fixture tests exist to catch.

## Licence

MIT.
