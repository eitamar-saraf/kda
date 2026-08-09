"""The task generators have to be right, or every curve downstream is meaningless.

A synthetic task with a subtly broken label is the classic way to publish a confident
wrong result: the model learns the leak, the curves look great, and nothing is being
measured. These tests check the answers are actually derivable from the context, the
loss mask lands on the answer positions, and no target is reachable without reading
the earlier part of the sequence.
"""

from __future__ import annotations

import pytest
import torch

from kda.model import LanguageModel, ModelConfig, layer_plan
from kda.recurrent import VARIANTS
from kda.tasks import mqar, palindrome, stack


def test_mqar_answers_are_recoverable_from_the_pairs():
    """Every queried key must appear earlier with exactly the value being asked for."""
    cfg = mqar.MQARConfig(seq_len=128, num_pairs=8, num_queries=4)
    tok, tgt, mask = mqar.make_batch(cfg, 16, torch.Generator().manual_seed(0))

    assert tok.shape == (16, 128)
    assert mask.sum(1).eq(4).all(), "expected exactly num_queries answer positions"

    q_start = 128 - 2 * cfg.num_queries
    for i in range(16):
        # pairs are scattered through the prefix, so find them by token range
        pairs = {}
        for p in range(q_start - 1):
            tokid = int(tok[i, p])
            if cfg.key0 <= tokid < cfg.val0:
                pairs[tokid] = int(tok[i, p + 1])
        assert len(pairs) == cfg.num_pairs, "keys must be distinct within a sequence"
        for pos in mask[i].nonzero().flatten():
            queried_key = int(tok[i, pos])
            answer = int(tgt[i, pos])
            assert queried_key in pairs, "query asks about a key that was never written"
            assert pairs[queried_key] == answer, "label disagrees with the written pair"


def test_mqar_answer_is_not_guessable_from_the_key_alone():
    """Across sequences the same key must take different values.

    Otherwise the model could memorise a fixed key->value table during training and
    never use the context at all -- and the task would measure nothing.
    """
    cfg = mqar.MQARConfig(seq_len=128, num_pairs=8, num_queries=4)
    g = torch.Generator().manual_seed(1)
    seen: dict[int, set[int]] = {}
    for _ in range(20):
        tok, tgt, mask = mqar.make_batch(cfg, 32, g)
        for i in range(32):
            for pos in mask[i].nonzero().flatten():
                seen.setdefault(int(tok[i, pos]), set()).add(int(tgt[i, pos]))
    multi = [k for k, v in seen.items() if len(v) > 1]
    assert len(multi) > 0.9 * len(seen), "keys should map to many values across batches"


def test_palindrome_targets_are_the_reversal():
    cfg = palindrome.PalindromeConfig(seq_len=65)
    tok, tgt, mask = palindrome.make_batch(cfg, 8, torch.Generator().manual_seed(0))
    n = cfg.n
    assert (tok[:, n] == cfg.SEP).all()
    torch.testing.assert_close(tok[:, n + 1 :], tok[:, :n].flip(-1))
    # the masked positions must predict the reversed sequence
    for i in range(8):
        pred_positions = mask[i].nonzero().flatten()
        got = tgt[i, pred_positions]
        torch.testing.assert_close(got, tok[i, :n].flip(-1)[: len(pred_positions)])


def test_stack_pops_match_a_simulated_lifo():
    """Re-simulate the stacks independently and check every labelled pop."""
    cfg = stack.StackConfig(seq_len=180, num_stacks=8, num_symbols=16)
    tok, tgt, mask = stack.make_batch(cfg, 16, torch.Generator().manual_seed(0))

    total_pops = 0
    for i in range(16):
        stacks: dict[int, list[int]] = {}
        for j in range(cfg.num_ops):
            base = 3 * j
            op, sid, sym = (int(tok[i, base + o]) for o in range(3))
            if op == cfg.PUSH:
                stacks.setdefault(sid, []).append(sym)
                assert not mask[i, base + 1]
            else:
                assert op == cfg.POP
                assert mask[i, base + 1], "pop position should be in the loss mask"
                expected = stacks[sid].pop()
                assert expected == sym, "sequence disagrees with LIFO order"
                assert int(tgt[i, base + 1]) == sym, "label is not the popped element"
                total_pops += 1
    assert total_pops > 0, "generated no pops at all"


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_model_trains_one_step_for_every_variant(variant):
    """Every architecture must build, run, and produce finite gradients."""
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=2, n_heads=2, variant=variant)
    model = LanguageModel(cfg)
    tok = torch.randint(0, 64, (2, 48))
    tgt = torch.randint(0, 64, (2, 48))
    loss, acc = model.loss(tok, tgt)
    loss.backward()

    assert torch.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients at all"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_layer_plan_matches_the_papers_3to1():
    assert layer_plan(4, 3) == ["linear", "linear", "linear", "full"]
    assert layer_plan(8, 3) == ["linear"] * 3 + ["full"] + ["linear"] * 3 + ["full"]
    assert layer_plan(4, 1) == ["linear", "full", "linear", "full"]
    assert layer_plan(4, 0) == ["full"] * 4
    assert layer_plan(4, None) == ["linear"] * 4


def test_hybrid_model_runs():
    cfg = ModelConfig(vocab_size=64, d_model=64, n_layers=4, n_heads=2, hybrid_ratio=3)
    model = LanguageModel(cfg)
    assert model.plan.count("full") == 1
    out = model(torch.randint(0, 64, (2, 32)))
    assert out.shape == (2, 32, 64)


def test_chunkwise_and_recurrent_layers_agree():
    """The layer wrapper must not change the maths when it switches kernels."""
    torch.manual_seed(0)
    cfg_kw = dict(vocab_size=32, d_model=64, n_layers=2, n_heads=2, variant="kda")
    a = LanguageModel(ModelConfig(**cfg_kw, chunkwise=True)).double()
    b = LanguageModel(ModelConfig(**cfg_kw, chunkwise=False)).double()
    b.load_state_dict(a.state_dict())

    tok = torch.randint(0, 32, (2, 40))
    torch.testing.assert_close(a(tok), b(tok), rtol=1e-8, atol=1e-9)


def test_decay_gate_init_keeps_memory_alive():
    """Regression: a zero bias on the decay projection makes the model untrainable.

    log a = -exp(A_log) * softplus(proj(x) + bias). With bias = 0 that is
    softplus(0) = 0.69, and with exp(A_log) up to 16 the decay reaches e^-11 per
    token -- the state is erased between writing a fact and reading it, and the loss
    goes NaN once the cumulative decay underflows. Mamba's dt initialisation is what
    keeps alpha near 1 at the start.

    Heads are deliberately given different timescales (``exp(A_log)`` spans 1..16), so
    the fastest-forgetting head sits well below the typical one. What must not happen
    is any channel decaying to nothing on the very first token.
    """
    from kda.layers import LinearAttentionLayer

    torch.manual_seed(0)
    layer = LinearAttentionLayer(64, 2, variant="kda")
    _, gates = layer(torch.randn(2, 64, 64), return_gates=True)
    alpha = gates["log_alpha"].exp()

    assert alpha.min() > 0.15, f"decay far too aggressive at init: min alpha {alpha.min():.3e}"
    assert alpha.median() > 0.8, f"typical memory too short: median alpha {alpha.median():.3f}"
    assert alpha.max() < 1.0, "gate should still be able to forget something"

    # and the cumulative decay over a long sequence must not underflow
    gc = gates["log_alpha"].cumsum(-2)
    assert torch.isfinite(gc.exp()).all()


def test_model_wide_init_does_not_clobber_the_decay_bias():
    """LanguageModel.apply(init) must leave the dt bias alone."""
    torch.manual_seed(0)
    model = LanguageModel(ModelConfig(vocab_size=32, d_model=64, n_layers=2,
                                      n_heads=2, variant="kda"))
    bias = model.blocks[0].mixer.a_up.bias
    assert not torch.allclose(bias, torch.zeros_like(bias)), "dt bias was zeroed"


def test_decay_gram_masked_region_is_zero_not_nan():
    """Regression: exp() of the upper-triangle exponent overflows, and inf * 0 = NaN."""
    from kda.chunk import decay_gram

    c, d = 32, 8
    torch.manual_seed(0)
    x = torch.randn(1, c, d, dtype=torch.float64)
    y = torch.randn(1, c, d, dtype=torch.float64)
    # a strong decay, so gc_r - gc_i for i > r is hugely positive
    gc = (-torch.rand(1, c, d, dtype=torch.float64) * 20).cumsum(dim=-2)
    assert gc.min() < -300, "test needs an exponent large enough to overflow"

    out = decay_gram(x, y, gc, stable=True)
    assert torch.isfinite(out).all(), "decay_gram produced non-finite entries"
    upper = torch.triu(torch.ones(c, c, dtype=torch.bool), diagonal=1)
    assert (out[0][upper] == 0).all(), "masked region should be exactly zero"
