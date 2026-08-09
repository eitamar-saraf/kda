"""Multi-Query Associative Recall.

From the paper's Table 5.1::

    Input   A 1 C 3 B 0 M 8 G 5 E 4 <SEP> B G
    Output  . . . . . . . . . . . .  .    0 5

Read a stream of key-value pairs, then answer several queries about them. Zhang et al.
found MQAR accuracy tracks language-modelling quality closely, which is why it is the
first thing anyone runs on a new recurrent architecture.

It is also the cleanest possible probe of the thing linear attention is bad at. A
fixed-size state has to hold every pair at once, and the delta rule decides whether a
second write to a key *replaces* the first or piles on top of it. We put a long stretch
of distractor tokens between the pairs and the queries, so a model that cannot choose
what to keep has to keep everything -- and interferes with itself.

Vocabulary layout (contiguous, so a model sees no structure for free):
    0            PAD / filler answer slot
    1            SEP
    2 .. 2+nk    keys
    then         values
    then         noise symbols used as distractors
"""

from __future__ import annotations

import torch

__all__ = ["MQARConfig", "make_batch", "vocab_size"]


class MQARConfig:
    def __init__(
        self,
        seq_len: int = 256,
        num_pairs: int = 8,
        num_queries: int = 4,
        num_keys: int = 32,
        num_values: int = 32,
        num_noise: int = 16,
    ):
        if num_queries > num_pairs:
            raise ValueError("cannot query more pairs than were written")
        # pairs (2 each) + SEP + query/answer (2 each) must fit, with room for noise
        if 2 * num_pairs + 1 + 2 * num_queries > seq_len:
            raise ValueError(f"seq_len {seq_len} too short for this config")
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_noise = num_noise

    # --- token id layout ---
    PAD, SEP = 0, 1

    @property
    def key0(self) -> int:
        return 2

    @property
    def val0(self) -> int:
        return self.key0 + self.num_keys

    @property
    def noise0(self) -> int:
        return self.val0 + self.num_values

    @property
    def vocab_size(self) -> int:
        return self.noise0 + self.num_noise


def vocab_size(cfg: MQARConfig) -> int:
    return cfg.vocab_size


def make_batch(cfg: MQARConfig, batch_size: int, generator: torch.Generator | None = None):
    """Build one batch.

    Returns:
        tokens: ``(B, T)`` int64 input sequence.
        targets: ``(B, T)`` int64, next-token targets.
        mask: ``(B, T)`` bool, True only at the answer positions. Loss is taken there
            and nowhere else -- predicting the distractors is not the task, and
            including them would let a model score well by learning the noise
            distribution.
    """
    g = generator
    b, t = batch_size, cfg.seq_len
    n_pairs, n_q = cfg.num_pairs, cfg.num_queries

    # distinct keys per sequence, so a key never has two different values
    keys = torch.stack([
        torch.randperm(cfg.num_keys, generator=g)[:n_pairs] for _ in range(b)
    ]) + cfg.key0
    vals = torch.randint(0, cfg.num_values, (b, n_pairs), generator=g) + cfg.val0

    # which of the written pairs get queried, and in what order
    qidx = torch.stack([
        torch.randperm(n_pairs, generator=g)[:n_q] for _ in range(b)
    ])
    qkeys = keys.gather(1, qidx)
    qvals = vals.gather(1, qidx)

    tokens = torch.full((b, t), cfg.PAD, dtype=torch.long)
    mask = torch.zeros(b, t, dtype=torch.bool)

    # Layout: pairs are *scattered* through the prefix among noise, then SEP, then the
    # queries. The obvious alternative -- all pairs at the front, one long noise span,
    # queries at the end -- is unlearnable for a decaying state, and instructively so:
    # every pair then sits the same ~T tokens from its query, so the state has decayed
    # by alpha^T (about 1e-4 at initialisation) before any query arrives. The loss sits
    # at exactly ln(num_values) forever because there is no gradient to tell the gate
    # to decay more slowly. Scattering spreads the retention distances from near-zero
    # to near-T, so short-range pairs give a learning signal that the long-range ones
    # can then build on. This also matches the paper's own example, which is dense with
    # pairs rather than dominated by filler.
    q_start = t - 2 * n_q
    prefix = q_start - 1                      # room before the SEP
    n_slots = prefix // 2

    # noise everywhere in the prefix first, then overwrite chosen slots with pairs
    tokens[:, :prefix] = (
        torch.randint(0, cfg.num_noise, (b, prefix), generator=g) + cfg.noise0
    )
    slot_choice = torch.stack([
        torch.randperm(n_slots, generator=g)[:n_pairs].sort().values for _ in range(b)
    ])                                        # (b, n_pairs) sorted slot indices
    pos = slot_choice * 2
    tokens.scatter_(1, pos, keys)
    tokens.scatter_(1, pos + 1, vals)

    query_block = torch.stack([qkeys, qvals], dim=-1).reshape(b, 2 * n_q)
    tokens[:, q_start - 1] = cfg.SEP
    tokens[:, q_start:] = query_block

    # Next-token prediction: position i predicts token i+1. The answer to query j
    # sits at q_start + 2j + 1, so it is predicted from position q_start + 2j.
    targets = torch.full((b, t), cfg.PAD, dtype=torch.long)
    targets[:, :-1] = tokens[:, 1:]
    for j in range(n_q):
        mask[:, q_start + 2 * j] = True

    return tokens, targets, mask
