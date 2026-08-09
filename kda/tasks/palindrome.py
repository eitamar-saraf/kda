"""Palindrome: reproduce a random sequence in reverse.

From the paper's Table 5.1::

    Input   O G R S U N E <SEP> E N U S R G O
    Output  . . . . . . .  .    N U S R G O .

Copying is the classic hard case for linear attention: to emit the reversal you must
hold the *entire* prefix in a fixed-size state, with no option to compress. Jelassi et
al. showed exactly this separation between transformers and state-space models, and it
is why the paper reports palindrome accuracy falling off as sequence length grows even
for the best recurrent models.

Unlike MQAR there is nothing to forget here -- every token matters -- so this task
mostly measures raw state capacity, and is the one where we should expect KDA's
advantage over Gated DeltaNet to be smallest.
"""

from __future__ import annotations

import torch

__all__ = ["PalindromeConfig", "make_batch"]


class PalindromeConfig:
    PAD, SEP = 0, 1

    def __init__(self, seq_len: int = 256, num_symbols: int = 32):
        # sequence is [n symbols][SEP][n symbols], so seq_len must be odd-ish
        self.n = (seq_len - 1) // 2
        if self.n < 1:
            raise ValueError(f"seq_len {seq_len} too short")
        self.seq_len = 2 * self.n + 1
        self.num_symbols = num_symbols

    @property
    def sym0(self) -> int:
        return 2

    @property
    def vocab_size(self) -> int:
        return self.sym0 + self.num_symbols


def make_batch(cfg: PalindromeConfig, batch_size: int, generator: torch.Generator | None = None):
    """Returns ``(tokens, targets, mask)``; loss is taken only on the reversed half."""
    b, n = batch_size, cfg.n
    syms = torch.randint(0, cfg.num_symbols, (b, n), generator=generator) + cfg.sym0

    tokens = torch.empty(b, cfg.seq_len, dtype=torch.long)
    tokens[:, :n] = syms
    tokens[:, n] = cfg.SEP
    tokens[:, n + 1 :] = syms.flip(-1)

    targets = torch.full_like(tokens, cfg.PAD)
    targets[:, :-1] = tokens[:, 1:]

    # Positions n .. 2n-1 predict the reversed sequence (position n, holding SEP,
    # predicts the first reversed symbol).
    mask = torch.zeros(b, cfg.seq_len, dtype=torch.bool)
    mask[:, n : 2 * n] = True
    return tokens, targets, mask
