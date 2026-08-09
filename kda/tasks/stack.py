"""Interleaved LIFO stacks -- a state-tracking probe.

The paper's setup: 64 independent stacks, each with an id, and a stream of operations::

    <PUSH> 1 G    push G onto stack 1
    <POP>  0 E    the element popped from stack 0 must be E

The model has to predict the popped element, which means maintaining 64 separate LIFO
structures inside one fixed-size state. This is the task from the state-tracking
literature (Deletang et al.) that separates architectures by *what they can represent*
rather than how much they can memorise.

It is the sharpest test of fine-grained gating in the whole set. A pop from stack 3
should disturb stack 3's memory and nothing else. With a single scalar decay per head,
every stack ages together on every operation; with a per-channel decay the model can
route different stacks to different channels and retire them independently. That is
precisely the claim KDA makes, so this is where we should expect the largest gap --
and where a null result would be the most informative.
"""

from __future__ import annotations

import torch

__all__ = ["StackConfig", "make_batch"]


class StackConfig:
    PAD, PUSH, POP = 0, 1, 2

    def __init__(
        self,
        seq_len: int = 256,
        num_stacks: int = 64,
        num_symbols: int = 32,
        push_prob: float = 0.6,
    ):
        self.num_ops = seq_len // 3          # each op is 3 tokens
        if self.num_ops < 2:
            raise ValueError(f"seq_len {seq_len} too short")
        self.seq_len = 3 * self.num_ops
        self.num_stacks = num_stacks
        self.num_symbols = num_symbols
        self.push_prob = push_prob

    @property
    def sid0(self) -> int:
        return 3

    @property
    def sym0(self) -> int:
        return self.sid0 + self.num_stacks

    @property
    def vocab_size(self) -> int:
        return self.sym0 + self.num_symbols


def make_batch(cfg: StackConfig, batch_size: int, generator: torch.Generator | None = None):
    """Simulate the stacks to build ground truth.

    A POP is only emitted for a stack that is non-empty; otherwise the operation is
    resampled as a PUSH. So every POP in the sequence has a well-defined answer, and
    the loss mask marks exactly those positions.
    """
    b, n_ops = batch_size, cfg.num_ops
    tokens = torch.full((b, cfg.seq_len), cfg.PAD, dtype=torch.long)
    mask = torch.zeros(b, cfg.seq_len, dtype=torch.bool)

    rand = torch.rand(b, n_ops, generator=generator)
    sids = torch.randint(0, cfg.num_stacks, (b, n_ops), generator=generator)
    syms = torch.randint(0, cfg.num_symbols, (b, n_ops), generator=generator)

    for i in range(b):
        stacks: list[list[int]] = [[] for _ in range(cfg.num_stacks)]
        for j in range(n_ops):
            sid = int(sids[i, j])
            do_pop = rand[i, j] >= cfg.push_prob and len(stacks[sid]) > 0
            base = 3 * j
            if do_pop:
                sym = stacks[sid].pop()
                tokens[i, base] = cfg.POP
                tokens[i, base + 1] = cfg.sid0 + sid
                tokens[i, base + 2] = sym
                # position base+1 (the stack id) predicts the popped symbol
                mask[i, base + 1] = True
            else:
                sym = cfg.sym0 + int(syms[i, j])
                stacks[sid].append(sym)
                tokens[i, base] = cfg.PUSH
                tokens[i, base + 1] = cfg.sid0 + sid
                tokens[i, base + 2] = sym

    targets = torch.full_like(tokens, cfg.PAD)
    targets[:, :-1] = tokens[:, 1:]
    return tokens, targets, mask
