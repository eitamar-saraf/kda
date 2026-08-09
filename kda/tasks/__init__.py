"""Synthetic probes from Sec. 5.1 of the paper.

Each module exposes a ``Config`` and a ``make_batch(cfg, batch_size, generator)``
returning ``(tokens, targets, mask)``, where ``mask`` marks the positions the loss is
taken at. Restricting the loss to answer positions matters: on MQAR roughly 95% of the
sequence is distractor tokens, and a model that predicts those well while getting every
answer wrong would otherwise look like it was learning.
"""

from __future__ import annotations

from kda.tasks import mqar, palindrome, stack

TASKS = {
    "mqar": (mqar.MQARConfig, mqar.make_batch),
    "palindrome": (palindrome.PalindromeConfig, palindrome.make_batch),
    "stack": (stack.StackConfig, stack.make_batch),
}

__all__ = ["TASKS", "mqar", "palindrome", "stack", "build"]


def build(name: str, **kw):
    """Return ``(cfg, make_batch)`` for a named task."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; have {sorted(TASKS)}")
    cfg_cls, fn = TASKS[name]
    return cfg_cls(**kw), fn
