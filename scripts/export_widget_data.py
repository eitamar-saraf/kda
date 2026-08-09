"""Emit JSON fixtures so the browser-side maths can be checked against Python.

The interactive figures recompute the recurrence live in JavaScript. This writes the
inputs and the reference outputs from :func:`kda.recurrent.linear_attn` -- the same
function the test suite pins to the paper's equations and to the official kernel -- so
``js/kda-math.test.js`` can assert the two agree.

Usage::

    python -m scripts.export_widget_data --out js/fixtures.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kda.recurrent import VARIANTS, linear_attn


def case(name: str, t: int, dk: int, dv: int, gate: str, delta: bool, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, t, dk, generator=g, dtype=torch.float64)
    k = torch.nn.functional.normalize(
        torch.randn(1, 1, t, dk, generator=g, dtype=torch.float64), dim=-1
    )
    v = torch.randn(1, 1, t, dv, generator=g, dtype=torch.float64)
    beta = torch.rand(1, 1, t, generator=g, dtype=torch.float64)

    alpha = None
    if gate == "scalar":
        alpha = torch.rand(1, 1, t, generator=g, dtype=torch.float64) * 0.4 + 0.6
    elif gate == "channel":
        alpha = torch.rand(1, 1, t, dk, generator=g, dtype=torch.float64) * 0.4 + 0.6

    out, state = linear_attn(q, k, v, alpha, beta, gate=gate, delta=delta,
                             return_state=True)

    payload = {
        "name": name,
        "gate": gate,
        "delta": delta,
        "q": q[0, 0].tolist(),
        "k": k[0, 0].tolist(),
        "v": v[0, 0].tolist(),
        "beta": beta[0, 0].tolist(),
        "outputs": out[0, 0].tolist(),
        "final_state": state[0].tolist()[0] if state.shape[0] == 1 else None,
    }
    if alpha is not None:
        payload["alpha"] = alpha[0, 0].tolist()
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="js/fixtures.json")
    a = p.parse_args()

    cases = []
    for i, (name, spec) in enumerate(sorted(VARIANTS.items())):
        cases.append(case(name, t=16, dk=6, dv=4, gate=spec["gate"],
                          delta=spec["delta"], seed=i))
    # a longer one, to catch anything that only shows up after many steps
    cases.append(case("kda_long", t=64, dk=8, dv=8, gate="channel", delta=True, seed=99))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, indent=1))
    print(f"wrote {out}  ({len(cases)} cases)")


if __name__ == "__main__":
    main()
