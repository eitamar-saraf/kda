/**
 * The browser maths must equal the Python maths.
 *
 * Fixtures come from `scripts/export_widget_data.py`, which calls the same
 * `kda.recurrent.linear_attn` that the Python suite pins to the paper's equations and
 * to the Kimi team's released kernel. So a pass here chains all the way back:
 *
 *     paper equations  ==  kda.recurrent  ==  kda.chunk  ==  fla kernels
 *                                          \
 *                                           ==  js/kda-math.js  (this file)
 *
 * Run with:  node --test js/
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  linearAttn,
  interference,
  cacheBytes,
  layerPlan,
  flops,
  normalize,
  dot,
} from "./kda-math.js";

const here = dirname(fileURLToPath(import.meta.url));
const { cases } = JSON.parse(readFileSync(join(here, "fixtures.json"), "utf8"));

const TOL = 1e-9;

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[i].length; j++) m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  return m;
}

for (const c of cases) {
  test(`matches the Python reference: ${c.name}`, () => {
    const { outputs } = linearAttn(c.q, c.k, c.v, {
      alpha: c.alpha ?? null,
      beta: c.beta,
      gate: c.gate,
      delta: c.delta,
    });
    assert.equal(outputs.length, c.outputs.length);
    const err = maxAbsDiff(outputs, c.outputs);
    assert.ok(err < TOL, `${c.name}: max abs difference ${err} exceeds ${TOL}`);
  });
}

test("the delta rule overwrites a repeated key instead of accumulating", () => {
  // Same behavioural claim the Python suite makes, so the widget cannot drift from it.
  const k = normalize([1, 0.3, -0.2, 0.5]);
  const keys = [k, k];
  const v = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
  ];
  const withDelta = linearAttn(keys, keys, v, { beta: [1, 1], gate: "none", delta: true });
  const without = linearAttn(keys, keys, v, { beta: [1, 1], gate: "none", delta: false });

  for (let d = 0; d < 4; d++) {
    assert.ok(Math.abs(withDelta.outputs[1][d] - v[1][d]) < 1e-9);
    assert.ok(Math.abs(without.outputs[1][d] - (v[0][d] + v[1][d])) < 1e-9);
  }
});

test("interference vanishes exactly when keys are orthogonal", () => {
  const orth = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ];
  const vals = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];
  const clean = interference(orth, vals, 1);
  assert.ok(Math.hypot(...clean.noise) < 1e-12);
  assert.deepEqual(clean.signal, [3, 4]);

  // and does not when they are not
  const tilted = [orth[0], normalize([0.6, 0.8, 0, 0]), orth[2]];
  const dirty = interference(tilted, vals, 1);
  assert.ok(Math.hypot(...dirty.noise) > 0.1);
});

test("interference decomposition sums to the full retrieval", () => {
  const keys = [normalize([1, 0.2, 0]), normalize([0.4, 1, 0.1]), normalize([0, 0.3, 1])];
  const vals = [
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  const r = interference(keys, vals, 0);

  // signal + interference must reconstruct the full retrieval...
  for (let d = 0; d < 2; d++) {
    assert.ok(Math.abs(r.total[d] - (r.signal[d] + r.noise[d])) < 1e-12);
  }

  // ...and that retrieval must be what the recurrence actually produces. Write all
  // three pairs with plain linear attention, then query with k_0.
  const written = linearAttn(keys, keys, vals, { gate: "none", delta: false });
  const queried = linearAttn(
    [...keys, keys[0]], [...keys, keys[0]], [...vals, [0, 0]],
    { gate: "none", delta: false }
  );
  for (let d = 0; d < 2; d++) {
    assert.ok(
      Math.abs(r.total[d] - queried.outputs[3][d]) < 1e-12,
      `decomposition disagrees with the recurrence at dim ${d}`
    );
  }
  assert.equal(written.outputs.length, 3);
});

test("layerPlan reproduces the paper's 3:1 hybrid", () => {
  assert.deepEqual(layerPlan(4, 3), ["linear", "linear", "linear", "full"]);
  assert.deepEqual(layerPlan(4, 0), ["full", "full", "full", "full"]);
  assert.deepEqual(layerPlan(3, null), ["linear", "linear", "linear"]);
});

test("cache: attention grows with context, KDA does not", () => {
  const small = cacheBytes({ seqLen: 1024, layers: 8, heads: 8, headDim: 128, kdaPerAttn: 3 });
  const big = cacheBytes({ seqLen: 1_000_000, layers: 8, heads: 8, headDim: 128, kdaPerAttn: 3 });
  assert.equal(small.kda, big.kda, "the recurrent state must not depend on context length");
  assert.ok(big.attention > small.attention * 900);

  // the paper's headline: a 3:1 hybrid keeps a quarter of the KV cache
  const full = cacheBytes({ seqLen: 1_000_000, layers: 8, heads: 8, headDim: 128, kdaPerAttn: 0 });
  const ratio = big.attention / full.attention;
  assert.ok(Math.abs(ratio - 0.25) < 1e-9, `expected 25% of the KV cache, got ${ratio}`);
});

test("FLOPs: KDA is linear in T, attention quadratic", () => {
  const a = flops(4096, 128);
  const b = flops(8192, 128);
  assert.ok(Math.abs(b.kda / a.kda - 2) < 1e-9, "KDA should double when T doubles");
  assert.ok(Math.abs(b.attention / a.attention - 4) < 1e-9, "attention should quadruple");
});

test("dot and normalize behave", () => {
  assert.ok(Math.abs(dot([1, 2, 3], [4, 5, 6]) - 32) < 1e-12);
  assert.ok(Math.abs(Math.hypot(...normalize([3, 4])) - 1) < 1e-12);
});
