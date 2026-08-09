/**
 * Interactive figures for the Kimi Linear writeup.
 *
 * Every widget here computes the real recurrence through `kda-math.js`, which is
 * checked against the Python reference in `kda-math.test.js`. Nothing is a recording.
 *
 * Each widget is `mount<Name>(el)` and reads its options from data- attributes, so the
 * .astro page stays prose plus mount points.
 */

import {
  linearAttn, interference, cumulativeDecay, cacheBytes, layerPlan,
  flops, mulberry32, randomUnit, normalize, dot,
} from "./kda-math.js";

// ---------------------------------------------------------------- design tokens
// Categorical slots 1-3 of the validated palette (all-pairs clean: worst CVD dE 9.2,
// worst normal-vision dE 24.0). Aqua sits below 3:1 on a light surface, so every
// series is direct-labelled and the result charts also ship a table.
const C = {
  s1: "#2a78d6",   // KDA / "with the delta rule" / the good case
  s2: "#eb6834",   // Gated DeltaNet / "without" / the problem
  s3: "#1baf7a",   // third series
  ref: "#6b7280",  // reference lines: softmax attention, ideal, etc. Never a series.
  ink: "#111827",
  ink2: "#4b5563",
  muted: "#9ca3af",
  grid: "#e5e7eb",
  surface: "#ffffff",
  warm: "#fef3c7",
  cool: "#dbeafe",
};

const NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, parent = null) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(el);
  return el;
}

function el(tag, cls, parent, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  if (parent) parent.appendChild(n);
  return n;
}

/** A labelled range input that calls back on input. Returns a read function. */
function slider(parent, { label, min, max, step, value, format = (v) => v }, onChange) {
  const wrap = el("label", "block text-sm", parent);
  const head = el("div", "flex justify-between items-baseline gap-3", wrap);
  el("span", "text-gray-600", head, label);
  const out = el("span", "font-mono text-gray-900 tabular-nums", head, format(value));
  const input = el("input", "w-full accent-blue-600", wrap);
  Object.assign(input, { type: "range", min, max, step, value });
  input.addEventListener("input", () => {
    out.textContent = format(Number(input.value));
    onChange(Number(input.value));
  });
  return () => Number(input.value);
}

function fmtBytes(b) {
  if (b >= 2 ** 30) return (b / 2 ** 30).toFixed(2) + " GB";
  if (b >= 2 ** 20) return (b / 2 ** 20).toFixed(1) + " MB";
  if (b >= 2 ** 10) return (b / 2 ** 10).toFixed(1) + " KB";
  return b + " B";
}

function fmtNum(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(1) + "T";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "G";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}

/** Blue-white-red cell colour for a signed matrix entry. */
function cellFill(x, scale) {
  const t = Math.max(-1, Math.min(1, x / (scale || 1)));
  if (t >= 0) {
    const a = t;
    return `rgb(${Math.round(255 - 213 * a)},${Math.round(255 - 135 * a)},${Math.round(255 - 41 * a)})`;
  }
  const a = -t;
  return `rgb(${Math.round(255 - 20 * a)},${Math.round(255 - 151 * a)},${Math.round(255 - 203 * a)})`;
}

/** Draw a [rows][cols] matrix as a heatmap grid. */
function drawMatrix(parent, M, { x, y, cell, scale, gap = 1, title = null }) {
  const g = svg("g", {}, parent);
  const rows = M.length, cols = M[0].length;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      svg("rect", {
        x: x + j * (cell + gap), y: y + i * (cell + gap),
        width: cell, height: cell, rx: 1.5,
        fill: cellFill(M[i][j], scale), stroke: C.grid, "stroke-width": 0.5,
      }, g);
    }
  }
  if (title) {
    svg("text", {
      x: x + (cols * (cell + gap)) / 2, y: y - 6, "text-anchor": "middle",
      "font-size": 11, fill: C.ink2, "font-family": "ui-sans-serif,system-ui",
    }, g).textContent = title;
  }
  return { g, w: cols * (cell + gap), h: rows * (cell + gap) };
}

function panel(root, title, blurb) {
  root.innerHTML = "";
  const box = el("div", "not-prose rounded-xl border border-gray-200 bg-white p-4 md:p-5 my-6 shadow-sm", root);
  if (title) el("h4", "text-base font-semibold text-gray-900 mb-1", box, title);
  if (blurb) el("p", "text-sm text-gray-600 mb-4", box, blurb);
  return box;
}

// =============================================================== 1. the KV cache
export function mountCacheGrowth(root) {
  const box = panel(root, "What has to be kept to write the next token",
    "Attention stores a key and a value for every token it has seen. A recurrent state stores one fixed-size matrix per head, whatever the context length.");

  const cfg = { layers: 32, heads: 32, headDim: 128 };
  const controls = el("div", "grid sm:grid-cols-2 gap-4 mb-4", box);
  const chart = svg("svg", { viewBox: "0 0 640 190", class: "w-full", role: "img" }, box);
  const caption = el("p", "text-sm text-gray-600 mt-2", box);

  let logLen = 12;
  const readLen = slider(controls, {
    label: "Context length", min: 10, max: 20, step: 1, value: logLen,
    format: (v) => fmtNum(2 ** v) + " tokens",
  }, (v) => { logLen = v; draw(); });

  function draw() {
    const seqLen = 2 ** readLen();
    chart.innerHTML = "";
    const full = cacheBytes({ seqLen, layers: cfg.layers, ...cfg, kdaPerAttn: 0 });
    const hybrid = cacheBytes({ seqLen, layers: cfg.layers, ...cfg, kdaPerAttn: 3 });

    const rows = [
      { label: "Full attention", bytes: full.attention, color: C.s2 },
      { label: "3:1 hybrid (KDA + attention)", bytes: hybrid.attention + hybrid.kda, color: C.s1 },
      { label: "Pure KDA state", bytes: cacheBytes({ seqLen, layers: cfg.layers, ...cfg }).kda, color: C.s3 },
    ];
    const max = Math.max(...rows.map((r) => r.bytes));
    rows.forEach((r, i) => {
      const y = 30 + i * 52;
      svg("text", { x: 0, y: y - 6, "font-size": 12, fill: C.ink2 }, chart).textContent = r.label;
      svg("rect", { x: 0, y, width: 470, height: 20, rx: 4, fill: "#f3f4f6" }, chart);
      const w = Math.max(2, (r.bytes / max) * 470);
      svg("rect", { x: 0, y, width: w, height: 20, rx: 4, fill: r.color }, chart);
      svg("text", {
        x: 480, y: y + 15, "font-size": 13, fill: C.ink, "font-family": "ui-monospace,monospace",
      }, chart).textContent = fmtBytes(r.bytes);
    });

    const saving = 1 - (hybrid.attention + hybrid.kda) / full.attention;
    caption.textContent =
      `At ${fmtNum(seqLen)} tokens, a ${cfg.layers}-layer model with ${cfg.heads} heads of ` +
      `dimension ${cfg.headDim} keeps ${fmtBytes(full.attention)} of KV cache with full attention. ` +
      `Replacing three layers in four with KDA cuts that by ${(saving * 100).toFixed(0)}% — ` +
      `and the KDA part does not grow at all as the context gets longer.`;
  }
  draw();
}

// ================================================== 2. the associativity swap
export function mountAssociativity(root) {
  const box = panel(root, "The trick: move the parentheses",
    "Attention without the softmax is three matrices multiplied together. Multiplying left-to-right builds a T×T map; right-to-left builds a d×d state. Same answer, different cost.");

  const chart = svg("svg", { viewBox: "0 0 640 250", class: "w-full", role: "img" }, box);
  const controls = el("div", "mt-3", box);
  const caption = el("p", "text-sm text-gray-600 mt-3", box);

  let logT = 12;
  const readT = slider(controls, {
    label: "Sequence length T (head dimension fixed at 128)",
    min: 6, max: 20, step: 1, value: logT, format: (v) => fmtNum(2 ** v) + " tokens",
  }, () => draw());

  function draw() {
    const T = 2 ** readT(), d = 128;
    chart.innerHTML = "";

    // schematic: (Q K^T) V   vs   Q (K^T V)
    const rng = mulberry32(7);
    const n = 8;
    const Q = Array.from({ length: n }, () => Array.from({ length: 6 }, () => rng() * 2 - 1));
    const K = Array.from({ length: n }, () => Array.from({ length: 6 }, () => rng() * 2 - 1));
    const V = Array.from({ length: n }, () => Array.from({ length: 6 }, () => rng() * 2 - 1));
    const QK = Q.map((qr) => K.map((kr) => dot(qr, kr)));
    const KV = Array.from({ length: 6 }, (_, a) =>
      Array.from({ length: 6 }, (_, b) => K.reduce((s, kr, t) => s + kr[a] * V[t][b], 0)));

    svg("text", { x: 0, y: 14, "font-size": 12, fill: C.s2, "font-weight": 600 }, chart)
      .textContent = "left to right:  (Q Kᵀ) V   — builds a T×T matrix";
    drawMatrix(chart, QK, { x: 10, y: 26, cell: 9, scale: 3, title: null });
    svg("text", { x: 110, y: 70, "font-size": 20, fill: C.muted }, chart).textContent = "×";
    drawMatrix(chart, V, { x: 135, y: 26, cell: 9, scale: 2 });
    svg("text", { x: 10, y: 130, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = `T×T = ${fmtNum(T)}² entries`;

    svg("text", { x: 330, y: 14, "font-size": 12, fill: C.s1, "font-weight": 600 }, chart)
      .textContent = "right to left:  Q (Kᵀ V)   — builds a d×d state";
    drawMatrix(chart, Q, { x: 340, y: 26, cell: 9, scale: 2 });
    svg("text", { x: 415, y: 70, "font-size": 20, fill: C.muted }, chart).textContent = "×";
    drawMatrix(chart, KV, { x: 440, y: 26, cell: 9, scale: 8 });
    svg("text", { x: 340, y: 130, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = `d×d = ${d}² entries, whatever T is`;

    // cost bars
    const f = flops(T, d);
    const max = Math.max(f.kda, f.attention);
    [["Attention  2T²d", f.attention, C.s2], ["Linear  ~6Td²", f.kda, C.s1]].forEach(([label, val, col], i) => {
      const y = 165 + i * 38;
      svg("text", { x: 0, y: y - 4, "font-size": 12, fill: C.ink2 }, chart).textContent = label;
      svg("rect", { x: 0, y, width: 470, height: 18, rx: 4, fill: "#f3f4f6" }, chart);
      svg("rect", { x: 0, y, width: Math.max(2, (val / max) * 470), height: 18, rx: 4, fill: col }, chart);
      svg("text", { x: 480, y: y + 14, "font-size": 13, fill: C.ink, "font-family": "ui-monospace,monospace" }, chart)
        .textContent = fmtNum(val) + " FLOPs";
    });

    const ratio = f.attention / f.kda;
    caption.textContent = ratio >= 1
      ? `At ${fmtNum(T)} tokens the quadratic form costs ${ratio.toFixed(1)}× more. `
        + `The crossover is around T ≈ 3d, so for short sequences attention is actually the cheaper one — linear attention is not a free lunch, it is a different asymptote.`
      : `At ${fmtNum(T)} tokens attention is still ${(1 / ratio).toFixed(1)}× cheaper. `
        + `Below the crossover (T ≈ 3d) the d×d state is bigger than the T×T map, and quadratic attention wins.`;
  }
  draw();
}

// ============================================ 3. interference and the delta rule
export function mountInterference(root) {
  const box = panel(root, "Why a fixed-size memory goes wrong, and what fixes it",
    "Write key–value pairs into one matrix, then read one back. The retrieval splits into the value you wanted plus everything else leaking through.");

  const controls = el("div", "grid sm:grid-cols-2 gap-4 mb-4", box);
  const chart = svg("svg", { viewBox: "0 0 640 240", class: "w-full", role: "img" }, box);
  const caption = el("p", "text-sm text-gray-600 mt-2", box);

  const dk = 8, dv = 6;
  let nPairs = 5, useDelta = false;

  const readN = slider(controls, {
    label: "Key–value pairs written", min: 2, max: 16, step: 1, value: nPairs,
    format: (v) => `${v} of ${dk} dims`,
  }, () => draw());

  const toggleWrap = el("label", "flex items-center gap-2 text-sm self-end", controls);
  const toggle = el("input", "accent-blue-600 w-4 h-4", toggleWrap);
  toggle.type = "checkbox";
  el("span", "text-gray-700", toggleWrap, "Use the delta rule");
  toggle.addEventListener("change", () => { useDelta = toggle.checked; draw(); });

  function draw() {
    nPairs = readN();
    chart.innerHTML = "";
    const rng = mulberry32(3);
    const keys = Array.from({ length: nPairs }, () => randomUnit(rng, dk));
    const values = Array.from({ length: nPairs }, (_, i) =>
      Array.from({ length: dv }, (_, j) => (j === i % dv ? 1 : 0)));

    const probe = 0;
    const beta = new Array(nPairs).fill(1);
    const res = linearAttn([...keys, keys[probe]], [...keys, keys[probe]],
      [...values, new Array(dv).fill(0)],
      { beta: [...beta, 0], gate: "none", delta: useDelta });
    const got = res.outputs[nPairs];
    const want = values[probe];

    const decomp = interference(keys, values, probe);
    const noiseNorm = Math.hypot(...decomp.noise);
    const sigNorm = Math.hypot(...decomp.signal);

    // bars: wanted vs retrieved
    const bw = 34, x0 = 20;
    svg("text", { x: x0, y: 18, "font-size": 12, fill: C.ink2 }, chart)
      .textContent = "what we stored for key #1";
    want.forEach((val, j) => {
      svg("rect", { x: x0 + j * (bw + 6), y: 30 + (1 - val) * 60, width: bw,
        height: Math.max(1, val * 60), rx: 4, fill: C.ref }, chart);
    });

    svg("text", { x: x0, y: 128, "font-size": 12, fill: C.ink2 }, chart)
      .textContent = "what we get back";
    const scale = Math.max(1, ...got.map(Math.abs));
    got.forEach((val, j) => {
      const h = Math.max(1, (Math.abs(val) / scale) * 60);
      svg("rect", { x: x0 + j * (bw + 6), y: val >= 0 ? 200 - h : 200, width: bw,
        height: h, rx: 4, fill: useDelta ? C.s1 : C.s2 }, chart);
    });
    svg("line", { x1: x0 - 6, y1: 200, x2: x0 + dv * (bw + 6), y2: 200,
      stroke: C.grid, "stroke-width": 2 }, chart);

    // numbers
    const err = Math.hypot(...got.map((g, j) => g - want[j]));
    const tx = 330;
    const lines = useDelta
      ? [["retrieval error", err.toFixed(3)],
         ["", ""],
         ["The delta rule subtracts what the", ""],
         ["memory already answers before it", ""],
         ["writes, so each key is corrected", ""],
         ["rather than piled on.", ""]]
      : [["signal ‖w‖", sigNorm.toFixed(3)],
         ["interference ‖noise‖", noiseNorm.toFixed(3)],
         ["retrieval error", err.toFixed(3)],
         ["", ""],
         [`${nPairs} keys in ${dk} dimensions:`, ""],
         [nPairs > dk
            ? "more keys than dimensions, so no"
            : "random keys are never exactly", ""],
         [nPairs > dk
            ? "arrangement can be orthogonal."
            : "orthogonal, so some leaks through.", ""]];
    lines.forEach(([a, b], i) => {
      const y = 30 + i * 20;
      svg("text", { x: tx, y, "font-size": 12, fill: C.ink2 }, chart).textContent = a;
      if (b) svg("text", { x: 620, y, "font-size": 13, "text-anchor": "end",
        fill: C.ink, "font-family": "ui-monospace,monospace" }, chart).textContent = b;
    });

    caption.textContent = useDelta
      ? `With the delta rule the memory reproduces the stored value almost exactly (error ${err.toFixed(3)}), even with ${nPairs} pairs in an ${dk}-dimensional state. This is the whole reason DeltaNet exists.`
      : `Every other key that is not perfectly orthogonal to key #1 leaks into the answer. With ${nPairs} pairs the leak has norm ${noiseNorm.toFixed(2)} against a signal of ${sigNorm.toFixed(2)}. Tick the box to turn on the delta rule.`;
  }
  draw();
}

// ================================================= 4. beta as a learning rate
export function mountBeta(root) {
  const box = panel(root, "β is a learning rate",
    "The delta rule is one step of gradient descent on ½‖Sᵀk − v‖². β says how much of the correction to apply: 0 writes nothing, 1 overwrites exactly.");

  const controls = el("div", "mb-4", box);
  const chart = svg("svg", { viewBox: "0 0 640 200", class: "w-full", role: "img" }, box);
  const caption = el("p", "text-sm text-gray-600 mt-2", box);

  let beta = 0.5;
  const readB = slider(controls, {
    label: "β (write strength)", min: 0, max: 1, step: 0.05, value: beta,
    format: (v) => v.toFixed(2),
  }, () => draw());

  function draw() {
    beta = readB();
    chart.innerHTML = "";
    const dk = 6, reps = 8;
    const rng = mulberry32(11);
    const k = randomUnit(rng, dk);
    const v = [1, -0.5, 0.8, 0.2];

    // write the same (k, v) repeatedly; watch the retrieval approach v
    const keys = Array.from({ length: reps }, () => k);
    const vals = Array.from({ length: reps }, () => v);
    const res = linearAttn(keys, keys, vals, {
      beta: new Array(reps).fill(beta), gate: "none", delta: true,
    });
    // Normalise by the error *before* any write, so the axis means "fraction of the
    // original gap left" and a run that lands immediately still shows the drop.
    const e0 = Math.hypot(...v);
    const errs = [e0, ...res.outputs.map((o) => Math.hypot(...o.map((x, j) => x - v[j])))];

    const x0 = 46, y0 = 18, w = 545, h = 120;
    svg("line", { x1: x0, y1: y0 + h, x2: x0 + w, y2: y0 + h, stroke: C.grid, "stroke-width": 2 }, chart);
    svg("line", { x1: x0, y1: y0, x2: x0, y2: y0 + h, stroke: C.grid, "stroke-width": 2 }, chart);
    [[0, "0"], [0.5, "50%"], [1, "100%"]].forEach(([frac, label]) => {
      const y = y0 + h - frac * h;
      svg("line", { x1: x0, y1: y, x2: x0 + w, y2: y, stroke: C.grid, "stroke-width": 1,
        "stroke-dasharray": frac === 0 ? null : "3 3" }, chart);
      svg("text", { x: x0 - 6, y: y + 4, "font-size": 10, fill: C.muted, "text-anchor": "end" }, chart)
        .textContent = label;
    });
    svg("text", { x: x0, y: y0 + h + 26, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = "0 = before any write, then one write per step →";
    svg("text", { x: 2, y: y0 - 4, "font-size": 10, fill: C.ink2 }, chart)
      .textContent = "error left";

    const pts = errs.map((e, i) => [x0 + (i / (errs.length - 1)) * w,
                                    y0 + h - Math.min(1, e / e0) * h]);
    svg("polyline", {
      points: pts.map((p) => p.join(",")).join(" "),
      fill: "none", stroke: C.s1, "stroke-width": 2, "stroke-linejoin": "round",
    }, chart);
    pts.forEach(([px, py], i) => {
      svg("circle", { cx: px, cy: py, r: 4.5, fill: C.s1, stroke: C.surface, "stroke-width": 2 }, chart);
      if (i === 1) {
        svg("text", { x: px + 8, y: py - 8, "font-size": 10, fill: C.ink2 }, chart)
          .textContent = `after 1 write: ${((errs[1] / e0) * 100).toFixed(0)}% left`;
      }
    });

    caption.textContent = beta === 0
      ? "β = 0: the correction is multiplied by zero, so nothing is ever written and the error never moves."
      : beta >= 0.999
        ? "β = 1: a single write lands the value exactly. This is a full overwrite — the memory answers this key perfectly from then on, and further writes change nothing."
        : `β = ${beta.toFixed(2)}: each write closes ${(beta * 100).toFixed(0)}% of the remaining gap, so the error decays geometrically rather than in one jump. After ${reps} writes, ${((errs[reps] / e0) * 100).toFixed(2)}% of the original error is left.`;
  }
  draw();
}

// ======================================= 5. one dial per head vs one per channel
export function mountGateGranularity(root) {
  const box = panel(root, "One forgetting dial, or one per channel",
    "This is the entire difference between Gated DeltaNet and KDA. Two facts arrive at different times and are needed at different times. A single decay per head has to compromise; a decay per channel does not.");

  const controls = el("div", "grid sm:grid-cols-2 gap-4 mb-4", box);
  const chart = svg("svg", { viewBox: "0 0 640 300", class: "w-full", role: "img" }, box);
  const caption = el("p", "text-sm text-gray-600 mt-2", box);

  const T = 30, nShow = 8;
  let halfLife = 12;
  const readH = slider(controls, {
    label: "Scalar gate: memory half-life", min: 2, max: 40, step: 1, value: halfLife,
    format: (v) => `${v} tokens`,
  }, () => draw());

  // Half-lives for the channels we display: four that hold on, four that turn over.
  // Showing a mix is the whole point -- a slice of only the slow ones looks identical
  // to a scalar gate that happens to be slow, which is exactly the wrong impression.
  const SLOW = 160, FAST = 4;
  const shownHalfLives = [SLOW, SLOW * 0.6, SLOW * 0.35, SLOW * 0.2, FAST * 4, FAST * 2, FAST, FAST * 0.6];

  function draw() {
    halfLife = readH();
    chart.innerHTML = "";

    const cell = 10, x0 = 66, y0 = 46, statsX = 66 + T * (cell + 1) + 22;

    // scalar gate: every channel decays at the same rate
    const aScalar = Math.pow(0.5, 1 / halfLife);
    const scalarRows = Array.from({ length: nShow }, () =>
      Array.from({ length: T }, (_, t) => Math.pow(aScalar, t + 1)));

    // channel gate: a spread of rates
    const channelRows = shownHalfLives.map((h) =>
      Array.from({ length: T }, (_, t) => Math.pow(Math.pow(0.5, 1 / h), t + 1)));

    svg("text", { x: x0, y: 18, "font-size": 12, fill: C.s2, "font-weight": 600 }, chart)
      .textContent = "Gated DeltaNet — one α per head";
    svg("text", { x: x0, y: 32, "font-size": 10, fill: C.muted }, chart)
      .textContent = "every channel fades at the same rate";
    drawMatrix(chart, scalarRows, { x: x0, y: y0, cell, scale: 1 });

    const y1 = y0 + nShow * (cell + 1) + 44;
    svg("text", { x: x0, y: y1 - 26, "font-size": 12, fill: C.s1, "font-weight": 600 }, chart)
      .textContent = "KDA — one α per channel";
    svg("text", { x: x0, y: y1 - 12, "font-size": 10, fill: C.muted }, chart)
      .textContent = "top channels hold on, bottom ones turn over";
    drawMatrix(chart, channelRows, { x: x0, y: y1, cell, scale: 1 });

    svg("text", { x: 2, y: y0 + 42, "font-size": 10, fill: C.ink2 }, chart).textContent = "channel";
    svg("text", { x: 2, y: y1 + 42, "font-size": 10, fill: C.ink2 }, chart).textContent = "channel";
    svg("text", { x: x0, y: y1 + nShow * (cell + 1) + 16, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = "tokens since the fact was written →";

    // how much survives to the end of the window
    const keepScalar = Math.pow(aScalar, T);
    const keepSlow = Math.pow(Math.pow(0.5, 1 / SLOW), T);
    const keepFast = Math.pow(Math.pow(0.5, 1 / FAST), T);
    svg("text", { x: statsX, y: y0 - 14, "font-size": 11, fill: C.ink2, "font-weight": 600 }, chart)
      .textContent = `still there after ${T} tokens`;
    [["scalar gate", keepScalar, C.s2],
     ["KDA slow channel", keepSlow, C.s1],
     ["KDA fast channel", keepFast, C.s1],
    ].forEach(([label, val, col], i) => {
      const y = y0 + 6 + i * 40;
      svg("text", { x: statsX, y, "font-size": 10, fill: C.ink2 }, chart).textContent = label;
      svg("rect", { x: statsX, y: y + 6, width: 150, height: 10, rx: 2, fill: "#f3f4f6" }, chart);
      svg("rect", { x: statsX, y: y + 6, width: Math.max(1.5, val * 150), height: 10, rx: 2, fill: col }, chart);
      svg("text", { x: statsX + 156, y: y + 15, "font-size": 11, fill: C.ink,
        "font-family": "ui-monospace,monospace" }, chart)
        .textContent = val < 0.001 ? val.toExponential(1) : (val * 100).toFixed(1) + "%";
    });

    caption.textContent =
      `With one dial per head, a half-life of ${halfLife} tokens is the model's only choice — it applies ` +
      `to the fact it needs to keep and the noise it wants gone alike. A per-channel gate lets it route ` +
      `long-lived facts to slow channels and scratch work to fast ones, in the same head at the same time. ` +
      `That is the whole of KDA's change to Gated DeltaNet: in the paper's configuration the one gate value ` +
      `per head becomes 128, one per channel of the state.`;
  }
  draw();
}

function transpose(M) {
  return M[0].map((_, j) => M.map((row) => row[j]));
}

// ========================== 6. KDA as a learnable positional encoding (Sec. 6.1)
export function mountDecayAsPosition(root) {
  const box = panel(root, "The decay is a positional encoding the model learns",
    "RoPE gives each pair of dimensions a fixed rotation frequency, so position enters as a set of hand-chosen periodic scales. KDA's per-channel decay does the same job with learned exponential scales — which is why the paper can drop positional encodings from its attention layers entirely.");

  const chart = svg("svg", { viewBox: "0 0 640 260", class: "w-full", role: "img" }, box);
  const controls = el("div", "mt-3", box);
  const caption = el("p", "text-sm text-gray-600 mt-3", box);

  let dist = 64;
  const readD = slider(controls, {
    label: "Distance between two tokens", min: 1, max: 256, step: 1, value: dist,
    format: (v) => `${v} tokens`,
  }, () => draw());

  function draw() {
    dist = readD();
    chart.innerHTML = "";
    const nCh = 8, maxD = 256;
    const x0 = 40, w = 250, y0 = 40, h = 80;

    // RoPE: cos(theta_c * d), theta_c = 10000^(-c/nCh)
    // KDA: gamma_c^d = exp(-r_c * d), r_c log-spaced
    const ropeFreq = Array.from({ length: nCh }, (_, c) => Math.pow(10000, -c / nCh));
    const kdaRate = Array.from({ length: nCh }, (_, c) =>
      Math.exp(Math.log(0.5) + (Math.log(0.002) - Math.log(0.5)) * (c / (nCh - 1))));

    function axes(ox, title, color) {
      svg("text", { x: ox, y: 18, "font-size": 12, fill: color, "font-weight": 600 }, chart)
        .textContent = title;
      svg("line", { x1: ox, y1: y0 + h, x2: ox + w, y2: y0 + h, stroke: C.grid, "stroke-width": 2 }, chart);
      svg("line", { x1: ox, y1: y0 - 10, x2: ox, y2: y0 + h, stroke: C.grid, "stroke-width": 2 }, chart);
      svg("text", { x: ox, y: y0 + h + 16, "font-size": 10, fill: C.muted }, chart).textContent = "0";
      svg("text", { x: ox + w - 14, y: y0 + h + 16, "font-size": 10, fill: C.muted }, chart)
        .textContent = String(maxD);
      svg("text", { x: ox + w / 2 - 30, y: y0 + h + 30, "font-size": 10, fill: C.ink2 }, chart)
        .textContent = "distance →";
    }

    axes(x0, "RoPE: fixed rotation frequencies", C.s2);
    ropeFreq.forEach((f, c) => {
      const pts = [];
      for (let d = 0; d <= maxD; d += 0.5) {
        pts.push([x0 + (d / maxD) * w, y0 + h / 2 - (Math.cos(f * d) * h) / 2.4]);
      }
      svg("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none",
        stroke: C.s2, "stroke-width": 1.5, opacity: 0.18 + 0.82 * (c / (nCh - 1)) }, chart);
    });

    const ox2 = x0 + w + 60;
    axes(ox2, "KDA: learned decay rates", C.s1);
    kdaRate.forEach((r, c) => {
      const pts = [];
      for (let d = 0; d <= maxD; d += 2) {
        pts.push([ox2 + (d / maxD) * w, y0 + h - Math.exp(-r * d) * h]);
      }
      svg("polyline", { points: pts.map((p) => p.join(",")).join(" "), fill: "none",
        stroke: C.s1, "stroke-width": 1.5, opacity: 0.25 + 0.75 * (1 - c / nCh) }, chart);
    });

    [x0, ox2].forEach((ox) => {
      const px = ox + (Math.min(dist, maxD) / maxD) * w;
      svg("line", { x1: px, y1: y0 - 10, x2: px, y2: y0 + h, stroke: C.ink,
        "stroke-width": 1.5, "stroke-dasharray": "4 3" }, chart);
    });

    // the signature each mechanism gives this distance
    const sy = 185;
    svg("text", { x: x0, y: sy, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = `signature at distance ${dist}:`;
    ropeFreq.forEach((f, c) => {
      const v = Math.cos(f * dist);
      svg("rect", { x: x0 + c * 26, y: sy + 10, width: 22, height: 26, rx: 3,
        fill: cellFill(v, 1) , stroke: C.grid }, chart);
    });
    kdaRate.forEach((r, c) => {
      const v = Math.exp(-r * dist);
      svg("rect", { x: ox2 + c * 26, y: sy + 10, width: 22, height: 26, rx: 3,
        fill: cellFill(v, 1), stroke: C.grid }, chart);
    });
    svg("text", { x: x0, y: sy + 54, "font-size": 10, fill: C.muted }, chart)
      .textContent = "oscillating — ambiguous at long range";
    svg("text", { x: ox2, y: sy + 54, "font-size": 10, fill: C.muted }, chart)
      .textContent = "monotone — a clean recency ordering";

    caption.textContent =
      `Both give every distance a distinctive pattern across channels, which is all a positional code has to do. ` +
      `RoPE's is periodic, so the same pattern recurs at distances the model may never have trained on. ` +
      `KDA's is monotone decreasing and its rates are learned rather than fixed, which is the argument in §6.1 ` +
      `for letting the linear layers carry position and giving the attention layers none at all.`;
  }
  draw();
}

// ============================================= 7. the chunkwise algorithm
export function mountChunkwise(root) {
  const box = panel(root, "Why it is computed in chunks",
    "The recurrence is sequential, which a GPU hates. Splitting into chunks of C makes everything inside a chunk a few dense matmuls and leaves only chunk-to-chunk state passing sequential.");

  const controls = el("div", "grid sm:grid-cols-2 gap-4 mb-4", box);
  const chart = svg("svg", { viewBox: "0 0 640 250", class: "w-full", role: "img" }, box);
  const caption = el("p", "text-sm text-gray-600 mt-2", box);

  let logT = 12, C_ = 64;
  const readT = slider(controls, {
    label: "Sequence length", min: 8, max: 20, step: 1, value: logT,
    format: (v) => fmtNum(2 ** v),
  }, () => draw());
  const readC = slider(controls, {
    label: "Chunk size C", min: 16, max: 256, step: 16, value: C_,
    format: (v) => String(v),
  }, () => draw());

  function draw() {
    const T = 2 ** readT(), cs = readC(), d = 128;
    chart.innerHTML = "";

    // the block picture: a T x T causal map, tiled
    const nShow = 8, cell = 18, x0 = 20, y0 = 30;
    svg("text", { x: x0, y: 18, "font-size": 12, fill: C.ink2 }, chart)
      .textContent = "how a chunked causal map is covered";
    for (let i = 0; i < nShow; i++) {
      for (let j = 0; j < nShow; j++) {
        let fill = "#f9fafb", label = null;
        if (j < i) { fill = C.cool; label = "inter"; }
        else if (j === i) { fill = C.warm; label = "intra"; }
        svg("rect", { x: x0 + j * (cell + 2), y: y0 + i * (cell + 2), width: cell,
          height: cell, rx: 2, fill, stroke: C.grid }, chart);
      }
    }
    svg("rect", { x: 200, y: y0, width: 12, height: 12, rx: 2, fill: C.warm, stroke: C.grid }, chart);
    svg("text", { x: 218, y: y0 + 10, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = "intra-chunk: one masked matmul + a triangular solve";
    svg("rect", { x: 200, y: y0 + 22, width: 12, height: 12, rx: 2, fill: C.cool, stroke: C.grid }, chart);
    svg("text", { x: 218, y: y0 + 32, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = "inter-chunk: carried in the state, not recomputed";
    svg("text", { x: 218, y: y0 + 54, "font-size": 11, fill: C.muted }, chart)
      .textContent = `${fmtNum(Math.ceil(T / cs))} chunks of ${cs} tokens`;

    // cost curve
    const bx = 20, by = 190, bw = 600, bh = 0;
    const f = flops(T, d, cs);
    const total = f.kda;
    const parts = [
      ["6Td²  the projections", 6 * T * d * d, C.s1],
      ["3TCd  intra-chunk matmuls", 3 * T * cs * d, C.s3],
      ["TC²   the triangular solve", T * cs * cs, C.s2],
    ];
    let acc = 0;
    parts.forEach(([label, val, col]) => {
      const w = (val / total) * bw;
      svg("rect", { x: bx + acc, y: by, width: Math.max(1, w - 2), height: 22, rx: 3, fill: col }, chart);
      acc += w;
    });
    parts.forEach(([label, val, col], i) => {
      svg("rect", { x: bx + i * 200, y: by + 32, width: 10, height: 10, rx: 2, fill: col }, chart);
      svg("text", { x: bx + i * 200 + 15, y: by + 41, "font-size": 10, fill: C.ink2 }, chart)
        .textContent = `${label} (${((val / total) * 100).toFixed(0)}%)`;
    });
    svg("text", { x: bx, y: by - 8, "font-size": 12, fill: C.ink2 }, chart)
      .textContent = `where the ${fmtNum(total)} FLOPs per head go`;

    const vsAttn = f.attention / f.kda;
    caption.textContent =
      `C trades two costs against each other: the triangular solve inside a chunk grows as C², ` +
      `while the number of sequential chunk steps falls as 1/C. C = 64 is the paper's choice. ` +
      `At T = ${fmtNum(T)} this costs ${vsAttn >= 1 ? vsAttn.toFixed(1) + "× less" : (1 / vsAttn).toFixed(1) + "× more"} than full attention.`;
  }
  draw();
}

// ================================================ 8. the hybrid stack
export function mountHybridStack(root) {
  const box = panel(root, "Three KDA layers, then one that sees everything",
    "Pure linear attention cannot copy exactly at long range; full attention costs a cache that grows forever. The paper's answer is to interleave them 3:1. Click a layer to change its type.");

  const controls = el("div", "mb-4", box);
  const chart = svg("svg", { viewBox: "0 0 640 220", class: "w-full" }, box);
  const caption = el("p", "text-sm text-gray-600 mt-2", box);

  const nLayers = 16;
  let plan = layerPlan(nLayers, 3);
  let logLen = 17;

  const readLen = slider(controls, {
    label: "Context length", min: 10, max: 20, step: 1, value: logLen,
    format: (v) => fmtNum(2 ** v) + " tokens",
  }, () => draw());

  const presets = el("div", "flex flex-wrap gap-2 mt-3", controls);
  [["all attention", 0], ["1:1", 1], ["3:1 (the paper)", 3], ["7:1", 7], ["all KDA", null]]
    .forEach(([label, r]) => {
      const b = el("button", "px-2.5 py-1 text-xs rounded border border-gray-300 hover:bg-gray-50", presets, label);
      b.addEventListener("click", () => { plan = layerPlan(nLayers, r); draw(); });
    });

  function draw() {
    const seqLen = 2 ** readLen();
    chart.innerHTML = "";
    const heads = 32, headDim = 128, dModel = heads * headDim;
    const w = 30, gap = 6, x0 = 20;

    plan.forEach((kind, i) => {
      const x = x0 + i * (w + gap);
      const isAttn = kind === "full";
      const r = svg("rect", { x, y: 40, width: w, height: 70, rx: 5,
        fill: isAttn ? C.s2 : C.s1, opacity: isAttn ? 1 : 0.85,
        style: "cursor:pointer" }, chart);
      r.addEventListener("click", () => {
        plan[i] = isAttn ? "linear" : "full";
        draw();
      });
      svg("text", { x: x + w / 2, y: 80, "text-anchor": "middle", "font-size": 10,
        fill: "#fff", "font-weight": 600, style: "pointer-events:none" }, chart)
        .textContent = isAttn ? "A" : "K";
    });
    svg("text", { x: x0, y: 30, "font-size": 11, fill: C.ink2 }, chart)
      .textContent = `${nLayers} layers — K = KDA, A = full attention`;
    svg("text", { x: x0, y: 126, "font-size": 10, fill: C.muted }, chart)
      .textContent = "input →  layer 1 … layer 16  → output";

    const nAttn = plan.filter((k) => k === "full").length;
    const kvBytes = nAttn * 2 * seqLen * dModel * 2;
    const stateBytes = (nLayers - nAttn) * heads * headDim * headDim * 2;
    const fullBytes = nLayers * 2 * seqLen * dModel * 2;

    const rows = [["this configuration", kvBytes + stateBytes, C.s1],
                  ["all full attention", fullBytes, C.s2]];
    const max = Math.max(...rows.map((r) => r[1]));
    rows.forEach(([label, val, col], i) => {
      const y = 155 + i * 30;
      svg("text", { x: x0, y: y - 3, "font-size": 11, fill: C.ink2 }, chart).textContent = label;
      svg("rect", { x: 180, y: y - 14, width: 340, height: 16, rx: 3, fill: "#f3f4f6" }, chart);
      svg("rect", { x: 180, y: y - 14, width: Math.max(2, (val / max) * 340), height: 16, rx: 3, fill: col }, chart);
      svg("text", { x: 530, y: y - 1, "font-size": 12, fill: C.ink,
        "font-family": "ui-monospace,monospace" }, chart).textContent = fmtBytes(val);
    });

    const pct = (1 - (kvBytes + stateBytes) / fullBytes) * 100;
    caption.textContent =
      `${nAttn} of ${nLayers} layers keep a growing KV cache; the other ${nLayers - nAttn} keep a fixed state. ` +
      `At ${fmtNum(seqLen)} tokens that is ${pct >= 0 ? pct.toFixed(0) + "% less" : (-pct).toFixed(0) + "% more"} memory than full attention. ` +
      (nAttn === 0
        ? "With no attention layer at all there is nothing that can copy a distant span exactly — which is what the ablation in Table 1 shows costs the most."
        : "The attention layers are what preserve exact long-range retrieval; the ratio decides how much you pay for it.");
  }
  draw();
}

// ------------------------------------------------------------------ auto-mount
const REGISTRY = {
  "cache-growth": mountCacheGrowth,
  associativity: mountAssociativity,
  interference: mountInterference,
  beta: mountBeta,
  "gate-granularity": mountGateGranularity,
  "decay-as-position": mountDecayAsPosition,
  chunkwise: mountChunkwise,
  "hybrid-stack": mountHybridStack,
};

export function mountAll(scope = document) {
  for (const node of scope.querySelectorAll("[data-widget]")) {
    const fn = REGISTRY[node.dataset.widget];
    if (!fn) { console.warn("unknown widget", node.dataset.widget); continue; }
    try {
      fn(node);
    } catch (e) {
      console.error("widget failed:", node.dataset.widget, e);
      node.innerHTML = '<p class="text-sm text-gray-500 border rounded p-3">This figure failed to load.</p>';
    }
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => mountAll());
  } else {
    mountAll();
  }
}
