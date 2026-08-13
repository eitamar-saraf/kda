/**
 * Inline side-notes: a highlighted term that reveals a small popover.
 *
 * The point is pacing. A reader who already knows what HBM is should not have to wade
 * through a paragraph about it to reach the next idea, and a reader who does not should
 * not have to go and look it up. So the depth lives one click away, inline, next to the
 * word that raises the question.
 *
 * Markup the prose uses -- the note body is real content in the page, not a JS string:
 *
 *   <span class="kn">
 *     <button type="button" class="kn-t">read</button>
 *     <span class="kn-b"><b>From where?</b> Out of HBM, across the bus...</span>
 *   </span>
 *
 * Without JavaScript, nothing hides: `.kn-b` is only collapsed once this module sets
 * `data-kn="on"` on the root, so a failed script leaves every note visible as ordinary
 * parenthetical text rather than losing it. Same reason the body is not injected --
 * it stays in the HTML for search engines and screen readers either way.
 *
 * Interaction: click or Enter/Space toggles. On devices with a real pointer, hover
 * opens too. Escape closes, clicking elsewhere closes, and only one is open at a time.
 */

const STYLE_ID = "kn-style";

const CSS = `
.kn { position: relative; display: inline; }
.kn-t {
  font: inherit; color: #1d4ed8; background: none; border: 0; padding: 0 1px;
  cursor: pointer; text-decoration: underline;
  text-decoration-style: dotted; text-decoration-thickness: 1.5px;
  text-underline-offset: 2.5px;
}
.kn-t:hover, .kn-t[aria-expanded="true"] { background: #dbeafe; border-radius: 3px; }
.kn-t:focus-visible { outline: 2px solid #2a78d6; outline-offset: 2px; border-radius: 3px; }
.kn-t::after { content: "\\00a0?"; font-size: 0.7em; vertical-align: super; opacity: 0.65; }

/* Only hidden once the script has run -- see the note above. */
[data-kn="on"] .kn-b { display: none; }
[data-kn="on"] .kn.kn-open .kn-b {
  display: block; position: absolute; z-index: 40;
  top: calc(100% + 8px); left: 0;
  width: max-content; max-width: min(26rem, calc(100vw - 2.5rem));
  background: #fff; color: #374151;
  border: 1px solid #d1d5db; border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0,0,0,.10), 0 2px 6px rgba(0,0,0,.06);
  padding: 12px 14px; font-size: 0.875rem; line-height: 1.55;
  text-align: left; font-weight: 400; white-space: normal;
  /* A symbol legend plus a precision table can run to ~1000px. Cap it and scroll,
     or a long note runs off the bottom of a laptop screen with no way back. */
  max-height: min(70vh, 32rem); overflow-y: auto; overscroll-behavior: contain;
}
.kn-b code, .kn-b .m { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }
/* A note lives inside an article <p>, so its body may contain only phrasing content:
   a <p> or <div> or <table> in there makes the parser close the enclosing paragraph and
   spill the note into the article flow, leaving this span empty. Blocks are spans. */
.kn-p { display: block; margin: 0 0 8px; }
.kn-p:last-of-type { margin-bottom: 0; }
.kn-eq {
  display: block; margin: 0 0 8px; padding: 6px 8px;
  background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem;
}
.kn-eq-ok { background: #f0fdf4; border-color: #bbf7d0; }
.kn-tbl {
  display: grid; grid-template-columns: auto auto auto;
  column-gap: 16px; row-gap: 3px; margin: 8px 0 2px; font-size: 0.8rem;
}
.kn-tbl > .kn-hd { font-weight: 600; border-bottom: 1px solid #e5e7eb; padding-bottom: 2px; }
/* Symbol legend: a narrow mono column, a flexible description, a mono value. */
.kn-legend {
  display: grid; grid-template-columns: auto 1fr auto;
  column-gap: 12px; row-gap: 4px; margin: 8px 0; font-size: 0.8rem;
  align-items: baseline;
}
.kn-legend > .kn-sym {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 600; color: #1f2937; white-space: nowrap;
}
.kn-legend > .kn-val {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #1d4ed8; white-space: nowrap; text-align: right;
}
.kn-legend > .kn-hd { font-weight: 600; border-bottom: 1px solid #e5e7eb; padding-bottom: 3px; font-size: 0.75rem; color: #6b7280; }
.kn-close {
  display: block; margin-top: 10px; font-size: 0.75rem; color: #6b7280;
  background: none; border: 0; padding: 0; cursor: pointer; text-decoration: underline;
}
@media (max-width: 640px) {
  /* On a narrow screen, anchoring to the word overflows. Pin to the text column. */
  [data-kn="on"] .kn.kn-open .kn-b {
    position: fixed; left: 1rem; right: 1rem; bottom: 1rem; top: auto;
    width: auto; max-width: none; max-height: 60vh;
  }
}
`;

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement("style");
  el.id = STYLE_ID;
  el.textContent = CSS;
  document.head.appendChild(el);
}

let openNote = null;

function close(note) {
  if (!note) return;
  note.classList.remove("kn-open");
  note.querySelector(".kn-t")?.setAttribute("aria-expanded", "false");
  if (openNote === note) openNote = null;
}

function open(note) {
  if (openNote && openNote !== note) close(openNote);
  note.classList.add("kn-open");
  note.querySelector(".kn-t")?.setAttribute("aria-expanded", "true");
  openNote = note;
  clampIntoView(note);
}

/** Nudge the popover left if it would run off the right edge. */
function clampIntoView(note) {
  const body = note.querySelector(".kn-b");
  if (!body) return;
  body.style.left = "0px";
  body.style.top = "";
  body.style.bottom = "";
  body.style.maxHeight = "";
  if (window.innerWidth <= 640) return;          // pinned by CSS at that width

  // horizontal: nudge left if it would run off the right edge
  let r = body.getBoundingClientRect();
  const overflowX = r.right - (window.innerWidth - 16);
  if (overflowX > 0) body.style.left = `${-overflowX}px`;

  // Vertical: a legend plus a table can be taller than either side of the term, so
  // pick the roomier side *and* cap the height to what is actually there. Capping is
  // the part that matters -- flipping alone still left it hanging off the bottom of a
  // 900px laptop screen, because neither side had 512px to give.
  const term = note.querySelector(".kn-t").getBoundingClientRect();
  const GAP = 14;
  const roomBelow = window.innerHeight - term.bottom - GAP;
  const roomAbove = term.top - GAP;
  const cap = Math.min(window.innerHeight * 0.7, 32 * 16);
  const wanted = Math.min(cap, body.scrollHeight);

  const flip = wanted > roomBelow && roomAbove > roomBelow;
  const room = flip ? roomAbove : roomBelow;
  body.style.maxHeight = `${Math.max(150, Math.min(cap, room))}px`;
  if (flip) {
    body.style.top = "auto";
    body.style.bottom = "calc(100% + 8px)";
  } else {
    body.style.top = "calc(100% + 8px)";
    body.style.bottom = "auto";
  }
}

function wire(note, i) {
  const term = note.querySelector(".kn-t");
  const body = note.querySelector(".kn-b");
  if (!term || !body) return;

  // Guard against the failure that cost me an hour: a note body containing a <p>,
  // <div> or <table> is illegal here, because the note sits inside an article <p> and
  // the parser closes it on sight -- ejecting the content into the article flow and
  // leaving this span empty. It renders as stray prose rather than as an error, so
  // say so loudly.
  if (!body.textContent.trim()) {
    console.warn("[notes] empty note body -- did it contain a <p>/<div>/<table>? " +
                 "Only phrasing content is legal inside a note. Term:", term.textContent);
  }

  if (!body.id) body.id = `kn-body-${i}`;
  term.setAttribute("aria-expanded", "false");
  term.setAttribute("aria-controls", body.id);
  body.setAttribute("role", "note");

  term.addEventListener("click", (e) => {
    e.stopPropagation();
    note.classList.contains("kn-open") ? close(note) : open(note);
  });

  // Hover is an enhancement, never the only way in -- it does not exist on touch.
  if (window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
    let t;
    note.addEventListener("mouseenter", () => { t = setTimeout(() => open(note), 90); });
    note.addEventListener("mouseleave", () => { clearTimeout(t); close(note); });
  }

  // A close affordance, because on touch the popover can cover what you were reading.
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "kn-close";
  btn.textContent = "close";
  btn.addEventListener("click", (e) => { e.stopPropagation(); close(note); term.focus(); });
  body.appendChild(btn);

  body.addEventListener("click", (e) => e.stopPropagation());
}

export function mountNotes(scope = document) {
  const notes = [...scope.querySelectorAll(".kn")];
  if (!notes.length) return 0;
  ensureStyle();
  document.documentElement.setAttribute("data-kn", "on");
  notes.forEach(wire);

  document.addEventListener("click", () => close(openNote));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close(openNote);
  });
  window.addEventListener("resize", () => openNote && clampIntoView(openNote));
  return notes.length;
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => mountNotes());
  } else {
    mountNotes();
  }
}
