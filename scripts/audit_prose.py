"""Check the writeup for the failure modes I keep reintroducing.

Rewriting prose breaks prose in ways tests never catch. Each check here exists because
the corresponding mistake actually shipped:

1. **Block tags inside a paragraph.** A note body containing ``<p>`` or ``<table>`` makes
   the parser close the enclosing ``<p>`` and spill the note into the article flow,
   leaving the popover empty. It renders as stray text, not as an error.
2. **Orphaned jargon.** Replacing a section can delete the only place a term was
   introduced, leaving it used-but-undefined further down. This is what happened to
   "fast weights" when the associative-memory paragraph was rewritten.
3. **Stale claims.** A correction applied to the body but not to the summary leaves the
   post contradicting itself. The TL;DR asserted the delta rule fixes crowding for a
   day after section 5 was rewritten to say it does not.
4. **Over-packed sentences.** The symptom a reader reports as "I did not get this
   paragraph" is usually one sentence carrying four claims.

Usage::

    python -m scripts.audit_prose --post ~/code/eitamar-saraf.github.io/src/pages/blog/kimi-linear.astro
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Terms that need an introduction before first use. Value = the marker that counts as
#: introducing them (a note, or explanatory phrasing nearby).
JARGON = [
    "fast weights", "associative memory", "delta rule", "WY representation",
    "UT transform", "DPLR", "NoPE", "chunkwise", "HBM", "grouped-query",
]

#: Claims that were wrong and must not reappear.
BANNED = [
    (r"delta rule[^.]{0,80}fix[^.]{0,40}interferen", "the delta rule does NOT fix interference/crowding"),
    (r"interferen[^.]{0,60}which the delta rule fixes", "same claim, other phrasing"),
]

MAX_WORDS = 45


def strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True)
    ap.add_argument("--max-words", type=int, default=MAX_WORDS)
    a = ap.parse_args()
    src = Path(a.post).read_text()
    fails, warns = [], []

    # 1. block tags nested inside a paragraph
    for m in re.finditer(r"<p\b[^>]*>(.*?)</p>", src, re.S):
        for tag in ("<p ", "<p>", "<div", "<table", "<ul", "<ol", "<details"):
            if tag in m.group(1):
                line = src[: m.start()].count("\n") + 1
                fails.append(f"line {line}: {tag} nested inside <p> -- the parser will eject it")

    # 2. empty note bodies (the visible symptom of check 1)
    for m in re.finditer(r'<span class="kn-b"[^>]*>(.*?)</span></span>', src, re.S):
        if not strip_tags(m.group(1)).strip():
            fails.append(f"line {src[:m.start()].count(chr(10))+1}: empty note body")

    # 3. banned claims
    text = strip_tags(src)
    for pattern, why in BANNED:
        if re.search(pattern, text, re.I):
            fails.append(f"stale claim resurfaced: {why}")

    # 4. jargon used before it is introduced
    for term in JARGON:
        first = text.lower().find(term.lower())
        if first < 0:
            continue
        window = text[max(0, first - 400) : first + 400].lower()
        introduced = any(
            k in window for k in (f"<button", "call those", "that is", "means", "known as",
                                  "i.e.", "—", "the same object", "kn-t")
        )
        # a note on the term counts as an introduction
        noted = re.search(r'class="kn-t"[^>]*>\s*' + re.escape(term), src, re.I) is not None
        if not (introduced or noted):
            warns.append(f"'{term}' may be used before it is introduced")

    # 5. over-packed sentences, main prose only
    t = src
    for pat in (r'<span class="kn-b".*?</span></span>', r"<table.*?</table>",
                r"<figcaption.*?</figcaption>", r"<details.*?</details>", r"<nav.*?</nav>",
                r"^---.*?---"):
        t = re.sub(pat, " ", t, flags=re.S | re.M)
    # Displayed equations get glued onto the neighbouring sentence by any naive
    # splitter, which produces 100-word "sentences" that are really prose + maths +
    # prose. Drop fragments carrying maths notation -- what is left is real prose.
    MATHY = ("\u1d40", "\u03a3", "\u27f9", "\u00b7", "\u00d7", "&nbsp;", "\u2299", "\u2192")
    for sent in re.split(r"(?<=[.!?]) ", strip_tags(t)):
        n = len(sent.split())
        if n > a.max_words and not any(m in sent for m in MATHY):
            warns.append(f"{n}-word sentence: {sent.strip()[:110]}...")

    for f in fails:
        print(f"FAIL  {f}")
    for w in warns:
        print(f"warn  {w}")
    print(f"\n{len(fails)} failures, {len(warns)} warnings")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
