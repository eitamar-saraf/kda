"""Keep the article's copy of the browser maths identical to the tested one.

The writeup claims its interactive figures run the same maths the Python suite pins to
the paper. That claim rests on ``js/kda-math.js`` and ``js/widgets.js`` being byte-for-byte
the files served from the website repo -- and they live in two different repositories,
copied across by hand (the two-repo workflow in the site's AUTHORING.md).

So the copy is checkable rather than assumed:

    python -m scripts.sync_site --check     # exit 1 if the deployed copy has drifted
    python -m scripts.sync_site             # copy over and report what changed

Run ``--check`` before publishing. A stale copy would leave the figures silently
computing something the tests never saw, which is exactly the failure the whole
fixture-testing setup exists to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

#: Files the article loads directly, and where they land in the website repo.
ASSETS = ["kda-math.js", "widgets.js"]
DEFAULT_SITE = Path.home() / "code" / "eitamar-saraf.github.io"
SITE_SUBDIR = Path("public") / "kimi-linear"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--site", default=str(DEFAULT_SITE), help="path to the website repo")
    p.add_argument("--src", default="js", help="directory holding the source assets")
    p.add_argument("--check", action="store_true",
                   help="report drift and exit non-zero instead of copying")
    a = p.parse_args()

    src_dir = Path(a.src)
    dest_dir = Path(a.site) / SITE_SUBDIR
    if not dest_dir.exists():
        print(f"site asset directory not found: {dest_dir}", file=sys.stderr)
        return 2

    drift = []
    for name in ASSETS:
        src, dest = src_dir / name, dest_dir / name
        if not src.exists():
            print(f"missing source {src}", file=sys.stderr)
            return 2
        if not dest.exists():
            drift.append((name, "missing", digest(src), "-"))
            continue
        ds, dd = digest(src), digest(dest)
        if ds != dd:
            drift.append((name, "differs", ds, dd))

    if not drift:
        print(f"in sync: {', '.join(ASSETS)} match {dest_dir}")
        return 0

    for name, why, ds, dd in drift:
        print(f"{'DRIFT' if a.check else 'copying'}: {name} ({why})  source {ds}  deployed {dd}")

    if a.check:
        print("\nThe article would serve maths the test suite has not checked.\n"
              "Run `python -m scripts.sync_site` and rebuild the site.", file=sys.stderr)
        return 1

    for name, *_ in drift:
        shutil.copy2(src_dir / name, dest_dir / name)
    print(f"copied {len(drift)} file(s) to {dest_dir}; rebuild the site to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
