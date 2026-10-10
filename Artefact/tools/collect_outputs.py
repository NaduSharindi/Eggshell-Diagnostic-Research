#!/usr/bin/env python3
"""
collect_outputs.py — package everything a notebook produced into one zip.

Run this on the training machine after a notebook finishes, copy the zip to the
laptop, and unzip it at the repository root. The archive preserves the exact
folder structure, so unzipping puts every file where git expects it.

Usage
-----
    python Artefact/tools/collect_outputs.py                  # everything so far
    python Artefact/tools/collect_outputs.py --stage 01       # just notebook 01
    python Artefact/tools/collect_outputs.py --out D:/transfer

On the laptop
-------------
    unzip -o eggshell_outputs_01_2026-10-06.zip -d <repo root>
    git status          # review, then add / commit / push
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import date
from pathlib import Path

# make Artefact/src importable regardless of where this is run from
HERE = Path(__file__).resolve()
ARTEFACT = HERE.parents[1]
sys.path.insert(0, str(ARTEFACT / "src"))
from common import project_config as CFG  # noqa: E402

REPO = CFG.REPO_ROOT

# Files that are evidence for the dissertation and belong in git.
# Images and model weights are deliberately excluded: they are large and regenerable.
INCLUDE_GLOBS = [
    "Artefact/results/**/*",
    "Artefact/data/processed/*.csv",
    "Artefact/data/processed/*.json",
]
EXCLUDE_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".pyc"}
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def gather(stage: str | None) -> list[Path]:
    seen: list[Path] = []
    for pattern in INCLUDE_GLOBS:
        for p in REPO.glob(pattern):
            if not p.is_file():
                continue
            if p.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            if EXCLUDE_PARTS & set(p.parts):
                continue
            if stage and f"{stage}_" not in str(p.relative_to(REPO)):
                continue
            seen.append(p)
    return sorted(set(seen))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", help="limit to one stage, e.g. 01 or 05")
    ap.add_argument("--out", default=None, help="directory to write the zip into")
    args = ap.parse_args()

    files = gather(args.stage)
    if not files:
        print(f"Nothing to collect under {REPO}. Has the notebook been run?")
        return 1

    tag = f"_{args.stage}" if args.stage else ""
    out_dir = Path(args.out).expanduser() if args.out else REPO
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"eggshell_outputs{tag}_{date.today().isoformat()}.zip"

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(REPO))

    total_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"repository root : {REPO}")
    print(f"files packaged  : {len(files)}  ({total_mb:.1f} MB uncompressed)")
    print(f"archive written : {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)")
    print("\ncontents:")
    for f in files[:40]:
        print(f"  {f.relative_to(REPO)}")
    if len(files) > 40:
        print(f"  ... and {len(files) - 40} more")
    print("\nOn the laptop, unzip this at the repository root, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
