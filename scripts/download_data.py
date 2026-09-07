#!/usr/bin/env python3
"""Fetch the competition data into ``data/raw/``.

Credentials come from the environment, never from the source tree:

    export KAGGLE_USERNAME=your-username
    export KAGGLE_KEY=your-api-key
    python scripts/download_data.py

or place the JSON Kaggle gives you at ``~/.kaggle/kaggle.json``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "equity-post-HCT-survival-predictions"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def main() -> int:
    if not have_credentials():
        sys.exit(
            "No Kaggle credentials found.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY, or install ~/.kaggle/kaggle.json.\n"
            "Never commit either to the repository."
        )

    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / f"{COMPETITION}.zip"

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW)],
        check=True,
    )
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW)
    archive.unlink(missing_ok=True)

    print(f"Extracted to {RAW}")
    for path in sorted(RAW.glob("*.csv")):
        print(" ", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
