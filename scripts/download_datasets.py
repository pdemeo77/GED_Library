"""Download the full benchmark datasets from a GitHub release asset.

Overwrites the small samples shipped in the repo with the complete AIDS / IMDB /
Linux corpora. Configure the source via env vars (or edit the constants below).

Usage:
    export GED_DATA_OWNER=YOUR_GITHUB_USER
    export GED_DATA_REPO=YOUR_REPO
    python scripts/download_datasets.py
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OWNER = os.environ.get("GED_DATA_OWNER", "YOUR_GITHUB_USER")
REPO = os.environ.get("GED_DATA_REPO", "YOUR_REPO")
TAG = os.environ.get("GED_DATA_TAG", "v1.0-data")
ASSET = os.environ.get("GED_DATA_ASSET", "data.zip")
URL = f"https://github.com/{OWNER}/{REPO}/releases/download/{TAG}/{ASSET}"


def main() -> int:
    if "YOUR_GITHUB_USER" in OWNER:
        print("ERROR: set env vars GED_DATA_OWNER / GED_DATA_REPO (or edit this script).")
        print(f"  Expected asset: {URL}")
        return 1

    zip_path = ROOT / "data.zip"
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT)
    zip_path.unlink()

    print("Done. Full datasets extracted into Dataset/ and True_GED/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
