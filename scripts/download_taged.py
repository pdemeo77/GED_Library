"""Download TaGED.json: mapping-augmented ground truth for AIDS/IMDB/Linux.

TaGED.json (from chengqihao/GED-via-Optimal-Transport, the same data
distributed by ChengzhiPiao/GEDGNN) adds, on top of the scalar true GED,
a decomposition into node relabelings / node insertions / edge insertions
(TaGSim convention) and one optimal node mapping per pair. Verified
identical to True_GED/<DS>/<DS>_ged.csv on all AIDS pairs (245,350/245,350).

Entry format: [id1, id2, ged, ged_nc, ged_in, ged_ie, [[u, v], ...]]

Files are large (AIDS 55 MB, IMDB 90 MB, Linux 103 MB) and gitignored:
download them on demand per machine.

Usage:
    python scripts/download_taged.py                     # AIDS only
    python scripts/download_taged.py --datasets AIDS IMDB Linux
    python scripts/download_taged.py --datasets all
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/chengqihao/"
    "GED-via-Optimal-Transport/main/json_data"
)
DATASETS = {"AIDS": "AIDS", "IMDB": "IMDB", "LINUX": "Linux"}

_PKG_ROOT = Path(__file__).resolve().parent.parent


_last_pct = [-1]


def _progress(block_num: int, block_size: int, total: int) -> None:
    done = block_num * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        if pct != _last_pct[0] and (pct % 5 == 0 or pct == 100):
            _last_pct[0] = pct
            sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct}%)")
            sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  {done / 1e6:6.1f} MB")
        sys.stdout.flush()


def download(dataset_key: str, force: bool) -> None:
    folder = DATASETS[dataset_key]
    dest = _PKG_ROOT / "True_GED" / folder / "TaGED.json"
    if dest.exists() and not force:
        print(f"[{folder}] already present: {dest} (use --force to re-download)")
        return

    url = f"{BASE_URL}/{folder}/TaGED.json"
    print(f"[{folder}] downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _last_pct[0] = -1
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001 — report and continue with others
        print(f"\n  ERROR downloading {folder}: {exc}")
        if dest.exists():
            dest.unlink()
        return
    print(f"\n  saved: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TaGED.json ground-truth files (see module docstring)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AIDS"],
        help="Datasets to download (AIDS IMDB Linux, or 'all'). Default: AIDS",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    keys = list(DATASETS) if "all" in args.datasets else [d.upper() for d in args.datasets]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        parser.error(f"Unknown dataset(s): {unknown}. Choose from {list(DATASETS)} or 'all'.")

    for key in keys:
        download(key, args.force)


if __name__ == "__main__":
    main()
