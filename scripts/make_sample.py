"""Generate small sample datasets, overwriting Dataset/ and True_GED/.

Run ONCE after you have backed up the full corpus to a GitHub release asset
(see scripts/download_datasets.py). Keeps the first N graph ids that appear in
the ground truth, so that the sampled graphs and sampled pairs stay consistent
and examples/tests keep working out of the box.

Usage:
    python scripts/make_sample.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Number of graphs to keep per dataset.
N = 50

DATASETS = {
    "AIDS": {"graphs": "AIDS_graphs.csv", "ged": "AIDS_ged.csv"},
    "IMDB": {"graphs": "IMDB_graphs.csv", "ged": "IMDB_ged.csv"},
    "Linux": {"graphs": "Linux_graphs.csv", "ged": "Linux_ged.csv"},
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    for name, spec in DATASETS.items():
        g_path = ROOT / "Dataset" / name / spec["graphs"]
        d_path = ROOT / "True_GED" / name / spec["ged"]
        if not g_path.exists():
            print(f"  skip {name}: {g_path} not found")
            continue

        # Pick ids that actually appear in the ground truth so the sample has
        # internal pairs (otherwise examples/quickstart would find 0 pairs).
        d_rows = _read_csv(d_path) if d_path.exists() else []
        pair_ids = sorted({i for r in d_rows for i in (r["id_1"], r["id_2"])})
        selected = set(pair_ids[:N])

        g_rows = [r for r in _read_csv(g_path) if r["graph_id"] in selected]
        d_keep = [r for r in d_rows if r["id_1"] in selected and r["id_2"] in selected]

        _write_csv(g_path, g_rows)
        if d_keep:
            _write_csv(d_path, d_keep)

        print(f"  {name}: {len(g_rows)} graphs, {len(d_keep)} ged pairs")


if __name__ == "__main__":
    run()
    print("Done. Dataset/ and True_GED/ now contain small samples.")
