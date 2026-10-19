"""Driver: optimism of the reported guarantee, 5 days x 3 demand scales."""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from campaign import deep, optimism  # noqa: E402

SCALES = ((1, "deep"), (3, "deep_x3"), (4, "deep_x4"))


def main(workers: int = 4) -> int:
    jobs = []
    for scale, folder in SCALES:
        for d in range(deep.N_DAYS):
            for s in deep.SEARCH_SEEDS:
                cell = ROOT / "data" / folder / "cells" / f"d{d}_x{s}.json"
                if cell.exists():
                    jobs.append(dict(
                        root=str(ROOT / "simulator"),
                        sys_path=[str(ROOT), str(ROOT / "simulator")],
                        scale=scale, day=d, search_seed=s,
                        n_real=deep.N_REALISATIONS, cell=str(cell)))

    print(f"{len(jobs)} cells x {optimism.N_FRESH_POOLS} fresh pools", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(optimism.run_cell, jobs), 1):
            rows.extend(got)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    out = ROOT / "data" / "optimism"
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["scale"], r["day"], r["search_seed"], r["role"]))
    with (out / "optimism.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / 'optimism.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
