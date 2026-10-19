"""Driver: the saturation penalty across the four tariff patterns.

The valley hours and the purchasable floor are identical in all four patterns,
so the attainable-coverage ceiling cannot depend on the pattern -- it is set by
sockets, power and time, none of which price touches. What the pattern changes
is what spilling out of the valley costs. This run measures that, and verifies
the ceiling comes out identical, which is the check that the reasoning holds.
"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from campaign import deep  # noqa: E402

SEEDS = (0, 1, 2)          # the seed axis is established; three suffice here
SCALES = (3, 4)            # where the front has curvature


def main(workers: int = 5) -> int:
    out = ROOT / "data" / "patterns"
    jobs = [dict(root=str(ROOT / "simulator"),
                 sys_path=[str(ROOT), str(ROOT / "simulator")],
                 scale=sc, day=d, search_seed=s, pattern=pat,
                 n_real=deep.N_REALISATIONS, n_eval=deep.N_EVAL,
                 population=deep.POPULATION,
                 out=str(out / "cells" / f"{pat}_x{sc}_d{d}_s{s}.json"))
            for pat in deep.PATTERNS for sc in SCALES
            for d in range(deep.N_DAYS) for s in SEEDS]
    print(f"{len(jobs)} cells", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(deep.run_cell, jobs), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["pattern"], r["scale"], r["day"], r["search_seed"]))
    with (out / "patterns.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out / 'patterns.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
