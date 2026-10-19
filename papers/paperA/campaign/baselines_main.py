"""Driver: the two boundary solutions, 5 days x 3 demand scales."""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from campaign import deep, baselines  # noqa: E402

SCALES = ((1, "deep"), (3, "deep_x3"), (4, "deep_x4"))


def main(workers: int = 5) -> int:
    jobs = [dict(root=str(ROOT / "simulator"),
                 sys_path=[str(ROOT), str(ROOT / "simulator")],
                 scale=sc, day=d, n_real=deep.N_REALISATIONS,
                 n_eval=deep.N_EVAL, seeds=list(deep.SEARCH_SEEDS),
                 cells=str(ROOT / "data" / folder / "cells"))
            for sc, folder in SCALES for d in range(deep.N_DAYS)]
    print(f"{len(jobs)} groups", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(baselines.run_group, jobs), 1):
            rows.extend(got)
            print(f"  {i}/{len(jobs)}", flush=True)
    out = ROOT / "data" / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["scale"], r["day"], r["arm"]))
    with (out / "baselines.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out / 'baselines.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
