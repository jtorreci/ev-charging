"""Driver for the experiment: 5 fleets x 5 search seeds."""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulator"))

from campaign import deep  # noqa: E402


def main(workers: int = 4) -> int:
    scale = int(sys.argv[2]) if len(sys.argv) > 2 else deep.DEMAND_SCALE
    out_dir = ROOT / "data" / ("deep" if scale == 1 else f"deep_x{scale}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [dict(root=str(ROOT / "simulator"),
                 sys_path=[str(ROOT), str(ROOT / "simulator")],
                 day=d, search_seed=s, n_real=deep.N_REALISATIONS,
                 n_eval=deep.N_EVAL, population=deep.POPULATION,
                 scale=scale,
                 out=str(out_dir / "cells" / f"d{d}_x{s}.json"))
            for d in range(deep.N_DAYS) for s in deep.SEARCH_SEEDS]

    print(f"{len(jobs)} cells, {workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(deep.run_cell, jobs), 1):
            rows.append(row)
            print(f"  {i}/{len(jobs)}  day {row['day']} seed "
                  f"{row['search_seed']}  {row['seconds']}s", flush=True)

    rows.sort(key=lambda r: (r["day"], r["search_seed"]))
    with (out_dir / "deep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "manifest.json").write_text(json.dumps({
        "sockets": deep.SOCKETS, "charger_kw": deep.CHARGER_KW,
        "bess_kwh": deep.BESS_KWH, "grid_kw": deep.GRID_KW,
        "pattern": deep.PATTERN, "n_days": deep.N_DAYS,
        "n_realisations": deep.N_REALISATIONS, "demand_scale": scale,
        "search_seeds": list(deep.SEARCH_SEEDS),
        "n_eval": deep.N_EVAL, "population": deep.POPULATION,
        "coverage_levels": list(deep.COVERAGE_LEVELS)}, indent=1))
    print(f"\nwrote {out_dir / 'deep.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 4))
