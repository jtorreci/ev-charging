# Scheduling Under a Booking System — replication artifact

Code and data for "Scheduling Under a Booking System: Quantile Objectives for
Electric-Vehicle Charging With Known Fleet and Unknown State of Charge"
(J.M. Ceballos-Martínez, C. Calvo-Jurado, J.-A. Torrecilla-Pinero), submitted
to Applied Energy. The release tagged `paperA-v1.0` is the version the
manuscript cites. Everything reported in the paper is produced
by the scripts here from the data in `data/`; nothing is computed outside them.

## Layout

    simulator/    the Monte Carlo evaluator and the fitted demand model
    campaign/     the experiments, one file per experiment
    figures/      the figure scripts
    tables/       the table script and its recorded output
    data/         the campaign outputs the manuscript reports (tracked, 1.8 MB)

## What produces what

| Manuscript element | Script | Data |
|---|---|---|
| Table 1, attainable coverage against occupancy | `campaign/deep.py` → `tables/paper_a_tables.py` | `data/deep`, `data/deep_x3`, `data/deep_x4` |
| Table 2, coverage under a price cap | `campaign/deep.py` → `tables/paper_a_tables.py` | as above |
| Search-seed stability | `tables/paper_a_tables.py` | as above |
| Boundary solutions | `campaign/baselines.py` → `tables/paper_a_tables.py` | `data/baselines` |
| Out-of-sample optimism | `campaign/optimism.py` → `tables/paper_a_tables.py` | `data/optimism` |
| Optimism along the front | `campaign/front_optimism.py` → `tables/paper_a_tables.py` | `data/front_optimism` |
| Figures 1–3 | `figures/paper_a.py` | the above |

Each campaign folder holds the summary CSV and, for the deep campaign, a
`manifest.json` with the configuration and one `cells/d{day}_x{seed}.json` per
run carrying the full Pareto front with its schedules.

## Inputs

The campaigns read three fitted inputs, all tracked:

    simulator/Data/output/distribution_params_{Slow,Normal,Fast}.json
                                      energy distribution per charge type
    simulator/data_model/output/distribution_params_{Slow,Normal,Fast}.json
                                      the same files, at the path the fleet
                                      generator (`StatAnalisys2`) reads
    simulator/Files/PreciosPeriodo.csv
                                      the 3.0 TD tariff by period

They are byte-identical to the files the published runs read.

The raw charging-session records the energy distributions were fitted to are
not included. `simulator/data_model/` holds the fitting code, which reads them
from `*.csv` files in its own directory.

## Reproducing

    pip install numpy scipy pandas matplotlib platypus-opt holidays

All commands run from this directory. Regenerating the figures from the data
already present takes seconds; they are written to `output/`:

    python3 figures/paper_a.py

Regenerating Tables 1 and 2 and every summary number quoted in the text
(boundary-solution margins, search-seed stability, out-of-sample optimism,
optimism along the front) takes under a second:

    python3 tables/paper_a_tables.py

Its output is recorded in `tables/paper_a_tables.out`; a rerun reproduces it
byte for byte. Each block of the output names the section or table it serves.

Re-running a campaign is slower. One optimisation takes 57 s at the median and
207 s at the worst on one core, and the main campaign is 75 of them:

    python3 -m campaign.deep_main

Every campaign pins its search seed, so a rerun reproduces the published
numbers exactly rather than approximately.

## Two properties of the design worth knowing before reading the code

**The fleet is fixed across the ensemble.** A booking reveals which vehicles
arrive, so their power and capacity are data; only the requested energy varies
between realisations. `build_day` draws the fleet once, from a stream seeded by
the composition alone, and then draws one energy per vehicle per realisation.

**The arrival state is derived, not drawn.** The fitted distribution is on
energy, measured from real charging sessions, so energy is drawn and the state
of charge follows from it. Drawing both would let the capacity draw truncate
the fitted energy distribution, and the distribution fitted to data would not
be the one the simulation faces.
