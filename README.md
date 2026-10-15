# LeakLocate

Plan pressure sensors for a city water network, then point the repair crew to a small, reliable search area when a leak appears.

## Key gap

- Cities lose 25–30 % of treated water to leaks (World Bank estimates); small leaks hide inside normal demand swings.
- Leak *detection* is largely solved in simulation; *where to put sensors* and *how much pipe to search, and how sure*, are not.

## Fix

| Metric (held-out simulation) | Current best | This work |
|---|---|---|
| Distance from the crew's first stop to the leak | 2.8 km (best constant guess) | **1.0 km** (hybrid) |
| 80 % search areas that really contain the leak | 68 % (forest alone) | **81 %** (hybrid) |
| Junctions to search in the 80 % area | 37.4 (physics model) | **28.7** (hybrid) |
| Expected search distance, 6 sensors on Net3 | 2.9 km (random layout) | **2.2 km** (planned layout) |

- **Part 1 · sensor plan:** every junction of six benchmark networks simulated as a leak; greedy placement; a recommended sensor count per city.
- **Part 2 · find the leak:** a random forest fused with a hydraulic physics model (log-linear pooling, weights fitted on separate leaks).
- **Crew rule:** go to the junction with the shortest expected pipe distance, not the single most likely one.
- **Benchmark ladder:** best-guess baselines → tree ensembles → physics → hybrid, all on the same 4,000 held-out leaks.
- **Wrong pipe data:** 35 simulated cities whose hydraulic model has 10–30 % wrong roughness and/or base demands.

## When the hydraulic model is wrong

Real models never have exact pipe roughness or demands. Each test city below has a fixed, unknown model error; 1,000 leaks per city, 5 cities per level.

| Model error | Hybrid vs model (old) | vs leak-free baseline, recalibrated | Median error | Worst city |
|---|---|---|---|---|
| None | 80 % | 86 % | 968 m | 86 % |
| Roughness 10 % / 20 % / 30 % | 5 % / 2 % / 2 % | 85 % / 83 % / 67 % | 1.1 / 1.1 / 1.7 km | 76 % / 71 % / 17 % |
| Demand 10 % / 20 % / 30 % | 14 % / 6 % / 3 % | 86 % / 83 % / 83 % | 1.0 / 1.1 / 1.1 km | 85 % / 77 % / 80 % |
| Both 20 % | 3 % | 80 % | 1.2 km | 65 % |

Percentages: how often the 80 % search area held the leak (averaged over cities).

- **Against the model alone, locators fail:** 10 % roughness error creates 0.4–2 m of false residual; the largest leak (16 L/s) makes about 0.3 m.
- **Fix 1:** subtract the mean residual of 14 leak-free days in the same city; static model error cancels.
- **Fix 2:** refit the pooling weights on 5 separate wrong-data cities (one level: 20 % + 20 %), which restores calibration at the other levels too.
- **Trust check a utility can run:** if the leak-free baseline residual stays under 4 m at every sensor, coverage was at least 75 % in all 32 such test cities; the 4 cities above 4 m fell to 17–71 %. Above 4 m, recalibrate the hydraulic model before trusting search areas.
- Under wrong pipe data the hybrid no longer beats a recalibrated forest alone (83 % vs 82 % coverage, 34 vs 34 junctions at 20 % roughness).

## Business value

Narrows a district-level leak alarm to a few streets before acoustic crews pinpoint; uses loggers and hydraulic models utilities already have.

## Quick start

```bash
uv sync --dev
uv run pytest -q                                                     # simulator tests
uv run python -m leaklocate.generate --network Net3 --n 2000 --out data/net3   # leak scenarios
```

## Reproduce the results

One command rebuilds every number and the interactive report from nothing (about 20 minutes on a 10-core laptop):

```bash
bash scripts/reproduce.sh                # add SKIP_CITIES=1 to keep the committed multi-city results
```

It runs these steps in order:

| Step | Command | Output |
|---|---|---|
| Simulated scenarios | `bash scripts/make_data.sh` | `data/` (ladder, diagnosis, mismatch and calibration cities) |
| Signatures, noise, distances, greedy sensors | `python -m leaklocate.export_demo --network Net3 --max-sensors 12 --out reports/demo_data.json` | `reports/demo_data.json` |
| Residual maps | `python scripts/checkpoint1.py data/net3_v2 reports/figures` | `reports/figures/` |
| Pump-switching diagnosis | `python scripts/diagnosis_extract.py data/net3_v1 data/net3_v2 reports/diagnosis.json` | `reports/diagnosis.json` |
| Benchmark ladder | `python scripts/ladder.py data/ladder_train data/ladder_test reports/demo_data.json reports/ladder.json` | `reports/ladder.json` |
| Hybrid + calibration | `python scripts/hybrid.py data/ladder_train data/ladder_test reports/demo_data.json reports/ladder.json reports/hybrid.json` | `reports/hybrid.json` (adds rows to `ladder.json`) |
| Browser forest | `python scripts/export_forest.py data/ladder_train data/ladder_test reports/demo_data.json reports/hybrid.json reports/forest.json` | `reports/forest.json` |
| Wrong pipe data | `python scripts/mismatch.py data/ladder_train data/ladder_test reports/demo_data.json reports/hybrid.json data/mismatch data/mismatch_calib reports/mismatch.json` | `reports/mismatch.json` |
| City sensor planner | `python scripts/multicity.py reports/multicity.json Anytown Hanoi Net2 Net3 Balerma Ltown` and `... reports/multicity_extra.json ky4` | `reports/multicity*.json` |
| Interactive report | `python scripts/build_demo.py` | `reports/leaklocate_demo.html` |

All seeds are fixed. A run from a clean clone on a different machine reproduced the physics, diagnosis and city-planner results exactly. Forest-based results move slightly because EPANET output differs in the last digits between machines: ladder and hybrid rows within 1.1 points of coverage and 13 m of median error, single cities in the failure cases by up to about 150 m.

## Data and licences

- Networks: EPANET example networks (WNTR) and the EPyT-Flow benchmark collection; check each network's licence before redistribution.
- All leaks are simulated; no utility data is included.

## Limits

- Simulation only. Wrong pipe data is tested on simulated cities, not real ones.
- Needs a leak-free baseline period; a leak already present during the baseline would be partly subtracted away.
- Roughness errors near 30 % break some cities; the trust check flags them but cannot fix them.
- One leak at a time; no sensor faults.
- Net2 excluded (net-inflow demands); Richmond not yet working.
- Next: sensor dropout and two leaks, then field data.

Model details: [`MODEL_CARD.md`](MODEL_CARD.md). Licence: MIT.
