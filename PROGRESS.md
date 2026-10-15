# LeakLocate progress

**Demo deck:** built by `python scripts/build_demo.py` from `reports/demo_data.json`, which `python -m leaklocate.export_demo` writes.

## Done (2026-10-05)
- Step 0, prior art: EPyT-Flow, BattLeDIM, GNN locators, and placement methods. Gap statement is in `research/02_project_briefs.md` v0.2.
- Step 1, simulation harness:
  - `sim.py` (WNTR/EPANET, pressure-driven demand, emitter leaks, demand and roughness noise);
  - `generate.py` (parallel, 2,000 scenarios in about 9 s);
  - 4 passing tests (zero residual, determinism, mass balance, leak locality).
- Diagnosis:
  - Pump-switching confounder: 58 of 205 leak-free runs showed false residuals of 10–30 m. Fixed with the SCADA schedule (`_freeze_operations`).
  - Common-mode demand noise is 0.082 m; removing the common mode leaves 0.018 m.
- Baseline locator (Bayesian matching with empirical covariance) and greedy placement.
- Superposition check: median error 0.038 m.
- Interactive demo, tested headless with jsdom; no errors.

## Measured (Net3, 6 greedy sensors, 5 % demand noise, 400 random leaks of 0.5–16 L/s)
- Median best-guess error 1.5 km, against 5.1 km for a random guess.
- For 8–16 L/s leaks, the median error is 448 m.
- 80 % search regions held the leak 75 % of the time, so the locator is **overconfident**. This is the next thing to fix.

## Benchmark ladder (2026-10-05, held-out 4,000 full-EPANET leaks, 6 sensors, 5 % demand noise)
| Step | Model | Median | Mean | 80 % coverage |
|---|---|---|---|---|
| 0 | Constant network centre | 2.8 km | 3.4 km | — |
| 1 | Random forest (24 h means) | 1.28 km | 2.45 km | 70 % |
| 1b | Physics Bayesian matcher | 1.58 km | 3.97 km | 86 % |
| 2 | Random forest + min-expected-distance pick | **948 m** | **1.78 km** | 70 % |
| 2 | Physics + min-expected-distance pick | 1.12 km | 2.16 km | 86 % |

`scripts/ladder.py`; the deck reads `reports/ladder.json`.
- Deck v0.2: Tour, Report, Evidence and Roadmap, 11 slides. Built by `scripts/build_demo.py` from `demo/deck_template.html`.

## Next (as of 2026-10-05; superseded below)
1. Calibrate the forest with temperature scaling, or blend its guess with the physics regions. Then calibration of the physics model: inflate the covariance for model and superposition error, or use a finer size grid. Re-measure coverage.
2. Step 3: graph-neural-network locator against this baseline, with splits by region.
3. Step 5: roughness mismatch, sensor dropout, two leaks at once.
4. L-Town (782 junctions) through EPyT-Flow.
5. Repo hygiene before the first commit: README, pre-commit, CI, licence (MIT), then user review → push.

## v0.3 (2026-10-05): hybrid + company section
- `scripts/hybrid.py`: log-linear pooling of forest and physics posteriors, weights a=0.5 and b=0.4 fitted on 4,000 validation leaks (forest fit on 16k).
  - Test results: median 1,018 m; 80 % coverage 81 %; region 28.7 junctions, against physics 37.4 (23 % smaller).
  - Against a temperature-scaled forest (paired bootstrap): expected distance −151 m [−163, −140]; region −1.1 junctions [−1.2, −1.06]; NLL −0.06.
- Deck: 17 slides in Tour · Report · What's new (hybrid trio, frontier, reliability, decision rule) · Evidence · Roadmap & company (fit in the value chain, build-out + value calculator, gaps).
- Open: put the hybrid on the live platform (export a compact forest); GNN inside the pooling.

## v0.5 (2026-10-05): live hybrid + city planner + quantitative titles
- Live Physics/Forest/Hybrid switch: compact 40-tree forest in the browser (`scripts/export_forest.py`). Parity with sklearn is 1e-5. Hybrid median 1,049 m vs 1,018 m for the 400-tree forest; coverage 81 %.
- City planner (`scripts/multicity.py`): every junction simulated as a leak; leak sizes 0.25–2 % of each city's demand; greedy placement on MC expected distance.
  - Cities: Anytown, Hanoi, Net3, Balerma, L-Town, KY4.
  - Excluded: Net2 (net-inflow demands, mean −0.2 L/s); Richmond (setup error, not yet investigated).
  - With 8 sensors the error spans 9 % (Balerma) to 55 % (L-Town) of a random guess; Spearman ρ = 0.60 vs loops per 100 junctions.
- All titles state the measured result, computed from data.

## v0.6 (2026-10-06): wrong pipe data
- Simulator: persistent model error per city (`mismatch_seed`, `demand_bias_sigma`), plus a test.
- Data: 7 mismatch levels × 5 cities × 1,000 leaks, a clean control, 14 leak-free baseline days per city, and 5 separate calibration cities (`scripts/make_data.sh`).
- `scripts/mismatch.py` → `reports/mismatch.json`; it re-scores the clean test set first (hybrid 1,015 m / 81.1 % vs 1,018 m / 81.0 % in `hybrid.json`).
- Findings:
  - Absolute residuals collapse under any mismatch: 80 % areas hold the leak 2–14 % of the time.
  - Baseline-relative residuals + recalibrated weights: 83–86 % coverage for roughness ≤ 20 % or demand ≤ 30 %; 80 % at 20 % + 20 %; 67 % at 30 % roughness.
  - Per-city spread matters: the worst city at 20 % + 20 % is 65 %.
  - Trust check: a baseline residual under 4 m kept every city at ≥ 75 %; the 4 cities above 4 m fell to 17–71 % (Spearman ρ = −0.61 over 36 cities, including the clean one).
  - The hybrid's edge over the forest disappears under mismatch.
- Deck: new Evidence slide "Wrong pipe data"; Limits slide updated. jsdom check: no errors.

## Reproducibility (2026-10-07)
- `scripts/reproduce.sh` rebuilds everything from a clean clone in 15.5 min on a 10-core Mac; README lists every step.
- New `--rule-controls` flag regenerates the pre-fix diagnosis batch; it reproduces `diagnosis.json` exactly (58 of 205).
- Clean-clone check against the committed reports (made on the Windows PC):
  - identical: diagnosis, checkpoint table, physics rows, multi-city plans (all 7 networks), first 12 greedy sensors;
  - forest-based rows: within 1.1 points and 13 m (ladder, hybrid; largest is gradient boosting); recalibrated hybrid under mismatch within 0.2 points and 19 m per city.
- Fixed while checking: the checkpoint figures come from `net3_v2` (not v1), and the demo was built with `--max-sensors 12`.

## Next (updated)
1. Sensor dropout and two leaks at once.
2. Graph neural network locator, judged by the ladder and the mismatch study.
3. L-Town (782 junctions) through EPyT-Flow.
4. CLI and the push.
