# Model card: LeakLocate hybrid locator

## Model
- **Physics matcher:** Bayesian matching of 24 h mean pressure residuals against simulated leak signatures, with an empirical noise covariance.
- **Random forest:** 400 trees, `min_samples_leaf=3`, on 24 h mean residuals at the sensors (common mode removed, plus the common mode itself).
- **Hybrid:** log-linear pooling of the two posteriors, P ∝ P_forest^0.5 · P_physics^0.4. The weights were fitted on 4,000 validation leaks; the forest was fit on 16,000 (`src/leaklocate/fusion.py`, `scripts/hybrid.py`).
- **Crew rule:** go first to the junction with the minimum expected pipe distance under the posterior.
- A compact 40-tree forest runs in the browser demo (parity with scikit-learn 1e-5; median 1,049 m against 1,018 m for 400 trees).

## Intended use
- Narrowing a district-level leak alarm to a ranked search area before acoustic crews pinpoint.
- Planning how many pressure loggers to install, and where.
- Research and portfolio use. Not validated on any real utility.

## Data
- All leaks are simulated with WNTR/EPANET on Net3 (pressure-driven demand, emitter leaks of 0.5–16 L/s, 5 % demand noise, 0.05 m sensor noise, SCADA pump schedule frozen to nominal).
- 20,000 training and 4,000 held-out test scenarios with different seeds (`scripts/make_data.sh`).
- 6 pressure sensors placed greedily.

## Results (4,000 held-out leaks, 6 sensors)
| Model | Median error | 80 % region holds the leak | Region size (junctions) |
|---|---|---|---|
| Constant network centre | 2.8 km | — | — |
| Random forest | 963 m | 68 % | 21.1 |
| Physics matcher | 1,117 m | 86 % | 37.4 |
| Forest, temperature-calibrated | 1,040 m | 82 % | 29.8 |
| **Hybrid** | **1,018 m** | **81 %** | **28.7** |

Against the temperature-calibrated forest (paired bootstrap): expected search distance −151 m [−163, −140]. Source: `reports/hybrid.json`.

## Results with a wrong hydraulic model (`reports/mismatch.json`)
- 35 simulated cities with fixed roughness and/or base-demand errors of 10–30 %, 1,000 leaks each.
- Absolute residuals (field − model) fail: 80 % areas held the leak 2–14 % of the time.
- Deployed form: residual relative to the mean of 14 leak-free days in the same city, and pooling weights refitted on 5 separate wrong-data cities: P ∝ P_forest^0.25 · P_physics^0.4.
- Average coverage of 80 % areas: 83–86 % for roughness error up to 20 % and demand error up to 30 %; 80 % with both at 20 %; 67 % at 30 % roughness.
- Trust check: leak-free baseline residual under 4 m at every sensor → worst city 76 %; above 4 m → 17–71 %.

## Limits and risks
- Simulation only. Model mismatch is tested on simulated cities only.
- Requires a leak-free baseline period; above a 4 m baseline residual, recalibrate the hydraulic model first.
- Under model mismatch the hybrid is no better than a recalibrated forest alone.
- One leak at a time; no sensor dropout.
- One network for the locator (Net3); the sensor planner covers six. Net2 and Richmond are excluded.
- Errors are pipe distances on the network graph, not street distances.
