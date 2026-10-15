# LeakLocate decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-05 | Build on WNTR/EPyT-Flow instead of a custom simulator; contribution = mismatch-robust calibrated localization + sensor-budget decision curve | Prior-art check: simulators and benchmarks already exist; leak detection is largely solved in simulation |
| 2026-10-05 | The model gets the SCADA pump schedule (controls frozen to nominal) | 58 of 205 leak-free runs showed 10–30 m false residuals from control timing |
| 2026-10-05 | Remove the common-mode residual before matching | Common-mode demand noise is 0.082 m; 0.018 m remains after removal |
| 2026-10-05 | Benchmark ladder on one held-out set of 4,000 leaks with the same 6 greedy sensors | One results table, same split and metric for every model |
| 2026-10-05 | Crew goes to the minimum-expected-distance junction, not the most likely one | Cut the forest's median error from 1.28 km to 948 m |
| 2026-10-05 | Hybrid = log-linear pooling of forest and physics posteriors, weights fitted on separate validation leaks | Forest alone is overconfident (68–70 % coverage of 80 % regions); physics is calibrated but its regions are large |
| 2026-10-05 | Exclude Net2 and Richmond from the city planner | Net2 has net-inflow demands; Richmond fails to set up (not yet investigated) |
| 2026-10-05 | Interactive HTML demo is the primary showcase | User request |
| 2026-10-06 | Model mismatch is fixed per city (`--mismatch-seed`), not redrawn per scenario | A utility has one wrong model for all its leaks; per-scenario draws would average the error away |
| 2026-10-06 | Locate on residuals relative to a 14-day leak-free baseline, not the model alone | 10 % roughness error gives 0.4–2 m false residuals vs ≈0.3 m for a 16 L/s leak; absolute residuals gave 2–14 % coverage |
| 2026-10-06 | Refit pooling weights on separate wrong-data calibration cities (20 % + 20 %), tested at all levels | Clean-fitted weights under-covered (71–78 %); calibrating at one level and testing at others checks transfer |
| 2026-10-06 | Report the baseline-residual trust check (4 m) rather than claim robustness at 30 % error | It is observable before any leak and separated passing from failing cities in this study (36 cities, only 4 above the threshold: a small sample) |
