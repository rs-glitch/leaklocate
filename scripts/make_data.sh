#!/bin/bash
# Regenerate the simulated leak scenarios used by the benchmark ladder and hybrid (Net3, 5 % demand noise).
# Networks come from WNTR's bundled library; the multi-city planner downloads others via EPyT-Flow into data/networks.
set -e
uv run python -m leaklocate.generate --network Net3 --seed 2 --n 20000 --leak-max 16 --no-leak-frac 0 --out data/ladder_train
uv run python -m leaklocate.generate --network Net3 --seed 3 --n 4000 --leak-max 16 --no-leak-frac 0 --out data/ladder_test

# Pump-switching diagnosis: the same 2,000 scenarios with rule-based controls (v1) and with the SCADA schedule (v2).
uv run python -m leaklocate.generate --network Net3 --seed 1 --n 2000 --rule-controls --out data/net3_v1
uv run python -m leaklocate.generate --network Net3 --seed 1 --n 2000 --out data/net3_v2

# Model-mismatch study (scripts/mismatch.py): 5 'true cities' per mismatch level, the same 1,000 leaks in each,
# plus 14 leak-free baseline days per city for the differential residual.
uv run python -m leaklocate.generate --network Net3 --seed 4 --n 1000 --leak-max 16 --no-leak-frac 0 --out data/mismatch/clean_c1
uv run python -m leaklocate.generate --network Net3 --seed 5 --n 14 --no-leak-frac 1 --out data/mismatch/clean_c1_base
levels=("r10 0.1 0" "r20 0.2 0" "r30 0.3 0" "d10 0 0.1" "d20 0 0.2" "d30 0 0.3" "r20d20 0.2 0.2")
for lv in "${levels[@]}"; do
  read -r tag rs ds <<< "$lv"
  for city in 1 2 3 4 5; do
    out=data/mismatch/${tag}_c${city}
    [ -f $out/meta.json ] && continue
    uv run python -m leaklocate.generate --network Net3 --seed 4 --n 1000 --leak-max 16 --no-leak-frac 0 \
      --roughness-sigma $rs --demand-bias-sigma $ds --mismatch-seed $((100 + city)) --out $out
    uv run python -m leaklocate.generate --network Net3 --seed 5 --n 14 --no-leak-frac 1 \
      --roughness-sigma $rs --demand-bias-sigma $ds --mismatch-seed $((100 + city)) --out ${out}_base
  done
done

# Calibration cities for the mismatch study: one middle level (20 % roughness + 20 % demand), different city seeds and leaks.
for city in 1 2 3 4 5; do
  out=data/mismatch_calib/r20d20_c${city}
  [ -f $out/meta.json ] && continue
  uv run python -m leaklocate.generate --network Net3 --seed 6 --n 1000 --leak-max 16 --no-leak-frac 0 \
    --roughness-sigma 0.2 --demand-bias-sigma 0.2 --mismatch-seed $((200 + city)) --out $out
  uv run python -m leaklocate.generate --network Net3 --seed 7 --n 14 --no-leak-frac 1 \
    --roughness-sigma 0.2 --demand-bias-sigma 0.2 --mismatch-seed $((200 + city)) --out ${out}_base
done
