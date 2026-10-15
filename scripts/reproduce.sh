#!/bin/bash
# Rebuild every number and the interactive report from scratch, in dependency order.
#   bash scripts/reproduce.sh               # everything (about 20 min on a 10-core laptop)
#   SKIP_CITIES=1 bash scripts/reproduce.sh # keep the committed multi-city results (saves about 7 min)
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $*"; uv run python "$@"; }

bash scripts/make_data.sh                                                       # all simulated scenarios -> data/
run -m leaklocate.export_demo --network Net3 --max-sensors 12 --out reports/demo_data.json  # signatures, noise, distances, greedy sensors
run scripts/checkpoint1.py data/net3_v2 reports/figures                          # residual maps, detectability (after the pump fix)
run scripts/diagnosis_extract.py data/net3_v1 data/net3_v2 reports/diagnosis.json
run scripts/ladder.py data/ladder_train data/ladder_test reports/demo_data.json reports/ladder.json
run scripts/hybrid.py data/ladder_train data/ladder_test reports/demo_data.json reports/ladder.json reports/hybrid.json
run scripts/export_forest.py data/ladder_train data/ladder_test reports/demo_data.json reports/hybrid.json reports/forest.json
run scripts/mismatch.py data/ladder_train data/ladder_test reports/demo_data.json reports/hybrid.json data/mismatch data/mismatch_calib reports/mismatch.json
if [ -z "${SKIP_CITIES:-}" ]; then
  run scripts/multicity.py reports/multicity.json Anytown Hanoi Net2 Net3 Balerma Ltown   # networks download into data/networks
  run scripts/multicity.py reports/multicity_extra.json ky4
fi
run scripts/build_demo.py                                                       # -> reports/leaklocate_demo.html
