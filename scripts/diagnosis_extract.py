"""Numbers for the report's diagnosis slide, from the two generated batches.

v1 = rule-based pump controls; v2 = SCADA schedule (controls frozen). Same scenario seeds in both.
Usage: python scripts/diagnosis_extract.py data/net3_v1 data/net3_v2 reports/diagnosis.json
"""
import json
import sys

import numpy as np
import pandas as pd

v1, v2, out = sys.argv[1:4]
R1, R2 = np.load(f"{v1}/residuals.npy"), np.load(f"{v2}/residuals.npy")
df = pd.read_parquet(f"{v1}/scenarios.parquet")
nodes = json.load(open(f"{v1}/meta.json"))["nodes"]
noleak = df["leak_node"].isna().to_numpy()
big1 = np.abs(R1[noleak]).max(axis=(1, 2))
big2 = np.abs(R2[noleak]).max(axis=(1, 2))
k = int(np.where(noleak)[0][np.argmax(big1)])  # worst leak-free scenario under rule-based controls
j = int(np.abs(R1[k]).max(axis=0).argmax())
res = {
    "scenario": k, "node": nodes[j],
    "hourly_before_m": np.round(R1[k][:, j], 3).tolist(),
    "hourly_after_m": np.round(R2[k][:, j], 3).tolist(),
    "noleak_runs": int(noleak.sum()),
    "noleak_over_10m_before": int((big1 > 10).sum()), "noleak_over_10m_after": int((big2 > 10).sum()),
    "max_abs_before_m": float(big1.max()), "max_abs_after_m": float(big2.max()),
}
print({k_: v for k_, v in res.items() if not k_.startswith("hourly")})
json.dump(res, open(out, "w"))
