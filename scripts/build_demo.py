"""Build the LeakLocate deck: inline demo data + ladder + diagnosis into demo/deck_template.html.

Output: reports/leaklocate_demo.html (the published artifact).
"""
import json
import pathlib

import numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
rep = root / "reports"
data = json.loads((rep / "demo_data.json").read_text())
data["ladder"] = json.loads((rep / "ladder.json").read_text())
data["diagnosis"] = json.loads((rep / "diagnosis.json").read_text())
data["hybrid"] = json.loads((rep / "hybrid.json").read_text())
data["forest"] = json.loads((rep / "forest.json").read_text())
mm = json.loads((rep / "mismatch.json").read_text())
data["mismatch"] = {k: mm[k] for k in ["summary", "trust_check", "recalibrated"]}
data["mismatch"]["rows"] = [r for r in mm["rows"] if r["obs"] == "differential" and r["model"] == "Hybrid, recalibrated"]
# Multi-city study. Net2 is excluded: its junction demands are net inflows (mean demand -0.2 L/s),
# so demand-relative leak sizes are ~0 and the study is undefined for it.
cities = []
for f in ["multicity.json", "multicity_extra.json"]:
    if (rep / f).exists():
        cities += [c for c in json.loads((rep / f).read_text()) if c["name"] != "Net2"]
data["cities"] = cities
n = np.array(data["noise"])
data["noise_std_raw_m"] = float(np.median(n.std(axis=0)))
data["noise_std_cm_removed_m"] = float(np.median((n - n.mean(axis=1, keepdims=True)).std(axis=0)))
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = (root / "demo" / "deck_template.html").read_text(encoding="utf-8").replace("__DATA__", payload)
out = rep / "leaklocate_demo.html"
out.write_text(html, encoding="utf-8")
print(out, round(out.stat().st_size / 1e6, 2), "MB")
