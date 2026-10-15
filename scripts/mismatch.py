"""Model-mismatch study: how the locators hold up when the utility's hydraulic model is wrong.

Models are trained and fused exactly as in scripts/hybrid.py, on clean simulations (the model is right).
Each test city has wrong pipe roughness and/or wrong base demands, fixed for the whole city (data/mismatch/<level>_c<k>).
Two observation types:
  absolute      residual = field - model                         (what the ladder used so far)
  differential  residual = field - model - mean leak-free residual of 14 baseline days in the same city
The differential residual cancels static model error to first order; the leak shows up as a change from the baseline.
"Recalibrated" models refit the pooling weights (a, b) on separate calibration cities (data/mismatch_calib: one level,
20 % roughness + 20 % demand error, different city seeds and leaks) using differential residuals; a or b may be 0.

Usage: python scripts/mismatch.py data/ladder_train data/ladder_test reports/demo_data.json reports/hybrid.json data/mismatch data/mismatch_calib reports/mismatch.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from leaklocate.export_demo import posterior
from leaklocate.fusion import full_proba, nll, pool

tr_dir, te_dir, demo_path, hybrid_path, mm_dir, calib_dir, out_path = sys.argv[1:8]
with open(demo_path) as f:
    demo = json.load(f)
nodes = demo["nodes"]
N = len(nodes)
D = np.array(demo["dist_m"], float)
sig, noise = np.array(demo["signature"]), np.array(demo["noise"])
sensors = sorted(nodes.index(n) for n in demo["greedy_order"][:6])


def load(d):
    R = np.load(f"{d}/residuals.npy")[:, :, sensors]
    df = pd.read_parquet(f"{d}/scenarios.parquet")
    return R, np.array([nodes.index(n) for n in df["leak_node"]]), df["leak_lps"].to_numpy()


def feats(R):
    m = R.mean(axis=1)
    return np.hstack([m - m.mean(axis=1, keepdims=True), m.mean(axis=1, keepdims=True)])


# Physics posterior, batched. Same maths as export_demo.posterior (checked below).
_X = noise[:300][:, sensors]
_X = _X - _X.mean(axis=1, keepdims=True)
COV_INV = np.linalg.inv(np.cov(_X, rowvar=False) + 1e-3 * np.eye(len(sensors)))
_MC = sig[:, :, sensors] - sig[:, :, sensors].mean(axis=2, keepdims=True)


def phys_post(R):
    Y = R.mean(axis=1)
    out = np.empty((len(Y), N))
    for s in range(0, len(Y), 250):
        Yc = Y[s:s + 250] - Y[s:s + 250].mean(axis=1, keepdims=True)
        d = Yc[:, None, None, :] - _MC[None]
        ll = -0.5 * np.einsum("bnzi,ij,bnzj->bnz", d, COV_INV, d)
        ll -= ll.max(axis=(1, 2), keepdims=True)
        p = np.exp(ll).sum(axis=2)
        out[s:s + 250] = p / p.sum(axis=1, keepdims=True)
    return out


def metrics(P, y):
    pick = (P @ D).argmin(axis=1)  # minimum expected pipe distance
    err = D[y, pick]
    order = np.argsort(-P, axis=1)
    k = (np.take_along_axis(P, order, axis=1).cumsum(axis=1) < 0.8).sum(axis=1) + 1
    rank = np.argsort(order, axis=1)[np.arange(len(y)), y]
    return {"median_err_m": float(np.median(err)), "coverage80_%": 100 * float((rank < k).mean()), "region_size": float(k.mean()),
            "nll": nll(P, y)}


t0 = time.time()
Rtr, ytr, _ = load(tr_dir)
idx = np.random.default_rng(0).permutation(len(ytr))
fit_i, val_i = idx[:16000], idx[16000:]  # same split as hybrid.py
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(feats(Rtr[fit_i]), ytr[fit_i])
yv = ytr[val_i]
Pf_val, Pp_val = full_proba(rf, feats(Rtr[val_i]), N), phys_post(Rtr[val_i])
assert np.allclose(Pp_val[:20], [posterior(r, sig, noise[:300], sensors) for r in Rtr[val_i][:20].mean(axis=1)], atol=1e-8)
grid = np.round(np.arange(0.1, 2.01, 0.1), 2)
a_temp = float(min(grid, key=lambda a: nll(pool(Pf_val, Pp_val, a, 0.0), yv)))
ab = min(((a, b) for a in grid for b in grid), key=lambda ab: nll(pool(Pf_val, Pp_val, *ab), yv))
print(f"fitted in {time.time() - t0:.0f} s | temperature a = {a_temp} | fusion (a, b) = {ab}")




def differential(d):
    R, y, _ = load(d)
    base = np.load(f"{d}_base/residuals.npy")[:, :, sensors].mean(axis=0)  # [T, sensors] leak-free profile of the same city
    return R, R - base[None], y, baseline_size(base)


def baseline_size(base):
    """Trust check a utility can run before any leak: largest 24 h mean leak-free residual at a sensor, common mode removed (m)."""
    m = base.mean(axis=0)
    return float(np.abs(m - m.mean()).max())


# Recalibration on mismatched calibration cities (differential residuals).
cal = [differential(f"{calib_dir}/{n}") for n in sorted(os.listdir(calib_dir)) if not n.endswith("_base")]
yc = np.concatenate([c[2] for c in cal])
Rc = np.concatenate([c[1] for c in cal])
Pf_c, Pp_c = full_proba(rf, feats(Rc), N), phys_post(Rc)
grid0 = np.round(np.arange(0.0, 2.01, 0.05), 2)
rc_phys = float(min(grid0[1:], key=lambda b: nll(pool(Pf_c, Pp_c, 0.0, b), yc)))
rc_forest = float(min(grid0[1:], key=lambda a: nll(pool(Pf_c, Pp_c, a, 0.0), yc)))
rc_ab = min(((a, b) for a in grid0 for b in grid0 if a + b > 0), key=lambda ab: nll(pool(Pf_c, Pp_c, *ab), yc))
rc_ab = (float(rc_ab[0]), float(rc_ab[1]))
print(f"recalibrated on {len(yc)} leaks in {len(cal)} cities: physics b = {rc_phys} | forest a = {rc_forest} | hybrid (a, b) = {rc_ab}")


def evaluate(R, y):
    Pf, Pp = full_proba(rf, feats(R), N), phys_post(R)
    return {"Physics": metrics(Pp, y), "Forest (calibrated)": metrics(pool(Pf, Pp, a_temp, 0.0), y), "Hybrid": metrics(pool(Pf, Pp, *ab), y),
            "Physics, recalibrated": metrics(pool(Pf, Pp, 0.0, rc_phys), y), "Forest, recalibrated": metrics(pool(Pf, Pp, rc_forest, 0.0), y),
            "Hybrid, recalibrated": metrics(pool(Pf, Pp, *rc_ab), y)}


# Sanity check against hybrid.json on the original clean test set.
Rte, yte, _ = load(te_dir)
check = evaluate(Rte, yte)
with open(hybrid_path) as f:
    ref = {r["model"]: r for r in json.load(f)["rows"]}["Hybrid: forest × physics (fitted pooling)"]
print("clean ladder test, hybrid: median", check["Hybrid"]["median_err_m"], "vs", ref["median_err_m"], "| coverage",
      round(check["Hybrid"]["coverage80_%"], 2), "vs", ref["coverage80_%"])

rows = []
for name in sorted(n for n in os.listdir(mm_dir) if not n.endswith("_base")):
    level, city = name.rsplit("_c", 1)
    R, Rd, y, bsize = differential(f"{mm_dir}/{name}")
    with open(f"{mm_dir}/{name}/meta.json") as f:
        meta = json.load(f)
    for obs, Ro in [("absolute", R), ("differential", Rd)]:
        for model, m in evaluate(Ro, y).items():
            rows.append({"level": level, "roughness_sigma": meta["roughness_sigma"], "demand_bias_sigma": meta.get("demand_bias_sigma", 0.0),
                         "city": int(city), "baseline_m": bsize, "obs": obs, "model": model, **m})
    print(f"{name} done ({time.time() - t0:.0f} s)", flush=True)

df = pd.DataFrame(rows)
order = ["clean", "r10", "r20", "r30", "d10", "d20", "d30", "r20d20"]
summary = (df.groupby(["level", "obs", "model"], sort=False)
             .agg(median_err_m=("median_err_m", "mean"), coverage80=("coverage80_%", "mean"), coverage80_min=("coverage80_%", "min"),
                  region_size=("region_size", "mean"), cities=("city", "count")).reset_index())
summary["level"] = pd.Categorical(summary["level"], order, ordered=True)
summary = summary.sort_values(["level", "obs", "model"])
print(summary.round(1).to_string(index=False))
# Trust check: does the leak-free baseline residual predict which cities the recalibrated hybrid fails on?
hr = df[(df.obs == "differential") & (df.model == "Hybrid, recalibrated")]
trust = []
for lo, hi in [(0, 2), (2, 4), (4, np.inf)]:
    g = hr[(hr.baseline_m >= lo) & (hr.baseline_m < hi)]
    trust.append({"baseline_m": [lo, None if np.isinf(hi) else hi], "cities": len(g), "coverage80_min": float(g["coverage80_%"].min()),
                  "coverage80_mean": float(g["coverage80_%"].mean()), "median_err_m_max": float(g["median_err_m"].max())})
rho = float(hr[["baseline_m", "coverage80_%"]].corr(method="spearman").iloc[0, 1])
print("trust check (recalibrated hybrid):", trust, "| Spearman rho =", round(rho, 2))
with open(out_path, "w") as f:
    json.dump({"a_temp": a_temp, "fusion_ab": [float(ab[0]), float(ab[1])],
               "recalibrated": {"physics_b": rc_phys, "forest_a": rc_forest, "hybrid_ab": list(rc_ab), "calib_leaks": int(len(yc))}, "sensors": [nodes[s] for s in sensors], "baseline_days": 14,
               "leaks_per_city": 1000, "clean_check": check, "trust_check": {"bands": trust, "spearman_rho": rho}, "rows": rows, "summary": summary.astype({"level": str}).to_dict("records")}, f, indent=1)
print(f"wrote {out_path} ({time.time() - t0:.0f} s)")
