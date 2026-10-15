"""Step 2 innovation: fuse the forest (sharp, overconfident) with physics (broad, honest) into one posterior.

  log p_hybrid  =  a * log p_forest  +  b * log p_physics   (log-linear pooling, then renormalize)
  a, b are fitted on a validation split by maximum log-likelihood of the true leak; a 'temperature'
  variant of the forest alone (a only) is the standard calibration baseline.
  Decision: junction with the smallest expected pipe distance; 80 % region from the fused posterior.

Usage: python scripts/hybrid.py data/ladder_train data/ladder_test reports/demo_data.json reports/ladder.json reports/hybrid.json
Appends rows to ladder.json and writes figure data (reliability curves, example maps) to hybrid.json.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from leaklocate.export_demo import posterior

tr_dir, te_dir, demo_path, ladder_path, out_path = sys.argv[1:6]
demo = json.load(open(demo_path))
nodes = demo["nodes"]
N = len(nodes)
D = np.array(demo["dist_m"], float)
sig, noise = np.array(demo["signature"]), np.array(demo["noise"])
sensors = sorted(nodes.index(n) for n in demo["greedy_order"][:6])
EPS = 1e-6


def load(d):
    R = np.load(f"{d}/residuals.npy")[:, :, sensors]
    df = pd.read_parquet(f"{d}/scenarios.parquet")
    return R, np.array([nodes.index(n) for n in df["leak_node"]]), df["leak_lps"].to_numpy()


def feats(R):
    m = R.mean(axis=1)
    return np.hstack([m - m.mean(axis=1, keepdims=True), m.mean(axis=1, keepdims=True)])


def full_proba(clf, X):
    P = np.zeros((len(X), N))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def phys_post(R):
    m = R.mean(axis=1)
    return np.array([posterior(m[i], sig, noise[:300], sensors) for i in range(len(m))])


def pool(Pf, Pp, a, b):
    L = a * np.log(Pf + EPS) + b * np.log(Pp + EPS)
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=1, keepdims=True)


def nll(P, y):
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def coverage_curve(P, y, levels):
    order = np.argsort(-P, axis=1)
    cum = np.take_along_axis(P, order, axis=1).cumsum(axis=1)
    rank = np.argsort(order, axis=1)[np.arange(len(y)), y]  # position of the true leak in the sorted list
    out = []
    for lv in levels:
        k = (cum < lv).sum(axis=1) + 1
        out.append({"level": lv, "coverage": float((rank < k).mean()), "size": float(k.mean())})
    return out


def metrics(P, y, q, pick):
    err = D[y, pick]
    cc = coverage_curve(P, y, [0.8])[0]
    bins = [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16.01)]
    return {"top1_%": 100 * float((pick == y).mean()), "median_err_m": float(np.median(err)), "mean_err_m": float(err.mean()),
            "expected_err_m": float((P * D[y]).sum(axis=1).mean()), "coverage80_%": 100 * cc["coverage"], "region_size": cc["size"],
            "nll": nll(P, y), "median_err_by_size_m": [float(np.median(err[(q >= a) & (q < b)])) for a, b in bins]}


t0 = time.time()
Rtr, ytr, _ = load(tr_dir)
Rte, yte, qte = load(te_dir)
idx = np.random.default_rng(0).permutation(len(ytr))
fit_i, val_i = idx[:16000], idx[16000:]  # validation split never seen by the forest
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(feats(Rtr[fit_i]), ytr[fit_i])
Pf_val, Pf_te = full_proba(rf, feats(Rtr[val_i])), full_proba(rf, feats(Rte))
Pp_val, Pp_te = phys_post(Rtr[val_i]), phys_post(Rte)
yv = ytr[val_i]
print(f"posteriors ready {time.time() - t0:.0f} s")

grid = np.round(np.arange(0.1, 2.01, 0.1), 2)
a_temp = min(grid, key=lambda a: nll(pool(Pf_val, Pp_val, a, 0.0), yv))
ab = min(((a, b) for a in grid for b in grid), key=lambda ab: nll(pool(Pf_val, Pp_val, *ab), yv))
print("temperature a =", a_temp, "| fusion (a, b) =", ab)

pick = lambda P: (P @ D).argmin(axis=1)  # noqa: E731  minimum expected pipe distance
models = {
    "Random forest (16k fit)": Pf_te,
    "Physics matcher": Pp_te,
    "Forest, temperature-calibrated": pool(Pf_te, Pp_te, a_temp, 0.0),
    "Hybrid: forest × physics (fitted pooling)": pool(Pf_te, Pp_te, *ab),
}
rows = []
for name, P in models.items():
    rows.append({"model": name, **metrics(P, yte, qte, pick(P))})
print(pd.DataFrame(rows).drop(columns=["median_err_by_size_m"]).round(2).to_string(index=False))

# Paired bootstrap: is the hybrid really better than the temperature-calibrated forest?
Pt, Ph0 = models["Forest, temperature-calibrated"], models["Hybrid: forest × physics (fitted pooling)"]
ll_diff = -np.log(Ph0[np.arange(len(yte)), yte] + 1e-12) + np.log(Pt[np.arange(len(yte)), yte] + 1e-12)
ed_diff = (Ph0 * D[yte]).sum(axis=1) - (Pt * D[yte]).sum(axis=1)


def region_k(P):
    return (np.sort(P, axis=1)[:, ::-1].cumsum(axis=1) < 0.8).sum(axis=1) + 1


rk_diff = region_k(Ph0) - region_k(Pt)
br = np.random.default_rng(5)
boot = {name: [] for name in ["nll", "expected_err_m", "region_size"]}
for _ in range(2000):
    s = br.integers(0, len(yte), len(yte))
    boot["nll"].append(ll_diff[s].mean()); boot["expected_err_m"].append(ed_diff[s].mean()); boot["region_size"].append(rk_diff[s].mean())
bootstrap = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in boot.items()}
print("hybrid minus calibrated forest (negative = hybrid better):", {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in bootstrap.items()})

levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
reliability = {name: coverage_curve(P, yte, levels) for name, P in models.items()}

# Example maps: test cases where the forest's 80 % region misses the leak but the hybrid's holds it.
Ph = models["Hybrid: forest × physics (fitted pooling)"]


def in_region(P, i):
    o = np.argsort(-P[i]); k = int((P[i][o].cumsum() < 0.8).sum() + 1)
    return yte[i] in o[:k], k


cands = [i for i in range(len(yte)) if not in_region(Pf_te, i)[0] and in_region(Ph, i)[0] and 2 <= qte[i] <= 8]
cands.sort(key=lambda i: D[yte[i], pick(Ph[i:i + 1])[0]])
examples = []
for i in cands[:3]:
    ex = {"leak": nodes[yte[i]], "leak_lps": float(qte[i])}
    for key, P in [("physics", Pp_te), ("forest", Pf_te), ("hybrid", Ph)]:
        ok, k = in_region(P, i)
        ex[key] = {"p": np.round(P[i], 4).tolist(), "pick": nodes[int(pick(P[i:i + 1])[0])], "region_k": k, "holds": bool(ok),
                   "err_m": float(D[yte[i], pick(P[i:i + 1])[0]])}
    examples.append(ex)
share = len(cands) / len(yte)

lad = json.load(open(ladder_path))
lad["rows"] = [r for r in lad["rows"] if r["step"] != "2h"]
for r in rows[2:]:
    lad["rows"].append({"step": "2h", **{k: v for k, v in r.items() if k != "nll"}})
json.dump(lad, open(ladder_path, "w"), indent=1)
json.dump({"a_temp": float(a_temp), "fusion_ab": [float(ab[0]), float(ab[1])], "val_n": int(len(val_i)), "fit_n": int(len(fit_i)),
           "rows": rows, "bootstrap_vs_temp": bootstrap, "reliability": reliability, "examples": examples, "rescued_share": share}, open(out_path, "w"))
print(f"rescued cases (forest region misses, hybrid holds, 2-8 L/s): {share:.1%} of test · total {time.time() - t0:.0f} s")
