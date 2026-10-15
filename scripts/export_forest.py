"""Export a compact random forest and refitted pooling weights for the in-browser hybrid.

Trees are flat arrays; each leaf keeps its class distribution sparsely. The compact model's test
metrics are reported next to the 400-tree hybrid so the page can state what compression costs.

Usage: python scripts/export_forest.py data/ladder_train data/ladder_test reports/demo_data.json reports/hybrid.json reports/forest.json
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from leaklocate.export_demo import posterior
from leaklocate.fusion import feats, full_proba, load, nll, pool, region_stats

tr_dir, te_dir, demo_path, hybrid_path, out_path = sys.argv[1:6]
demo = json.load(open(demo_path))
nodes = demo["nodes"]; N = len(nodes)
D = np.array(demo["dist_m"], float)
sig, noise = np.array(demo["signature"]), np.array(demo["noise"])
sensors = sorted(nodes.index(n) for n in demo["greedy_order"][:6])
Rtr, ytr, _ = load(tr_dir, nodes, sensors)
Rte, yte, qte = load(te_dir, nodes, sensors)
idx = np.random.default_rng(0).permutation(len(ytr)); fit_i, val_i = idx[:16000], idx[16000:]  # same split as hybrid.py
phys = lambda R: np.array([posterior(m, sig, noise[:300], sensors) for m in R.mean(axis=1)])  # noqa: E731
Pp_val, Pp_te = phys(Rtr[val_i]), phys(Rte)
grid = np.round(np.arange(0.1, 2.01, 0.1), 2)
pick = lambda P: (P @ D).argmin(axis=1)  # noqa: E731

chosen = None
for n_est, leaf in [(40, 15), (30, 25), (24, 40)]:
    rf = RandomForestClassifier(n_estimators=n_est, min_samples_leaf=leaf, n_jobs=-1, random_state=0).fit(feats(Rtr[fit_i]), ytr[fit_i])
    Pf_val, Pf_te = full_proba(rf, feats(Rtr[val_i]), N), full_proba(rf, feats(Rte), N)
    ab = min(((a, b) for a in grid for b in grid), key=lambda ab: nll(pool(Pf_val, Pp_val, *ab), ytr[val_i]))
    Ph = pool(Pf_te, Pp_te, *ab)
    leaves = sum(int((e.tree_.children_left == -1).sum()) for e in rf.estimators_)
    res = {"trees": n_est, "min_leaf": leaf, "leaves": leaves, "ab": [float(ab[0]), float(ab[1])]}
    for name, P in [("forest", Pf_te), ("hybrid", Ph)]:
        cov, size = region_stats(P, yte)
        res[name] = {"median_err_m": float(np.median(D[yte, pick(P)])), "coverage80_%": 100 * cov, "region_size": size, "nll": nll(P, yte)}
    print(res)
    if leaves <= 26000:
        chosen = (rf, res)
        break
    chosen = (rf, res)

rf, res = chosen
trees = []
for e in rf.estimators_:
    t = e.tree_
    cls, prob, off = [], [], []
    for node in range(t.node_count):
        off.append(len(cls))
        if t.children_left[node] == -1:
            v = t.value[node][0]; v = v / v.sum()
            for c in np.nonzero(v > 1e-9)[0]:
                cls.append(int(rf.classes_[c])); prob.append(round(float(v[c]), 4))
    off.append(len(cls))
    trees.append({"f": t.feature.tolist(), "t": t.threshold.tolist(), "l": t.children_left.tolist(),
                  "r": t.children_right.tolist(), "off": off, "cls": cls, "p": prob})
Xc = feats(Rte[:5]); check = {"X": np.round(Xc, 8).tolist(), "P": np.round(full_proba(rf, Xc, N), 6).tolist()}
full = {r["model"]: r for r in json.load(open(hybrid_path))["rows"]}["Hybrid: forest × physics (fitted pooling)"]
json.dump({"sensors": [nodes[s] for s in sensors], "trained_noise": demo["demand_sigma"], **res, "full_hybrid": {k: full[k] for k in ["median_err_m", "coverage80_%", "region_size"]}, "tree_data": trees, "check": check},
          open(out_path, "w"), separators=(",", ":"))
print("wrote", out_path, round(os.path.getsize(out_path) / 1e6, 2), "MB")
