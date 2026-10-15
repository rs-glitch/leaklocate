"""Benchmark ladder for leak localization on one held-out test set (full EPANET runs, unseen seeds).

Step 0  best-guess baselines: random junction, constant 'network centre', largest-drop sensor
Step 1  standard ML: random forest (24 h means), gradient boosting (hourly profiles)
Step 1b classical physics: Bayesian signature matcher (this project's locator)
All use the same 6 greedy sensors and the same metrics.

Usage: python scripts/ladder.py data/ladder_train data/ladder_test reports/demo_data.json reports/ladder.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from leaklocate.export_demo import posterior

tr_dir, te_dir, demo_path, out_path = sys.argv[1:5]
demo = json.load(open(demo_path))
nodes = demo["nodes"]
N = len(nodes)
D = np.array(demo["dist_m"], float)
sig, noise = np.array(demo["signature"]), np.array(demo["noise"])
sensors = sorted(nodes.index(n) for n in demo["greedy_order"][:6])


def load(d):
    R = np.load(f"{d}/residuals.npy")
    df = pd.read_parquet(f"{d}/scenarios.parquet")
    y = np.array([nodes.index(n) for n in df["leak_node"]])
    return R[:, :, sensors], y, df["leak_lps"].to_numpy()


Rtr, ytr, _ = load(tr_dir)
Rte, yte, qte = load(te_dir)


def feats_mean(R):
    m = R.mean(axis=1)
    return np.hstack([m - m.mean(axis=1, keepdims=True), m.mean(axis=1, keepdims=True)])


def feats_hourly(R):
    c = R - R.mean(axis=2, keepdims=True)  # common mode removed per hour
    return np.hstack([c.reshape(len(R), -1), R.mean(axis=2)])


def metrics(P=None, pred=None):
    """P: [S, N] probabilities (optional); pred: [S] point predictions."""
    if pred is None:
        pred = P.argmax(axis=1)
    err = D[yte, pred]
    out = {"top1_%": 100 * float((pred == yte).mean()), "median_err_m": float(np.median(err)),
           "mean_err_m": float(err.mean())}
    if P is not None:
        out["expected_err_m"] = float((P * D[yte]).sum(axis=1).mean())
        order = np.argsort(-P, axis=1)
        cum = np.take_along_axis(P, order, axis=1).cumsum(axis=1)
        k = (cum < 0.8).sum(axis=1) + 1
        inreg = np.array([yte[i] in order[i, :k[i]] for i in range(len(yte))])
        out["coverage80_%"] = 100 * float(inreg.mean())
        out["region_size"] = float(k.mean())
    bins = [(0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16.01)]
    out["median_err_by_size_m"] = [float(np.median(err[(qte >= a) & (qte < b)])) for a, b in bins]
    return out


rows = []
rng = np.random.default_rng(0)
t = time.time()
m_rand = metrics(pred=rng.integers(0, N, len(yte)))
m_rand["expected_err_m"] = float(D[yte].mean(axis=1).mean())
rows.append({"step": "0", "model": "Random junction", **m_rand})
centre = int(np.argmin(D.mean(axis=1)))
rows.append({"step": "0", "model": f"Constant: network centre (junction {nodes[centre]})", **metrics(pred=np.full(len(yte), centre))})
c = Rte.mean(axis=1)
rows.append({"step": "0", "model": "Largest-drop sensor", **metrics(pred=np.array(sensors)[(c - c.mean(1, keepdims=True)).argmin(1)])})


def full_proba(clf, X):
    P = np.zeros((len(X), N))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(feats_mean(Rtr), ytr)
rows.append({"step": "1", "model": "Random forest (24 h means)", **metrics(P=full_proba(rf, feats_mean(Rte)))})
t1 = time.time()
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=True, random_state=0).fit(feats_hourly(Rtr), ytr)
rows.append({"step": "1", "model": "Gradient boosting (hourly profiles)", **metrics(P=full_proba(gb, feats_hourly(Rte)))})
print(f"gbm fit {time.time() - t1:.0f} s")
P = np.array([posterior(c[i], sig, noise[:300], sensors) for i in range(len(yte))])
rows.append({"step": "1b", "model": "Physics: Bayesian signature matcher", **metrics(P=P)})


def min_expected_distance(P):
    """Bayes decision rule for pipe-distance loss: the junction minimizing expected distance."""
    return (P @ D).argmin(axis=1)


Prf = full_proba(rf, feats_mean(Rte))
m = metrics(P=Prf); m.update({k: v for k, v in metrics(pred=min_expected_distance(Prf)).items()})
rows.append({"step": "2", "model": "Random forest + min-expected-distance pick", **m})
m = metrics(P=P); m.update({k: v for k, v in metrics(pred=min_expected_distance(P)).items()})
rows.append({"step": "2", "model": "Physics + min-expected-distance pick", **m})

res = {"test_n": int(len(yte)), "train_n": int(len(ytr)), "sensors": [nodes[s] for s in sensors],
       "leak_range_lps": [0.5, 16], "bins": ["0.5–1", "1–2", "2–4", "4–8", "8–16"], "rows": rows}
json.dump(res, open(out_path, "w"), indent=1)
show = pd.DataFrame(rows).drop(columns=["median_err_by_size_m"])
print(show.round(1).to_string(index=False))
print(pd.DataFrame({r["model"][:28]: r["median_err_by_size_m"] for r in rows}, index=res["bins"]).round(0).to_string())
print(f"total {time.time() - t:.0f} s")
