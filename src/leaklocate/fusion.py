"""Shared helpers for forest + physics fusion (used by scripts/hybrid.py logic and scripts/export_forest.py)."""
import numpy as np
import pandas as pd

EPS = 1e-6


def load(d, nodes, sensors):
    R = np.load(f"{d}/residuals.npy")[:, :, sensors]
    df = pd.read_parquet(f"{d}/scenarios.parquet")
    return R, np.array([nodes.index(n) for n in df["leak_node"]]), df["leak_lps"].to_numpy()


def feats(R):
    """24 h mean residual at the sensors, common mode removed, plus the common mode itself."""
    m = R.mean(axis=1)
    return np.hstack([m - m.mean(axis=1, keepdims=True), m.mean(axis=1, keepdims=True)])


def full_proba(clf, X, n_classes):
    P = np.zeros((len(X), n_classes))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def pool(Pf, Pp, a, b):
    L = a * np.log(Pf + EPS) + b * np.log(Pp + EPS)
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=1, keepdims=True)


def nll(P, y):
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def region_stats(P, y, level=0.8):
    order = np.argsort(-P, axis=1)
    cum = np.take_along_axis(P, order, axis=1).cumsum(axis=1)
    k = (cum < level).sum(axis=1) + 1
    rank = np.argsort(order, axis=1)[np.arange(len(y)), y]
    return float((rank < k).mean()), float(k.mean())
