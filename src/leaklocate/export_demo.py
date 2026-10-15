"""Export everything the interactive HTML demo needs, from real EPANET simulations.

The browser composes an observation as  signature(leak node, size) + noise sample  and localizes it.
This file computes those pieces and checks that superposition holds (reports the error).

Usage: python -m leaklocate.export_demo --network Net3 --out reports/demo_data.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np

from . import generate as G
from .sim import Scenario, load_network, nominal_pressure, simulate

SIZES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]  # L/s at 30 m


def _sig(args):
    node, q = args
    return simulate(G._BASE, Scenario(node, q, 0.0, 0.0, 0))["pressure"]


def _noise(seed_sigma):
    seed, sigma = seed_sigma
    return simulate(G._BASE, Scenario(None, 0.0, sigma, 0.0, seed))["pressure"]


def _combo(args):
    node, q, sigma, seed = args
    return simulate(G._BASE, Scenario(node, q, sigma, 0.0, seed))["pressure"]


def match_scores(y, sig, cov_inv, sensors):
    """Log-likelihood per (node, size) with the common mode removed. y [k], sig [N, Z, Ntot]."""
    m = sig[:, :, sensors]
    yc = y - y.mean()
    mc = m - m.mean(axis=2, keepdims=True)
    d = yc[None, None, :] - mc
    return -0.5 * np.einsum("nzi,ij,nzj->nz", d, cov_inv, d)


def posterior(y, sig, noise, sensors, reg=1e-3):
    X = noise[:, sensors]
    X = X - X.mean(axis=1, keepdims=True)
    cov = np.cov(X, rowvar=False) + reg * np.eye(len(sensors))
    ll = match_scores(y, sig, np.linalg.inv(cov), sensors)
    ll = ll - ll.max()
    p = np.exp(ll).sum(axis=1)  # marginalize over leak size
    return p / p.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="Net3")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--demand-sigma", type=float, default=0.05)
    ap.add_argument("--n-noise", type=int, default=400)
    ap.add_argument("--max-sensors", type=int, default=20)
    ap.add_argument("--out", default="reports/demo_data.json")
    a = ap.parse_args()

    base = load_network(a.network, a.hours)
    nodes = base.junction_name_list
    N = len(nodes)
    model = nominal_pressure(base).mean(axis=0)
    with ProcessPoolExecutor(initializer=G._init, initargs=(a.network, a.hours)) as ex:
        sig = np.array(list(ex.map(_sig, [(n, q) for n in nodes for q in SIZES], chunksize=8)))
        sig = sig.mean(axis=1).reshape(N, len(SIZES), N) - model  # [leak node, size, observed node]
        noise = np.array(list(ex.map(_noise, [(s, a.demand_sigma) for s in range(1, a.n_noise + 1)], chunksize=8)))
        noise = noise.mean(axis=1) - model  # [sample, node]
        rng = np.random.default_rng(0)
        checks = [(nodes[rng.integers(N)], float(rng.choice(SIZES[1:])), a.demand_sigma, 900_000 + i) for i in range(60)]
        combo = np.array(list(ex.map(_combo, checks))).mean(axis=1) - model
        lone = np.array(list(ex.map(_noise, [(c[3], c[2]) for c in checks])))
        lone = lone.mean(axis=1) - model

    # Superposition check: full (leak + noise) run vs signature + same-seed noise-only run.
    approx = np.array([sig[nodes.index(n), SIZES.index(q)] for n, q, _, _ in checks]) + lone
    leak_part = np.array([sig[nodes.index(n), SIZES.index(q)] for n, q, _, _ in checks])
    err = np.abs(combo - approx).max(axis=1)
    scale = np.abs(leak_part).max(axis=1)
    superposition = {"median_max_abs_error_m": float(np.median(err)), "p95_max_abs_error_m": float(np.percentile(err, 95)),
                     "median_error_over_leak_signal": float(np.median(err / np.maximum(scale, 1e-6)))}
    print("superposition:", superposition)

    # Pipe-length shortest paths between junctions (for error in metres).
    g = nx.Graph()
    for name, link in base.links():
        length = getattr(link, "length", 1.0) or 1.0
        g.add_edge(link.start_node_name, link.end_node_name, weight=float(length))
    dist = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
    D = np.array([[dist[i].get(j, np.inf) for j in nodes] for i in nodes])

    # Greedy sensor placement by Monte Carlo localization error (same matcher as the browser).
    rng = np.random.default_rng(1)
    trials = [(rng.integers(N), rng.integers(1, len(SIZES)), rng.integers(len(noise))) for _ in range(300)]

    def mean_error(sensors):
        errs = []
        for li, zi, ni in trials:
            y = sig[li, zi, sensors] + noise[ni, sensors]
            p = posterior(y, sig, noise, sensors)
            errs.append(float(p @ D[li]))  # expected pipe distance from true leak (m)
        return float(np.mean(errs))

    order, curve = [], []
    for _ in range(a.max_sensors):
        if len(order) < 2:  # need 2+ sensors before the common-mode removal is meaningful: seed with a pair
            cand = [(mean_error(order + [c]) if len(order) == 1 else 0.0, c) for c in range(N) if c not in order]
            if len(order) == 0:
                order.append(int(np.argmax(np.abs(sig).sum(axis=(0, 1)))))
                curve.append(None)
                continue
        else:
            cand = [(mean_error(order + [c]), c) for c in range(N) if c not in order]
        best = min(cand)
        order.append(int(best[1]))
        curve.append(round(best[0], 1))
        print(f"sensors={len(order)} expected error={best[0]:.0f} m")

    xy = {n: base.get_node(n).coordinates for n in base.node_name_list}
    data = {
        "network": a.network, "hours": a.hours, "demand_sigma": a.demand_sigma, "sizes_lps": SIZES,
        "nodes": nodes, "xy": [xy[n] for n in nodes],
        "other_nodes": [{"name": n, "type": base.get_node(n).node_type, "xy": xy[n]} for n in base.node_name_list if n not in nodes],
        "links": [{"a": l.start_node_name, "b": l.end_node_name, "type": l.link_type} for _, l in base.links()],
        "signature": np.round(sig, 4).tolist(), "noise": np.round(noise, 4).tolist(),
        "dist_m": np.where(np.isfinite(D), np.round(D, 0), -1).tolist(),
        "greedy_order": [nodes[i] for i in order], "greedy_error_m": curve,
        "superposition": superposition, "system_demand_lps": None,
    }
    tot = simulate(base, Scenario(None, 0.0, 0.0, 0.0, 0))["demand"].sum(axis=1).mean() * 1000
    data["system_demand_lps"] = round(float(tot), 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", a.out, round(os.path.getsize(a.out) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
