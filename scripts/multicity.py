"""Multi-city sensor-planning study: for each benchmark network, simulate a leak at EVERY junction,
then choose sensor locations greedily and measure error vs number of sensors. Compare topologies.

Per network: signatures (every junction x 4 leak sizes), 300 demand-noise runs, greedy placement
(objective = expected pipe distance from the true leak under the physics posterior, Monte Carlo),
random-placement baseline, topology metrics, and a recommended sensor count.

Placement uses a diagonal noise model after common-mode removal (fast: one matrix product per
candidate set); the same model is used for every network so the comparison is consistent.

Usage: python scripts/multicity.py reports/multicity.json [names...]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np

from leaklocate import generate as G
from leaklocate.sim import Scenario, load_network, nominal_pressure, simulate

FRACS = [0.0025, 0.005, 0.01, 0.02]  # leak size as a share of each network's mean demand (fair across city sizes)
K_MAX = 16


def inp_path(name):
    import wntr
    from epyt_flow.data import networks as nets
    lib = os.path.join(os.path.dirname(wntr.__file__), "library", "networks")
    if os.path.exists(os.path.join(lib, f"{name}.inp")):
        return os.path.join(lib, f"{name}.inp")
    cfg = getattr(nets, f"load_{name.lower()}")(download_dir="data/networks", verbose=False)
    return cfg.f_inp_in


def _sig(args):
    node, q = args  # q in L/s
    return simulate(G._BASE, Scenario(node, q, 0.0, 0.0, 0))["pressure"].mean(axis=0)


def _noise(seed):
    return simulate(G._BASE, Scenario(None, 0.0, 0.05, 0.0, seed))["pressure"].mean(axis=0)


def topology(wn, nodes):
    g = nx.Graph()
    for _, l in wn.links():
        g.add_edge(l.start_node_name, l.end_node_name, weight=float(getattr(l, "length", 1.0) or 1.0))
    loops = g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)
    deg = dict(g.degree())
    pipe_km = sum(float(p.length) for _, p in wn.pipes()) / 1000
    return g, {"junctions": len(nodes), "pipes": wn.num_pipes, "pipe_km": round(pipe_km, 1), "loops": int(loops),
               "loops_per_100_junctions": round(100 * loops / len(nodes), 1),
               "dead_end_share": round(sum(1 for n in nodes if deg.get(n, 0) == 1) / len(nodes), 3),
               "mean_degree": round(float(np.mean([deg.get(n, 0) for n in nodes])), 2),
               "tanks": wn.num_tanks, "reservoirs": wn.num_reservoirs, "pumps": wn.num_pumps, "valves": wn.num_valves}


def study(name, workers):
    t0 = time.time()
    path = inp_path(name)
    base = load_network(path, 24)
    nodes = base.junction_name_list
    N = len(nodes)
    model = nominal_pressure(base).mean(axis=0)
    demand = float(simulate(base, Scenario(None, 0, 0, 0, 0))["demand"].sum(axis=1).mean() * 1000)
    SIZES = [f * demand for f in FRACS]
    with ProcessPoolExecutor(workers, initializer=G._init, initargs=(path, 24)) as ex:
        sig = np.array(list(ex.map(_sig, [(n, q) for n in nodes for q in SIZES], chunksize=8))).reshape(N, len(SIZES), N) - model
        noise = np.array(list(ex.map(_noise, range(1, 301), chunksize=8))) - model
    g, topo = topology(base, nodes)
    dist = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight")) if N <= 1500 else None
    D = np.array([[dist[i].get(j, np.nan) for j in nodes] for i in nodes])
    D = np.where(np.isfinite(D), D, np.nanmax(D))
    rand_err = float(D.mean())

    # Monte Carlo trials shared by every candidate set: leak node, size index (log-uniform over SIZES), noise sample
    rng = np.random.default_rng(0)
    T = 240 if N < 300 else 160
    tl, tz, tn = rng.integers(0, N, T), rng.integers(0, len(SIZES), T), rng.integers(150, 300, T)
    sensor_sd = 0.05 / np.sqrt(24)
    sig_flat = sig.reshape(N * len(SIZES), N)

    def expected_error(S):
        S = np.asarray(S)
        Y = sig[tl, tz][:, S] + noise[tn][:, S] + rng.normal(0, sensor_sd, (T, len(S)))
        Yc = Y - Y.mean(axis=1, keepdims=True)
        M = sig_flat[:, S]; Mc = M - M.mean(axis=1, keepdims=True)
        nz = noise[:150][:, S]; nzc = nz - nz.mean(axis=1, keepdims=True)
        w = 1.0 / (nzc.var(axis=0) + sensor_sd ** 2 + 1e-8)  # diagonal precision
        d2 = (Yc ** 2 * w).sum(1)[:, None] - 2 * (Yc * w) @ Mc.T + (Mc ** 2 * w).sum(1)[None, :]
        ll = -0.5 * d2; ll -= ll.max(axis=1, keepdims=True)
        P = np.exp(ll).reshape(T, N, len(SIZES)).sum(axis=2); P /= P.sum(axis=1, keepdims=True)
        return float((P * D[tl]).sum(axis=1).mean())

    cand_pool = np.arange(N)
    order, curve = [], []
    first = int(np.argmax(np.abs(sig[:, 2, :] - sig[:, 2, :].mean(1, keepdims=True)).sum(0)))
    order.append(first)
    for k in range(2, min(K_MAX, N) + 1):
        cands = cand_pool[~np.isin(cand_pool, order)]
        if len(cands) > 120:
            cands = rng.choice(cands, 120, replace=False)
        scores = [(expected_error(order + [int(c)]), int(c)) for c in cands]
        e, c = min(scores)
        order.append(c); curve.append({"k": k, "greedy_m": round(e, 1)})
    rand_curve = {}
    for k in [2, 4, 6, 8, 12, 16]:
        if k <= N:
            rand_curve[k] = round(float(np.mean([expected_error(list(rng.choice(N, k, replace=False))) for _ in range(5)])), 1)
    errs = np.array([c["greedy_m"] for c in curve])
    # recommended count: smallest k whose next sensor improves expected error by < 2 % of the random-guess error
    gains = -np.diff(errs)
    rec = next((curve[i]["k"] for i, gn in enumerate(gains) if gn < 0.02 * rand_err), curve[-1]["k"])
    xy = {n: base.get_node(n).coordinates for n in base.node_name_list}
    out = {"name": name, "topology": topo, "demand_lps": round(demand, 1), "leak_sizes_lps": [round(q, 2) for q in SIZES], "leak_fracs": FRACS, "random_guess_m": round(rand_err, 1),
           "greedy_curve": curve, "random_curve": rand_curve, "greedy_order": [nodes[i] for i in order], "recommended_k": int(rec),
           "err_at_rec_m": float(errs[[c["k"] for c in curve].index(rec)]),
           "nodes": nodes, "xy": [xy[n] for n in nodes],
           "other": [{"name": n, "type": base.get_node(n).node_type, "xy": xy[n]} for n in base.node_name_list if n not in set(nodes)],
           "links": [[l.start_node_name, l.end_node_name, l.link_type] for _, l in base.links()],
           "seconds": round(time.time() - t0, 1), "trials": int(T)}
    print(f"{name}: N={N} loops={topo['loops']} rec_k={rec} err@rec={out['err_at_rec_m']:.0f} m rand={rand_err:.0f} m ({out['seconds']} s)", flush=True)
    return out


if __name__ == "__main__":
    out_path = sys.argv[1]
    names = sys.argv[2:] or ["Anytown", "Hanoi", "Net2", "Net3", "Balerma", "Ltown"]
    workers = max(1, (os.cpu_count() or 2) // 2)  # leave CPU for the CrackGauge job
    results = []
    for n in names:
        try:
            results.append(study(n, workers))
        except Exception as e:  # keep going; report which network failed and why
            print(f"{n}: FAILED {type(e).__name__}: {e}", flush=True)
        json.dump(results, open(out_path, "w"), separators=(",", ":"))
    print("done", [r["name"] for r in results])
