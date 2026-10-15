"""Checkpoint 1 figures: residual maps, detectability vs leak size, naive 'largest drop' baseline.

Usage: python scripts/checkpoint1.py data/net3_v2 reports/figures
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import wntr

from leaklocate.sim import load_network

data, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
R = np.load(os.path.join(data, "residuals.npy"))  # [S, T, N]
df = pd.read_parquet(os.path.join(data, "scenarios.parquet"))
meta = json.load(open(os.path.join(data, "meta.json")))
nodes = meta["nodes"]
wn = load_network(meta["network"], meta["hours"], not meta.get("rule_controls", False))
G = wn.to_graph().to_undirected()
G = nx.Graph(G)  # collapse parallel links

mean_r = R.mean(axis=1)  # [S, N] time-averaged residual per node
leak = df["leak_node"].notna().to_numpy()

# 1) Detectability with every junction instrumented (upper bound): signal = largest drop.
signal = -mean_r.min(axis=1)
thr = np.quantile(signal[~leak], 0.95)  # 5 % false-alarm threshold from no-leak scenarios
bins = [0.5, 1, 2, 4, 8.01]
size = df["leak_lps"].to_numpy()
rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = leak & (size >= lo) & (size < hi)
    rows.append({"leak_lps": f"{lo}-{hi:g}", "n": int(m.sum()), "detected_%": round(100 * float((signal[m] > thr).mean()), 1)})

# 2) Naive localization: the junction with the largest mean pressure drop.
pred = np.array(nodes)[mean_r.argmin(axis=1)]
lengths = dict(nx.all_pairs_shortest_path_length(G))
hops = np.array([lengths[t][p] if isinstance(t, str) else -1 for t, p in zip(df["leak_node"], pred)])
for r in rows:
    lo, hi = (float(x) for x in r["leak_lps"].split("-"))
    m = leak & (size >= lo) & (size < hi)
    r["exact_%"] = round(100 * float((hops[m] == 0).mean()), 1)
    r["within_2_hops_%"] = round(100 * float((hops[m] <= 2).mean()), 1)
    r["median_hops"] = float(np.median(hops[m]))
table = pd.DataFrame(rows)
table.to_csv(os.path.join(out, "checkpoint1_table.csv"), index=False)
print(f"noise threshold (95% of no-leak max drop): {thr:.3f} m")
print(table.to_string(index=False))

# 3) Residual maps for a small, medium and large leak at the same node.
example_node = df.loc[leak & (size > 4), "leak_node"].iloc[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, target in zip(axes, [1.0, 3.0, 7.0]):
    cand = df[(df["leak_node"].notna())]
    k = int((cand["leak_lps"] - target).abs().idxmin())
    attr = pd.Series(mean_r[k], index=nodes)
    wntr.graphics.plot_network(wn, node_attribute=attr, node_size=22, node_cmap="viridis", ax=ax,
                               title=f"leak {df.loc[k, 'leak_lps']:.1f} L/s at node {df.loc[k, 'leak_node']}",
                               node_colorbar_label="mean residual (m)")
    xy = wn.get_node(df.loc[k, "leak_node"]).coordinates
    ax.plot(*xy, marker="x", color="red", markersize=14, mew=3)
fig.suptitle("Net3: pressure residual (measured − model), demand noise 5 %, sensor noise 0.05 m. Red × = true leak")
fig.tight_layout()
fig.savefig(os.path.join(out, "checkpoint1_residual_maps.png"), dpi=130)

# 4) Signal vs leak size.
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.scatter(size[leak], signal[leak], s=6, alpha=0.4, label="leak scenarios")
ax.axhline(thr, color="k", ls="--", label=f"5 % false-alarm threshold ({thr:.2f} m)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("leak size at 30 m (L/s)"); ax.set_ylabel("largest mean pressure drop (m)")
ax.set_title("Upper bound: every junction measured"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "checkpoint1_signal_vs_size.png"), dpi=130)
print("figures ->", out)
