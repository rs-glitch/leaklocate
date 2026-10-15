"""Generate a batch of leak scenarios in parallel and store residuals.

Usage:  python -m leaklocate.generate --network Net3 --n 2000 --out data/net3_v1
Output: <out>/residuals.npy [S, T, N] float32, <out>/scenarios.parquet, <out>/meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .sim import Scenario, load_network, nominal_pressure, residual, simulate

_BASE = None
_MODEL = None


def _init(network: str, hours: int, freeze: bool = True) -> None:
    global _BASE, _MODEL
    # EPANET names its hydraulics scratch file inside TEMP; parallel workers sharing TEMP collide
    # (Error 307). Give each worker a private TEMP before any simulation runs.
    private = tempfile.mkdtemp(prefix=f"leaklocate_w{os.getpid()}_")
    os.environ["TMP"] = os.environ["TEMP"] = private
    tempfile.tempdir = private
    _BASE = load_network(network, hours, freeze)
    _MODEL = nominal_pressure(_BASE)


def _work(args: tuple[Scenario, float]) -> tuple[np.ndarray, float]:
    sc, sensor_sigma = args
    out = simulate(_BASE, sc)
    leak_flow = 0.0
    if sc.leak_node is not None:  # emitter law Q = C*sqrt(p), with C set from leak_lps at 30 m (see sim.simulate)
        i = out["nodes"].index(sc.leak_node)
        p = np.clip(out["pressure"][:, i], 0, None)
        leak_flow = float(np.mean(sc.leak_lps * np.sqrt(p / 30.0)))
    return residual(out["pressure"], _MODEL, sensor_sigma, sc.seed), leak_flow


def make_scenarios(nodes: list[str], n: int, seed: int, leak_lps=(0.5, 8.0), demand_sigma=0.05,
                   roughness_sigma=0.0, no_leak_frac=0.1, demand_bias_sigma=0.0, mismatch_seed=-1) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    out = []
    mm = {"demand_bias_sigma": demand_bias_sigma, "mismatch_seed": mismatch_seed}
    for k in range(n):
        if rng.random() < no_leak_frac:
            out.append(Scenario(None, 0.0, demand_sigma, roughness_sigma, int(seed * 1_000_000 + k), **mm))
        else:
            q = float(np.exp(rng.uniform(np.log(leak_lps[0]), np.log(leak_lps[1]))))  # log-uniform sizes
            out.append(Scenario(str(rng.choice(nodes)), q, demand_sigma, roughness_sigma, int(seed * 1_000_000 + k), **mm))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="Net3")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--demand-sigma", type=float, default=0.05)
    ap.add_argument("--roughness-sigma", type=float, default=0.0)
    ap.add_argument("--demand-bias-sigma", type=float, default=0.0, help="persistent per-junction base-demand error (lognormal sigma)")
    ap.add_argument("--mismatch-seed", type=int, default=-1, help=">= 0: every scenario shares one mismatched 'true city'")
    ap.add_argument("--leak-min", type=float, default=0.5)
    ap.add_argument("--leak-max", type=float, default=8.0)
    ap.add_argument("--no-leak-frac", type=float, default=0.1)
    ap.add_argument("--sensor-sigma", type=float, default=0.05, help="pressure sensor noise, metres")
    ap.add_argument("--rule-controls", action="store_true", help="keep rule-based pump controls (reproduces the pre-fix diagnosis batch)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = load_network(a.network, a.hours, not a.rule_controls)
    scs = make_scenarios(base.junction_name_list, a.n, a.seed, demand_sigma=a.demand_sigma,
                         roughness_sigma=a.roughness_sigma,
                         leak_lps=(a.leak_min, a.leak_max), no_leak_frac=a.no_leak_frac,
                         demand_bias_sigma=a.demand_bias_sigma, mismatch_seed=a.mismatch_seed)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.network, a.hours, not a.rule_controls)) as ex:
        results = list(ex.map(_work, [(s, a.sensor_sigma) for s in scs], chunksize=16))
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "residuals.npy"), np.stack([r for r, _ in results]).astype(np.float32))
    df = pd.DataFrame([s.to_dict() for s in scs])
    df["leak_flow_lps"] = [f for _, f in results]
    df.to_parquet(os.path.join(a.out, "scenarios.parquet"))
    meta = {**vars(a), "nodes": base.junction_name_list, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{len(scs)} scenarios in {meta['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
