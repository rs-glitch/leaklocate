"""Leak scenario simulation on EPANET networks (via WNTR).

A scenario = one leak (an emitter at a junction) + random demand noise + optional model mismatch
(pipe roughness and base demands that differ from the utility's model). The observable is the pressure residual: what a utility sees when it compares field
readings with its (nominal, leak-free) hydraulic model. Residual = measured - model.
"""
from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import asdict, dataclass

import numpy as np
import wntr

NETWORK_DIR = os.path.join(os.path.dirname(wntr.__file__), "library", "networks")


def load_network(name: str, hours: int = 24, freeze: bool = True) -> wntr.network.WaterNetworkModel:
    """Load a bundled network (e.g. 'Net3') or a path to an .inp file; hourly steps for `hours`.

    freeze=False keeps the network's own rule-based controls (only used to reproduce the pump-switching diagnosis).
    """
    path = name if name.endswith(".inp") else os.path.join(NETWORK_DIR, f"{name}.inp")
    wn = wntr.network.WaterNetworkModel(path)
    wn.options.time.duration = hours * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"  # pressure-driven: leaks cannot draw water at negative pressure
    wn.options.hydraulic.required_pressure = 10.0  # m, full service above this (EPANET rejects the 0.07 default)
    wn.options.hydraulic.minimum_pressure = 0.0
    wn.options.hydraulic.emitter_exponent = 0.5  # orifice law, Q = C * p^0.5
    if freeze:
        _freeze_operations(wn)
    return wn


def _freeze_operations(wn: wntr.network.WaterNetworkModel) -> None:
    """Replace level/time controls with the nominal hourly status schedule of every controlled link.

    Why: with rule-based controls, demand noise shifts tank levels, so pumps switch at different hours in
    the 'true' run than in the model, which gives 10-30 m residual spikes unrelated to leaks (diagnosed on Net3,
    58/205 no-leak scenarios). Utilities know pump and valve states from SCADA, so feeding the model the same
    schedule is the realistic setup.
    """
    from wntr.network.controls import Control, ControlAction, SimTimeCondition

    targets = sorted({a.target()[0].name for _, c in wn.controls() for a in c.actions()})
    if not targets:
        return
    status = _run(copy.deepcopy(wn)).link["status"][targets]
    for name in list(wn.control_name_list):
        wn.remove_control(name)
    for link_name in targets:
        link = wn.get_link(link_name)
        prev = None
        for t, s in status[link_name].items():
            s = int(round(s))
            if s == prev:
                continue
            st = wntr.network.LinkStatus.Open if s == 1 else wntr.network.LinkStatus.Closed
            wn.add_control(f"sched_{link_name}_{int(t)}",
                           Control(SimTimeCondition(wn, "=", int(t)), ControlAction(link, "status", st)))
            prev = s


@dataclass(frozen=True)
class Scenario:
    leak_node: str | None  # None = no leak
    leak_lps: float  # leak flow at the reference pressure, litres per second
    demand_sigma: float  # per-node, per-hour multiplicative demand noise (lognormal sigma)
    roughness_sigma: float  # model mismatch: true roughness = nominal * lognormal(sigma), per pipe
    seed: int
    demand_bias_sigma: float = 0.0  # model mismatch: true base demand = nominal * lognormal(sigma), per junction, all day
    mismatch_seed: int = -1  # >= 0: draw the mismatch from this seed, so many scenarios share one 'true city'

    def to_dict(self) -> dict:
        return asdict(self)


def _run(wn: wntr.network.WaterNetworkModel) -> wntr.sim.results.SimulationResults:
    # EPANET writes temp files; a private directory keeps parallel workers apart. Ignore cleanup errors
    # (Windows may still hold the .rpt handle) so they never mask a real simulation error.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(d, "s"))


def simulate(base: wntr.network.WaterNetworkModel, sc: Scenario, ref_pressure_m: float = 30.0) -> dict:
    """Simulate the 'true' network for one scenario. Returns pressures [T, N], demands [T, N], node names."""
    rng = np.random.default_rng(sc.seed)
    wn = copy.deepcopy(base)
    junctions = wn.junction_name_list

    if sc.demand_sigma > 0:  # each junction gets its own hourly multiplier series
        n_steps = int(wn.options.time.duration // wn.options.time.pattern_timestep) + 1
        for name in junctions:
            j = wn.get_node(name)
            for k, ts in enumerate(j.demand_timeseries_list):
                if ts.base_value == 0:
                    continue
                base_pat = ts.pattern.multipliers if ts.pattern is not None else np.ones(1)
                mult = np.resize(np.asarray(base_pat, float), n_steps) * rng.lognormal(0, sc.demand_sigma, n_steps)
                pname = f"noise_{name}_{k}"
                wn.add_pattern(pname, mult.tolist())
                ts.pattern_name = pname

    # Mismatch draws come after the demand noise, so a scenario with and without mismatch shares the same noise.
    mrng = np.random.default_rng(sc.mismatch_seed) if sc.mismatch_seed >= 0 else rng
    if sc.roughness_sigma > 0:
        for _, pipe in wn.pipes():
            pipe.roughness = float(pipe.roughness * mrng.lognormal(0, sc.roughness_sigma))

    if sc.demand_bias_sigma > 0:  # persistent error in the model's base demands (e.g. billing data allocated to the wrong nodes)
        for name in junctions:
            for ts in wn.get_node(name).demand_timeseries_list:
                if ts.base_value != 0:
                    ts.base_value = float(ts.base_value * mrng.lognormal(0, sc.demand_bias_sigma))

    if sc.leak_node is not None and sc.leak_lps > 0:
        # Emitter coefficient so that the leak is `leak_lps` at the reference pressure (SI: m3/s per m^0.5).
        wn.get_node(sc.leak_node).emitter_coefficient = (sc.leak_lps / 1000.0) / np.sqrt(ref_pressure_m)

    res = _run(wn)
    p = res.node["pressure"][junctions]
    d = res.node["demand"][junctions]
    return {"pressure": p.to_numpy(np.float32), "demand": d.to_numpy(np.float32), "nodes": junctions}


def nominal_pressure(base: wntr.network.WaterNetworkModel) -> np.ndarray:
    """The utility's model prediction: nominal demands, nominal roughness, no leak."""
    return simulate(base, Scenario(None, 0.0, 0.0, 0.0, 0))["pressure"]


def residual(measured: np.ndarray, model: np.ndarray, sensor_sigma_m: float = 0.0, seed: int = 0) -> np.ndarray:
    """Pressure residual [T, N] with additive sensor noise (metres)."""
    r = measured - model
    if sensor_sigma_m > 0:
        r = r + np.random.default_rng(seed + 1_000_003).normal(0, sensor_sigma_m, r.shape).astype(np.float32)
    return r
