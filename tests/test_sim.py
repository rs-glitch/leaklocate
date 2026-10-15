import numpy as np
import pytest

from leaklocate.sim import Scenario, load_network, nominal_pressure, residual, simulate


@pytest.fixture(scope="module")
def net3():
    return load_network("Net3", hours=24)


@pytest.fixture(scope="module")
def model(net3):
    return nominal_pressure(net3)


def test_no_leak_no_noise_gives_zero_residual(net3, model):
    out = simulate(net3, Scenario(None, 0.0, 0.0, 0.0, seed=7))
    assert np.abs(residual(out["pressure"], model)).max() < 1e-3


def test_same_seed_is_deterministic(net3):
    sc = Scenario("15", 3.0, 0.05, 0.1, seed=11)
    a, b = simulate(net3, sc)["pressure"], simulate(net3, sc)["pressure"]
    np.testing.assert_array_equal(a, b)


def _leak_node_with_good_pressure(net3, model):
    nodes = net3.junction_name_list
    return nodes[int(np.argmax(np.median(model, axis=0) > 40))]  # first node with median pressure > 40 m


def test_leak_adds_its_flow_to_network_demand(net3, model):
    """Mass balance: extra total outflow equals the emitter flow C*sqrt(p) at the leak node."""
    node = _leak_node_with_good_pressure(net3, model)
    q = 5.0
    leak = simulate(net3, Scenario(node, q, 0.0, 0.0, seed=1))
    free = simulate(net3, Scenario(None, 0.0, 0.0, 0.0, seed=1))
    extra_lps = (leak["demand"].sum(axis=1) - free["demand"].sum(axis=1)) * 1000.0
    i = leak["nodes"].index(node)
    expected = q * np.sqrt(np.clip(leak["pressure"][:, i], 0, None) / 30.0)
    # PDD can slightly reduce other consumers' supply, so allow a small tolerance
    np.testing.assert_allclose(extra_lps, expected, rtol=0.05, atol=0.05)


def test_leak_lowers_pressure_most_near_the_leak(net3, model):
    node = _leak_node_with_good_pressure(net3, model)
    out = simulate(net3, Scenario(node, 8.0, 0.0, 0.0, seed=1))
    r = residual(out["pressure"], model).mean(axis=0)
    i = out["nodes"].index(node)
    assert r[i] < 0, "pressure at the leak must drop"
    assert r[i] <= np.quantile(r, 0.25), "leak node should be among the largest drops"


def test_mismatch_seed_gives_one_shared_true_city(net3, model):
    """Two leak-free scenarios with different noise seeds but the same mismatch seed see the same model error."""
    a = simulate(net3, Scenario(None, 0.0, 0.0, 0.2, seed=1, demand_bias_sigma=0.2, mismatch_seed=5))
    b = simulate(net3, Scenario(None, 0.0, 0.0, 0.2, seed=2, demand_bias_sigma=0.2, mismatch_seed=5))
    c = simulate(net3, Scenario(None, 0.0, 0.0, 0.2, seed=1, demand_bias_sigma=0.2, mismatch_seed=6))
    ra, rb, rc = (residual(x["pressure"], model) for x in (a, b, c))
    np.testing.assert_allclose(ra, rb, atol=1e-4)
    assert np.abs(ra).max() > 0.05, "mismatch must change pressures"
    assert np.abs(ra - rc).max() > 0.05, "a different mismatch seed is a different city"
