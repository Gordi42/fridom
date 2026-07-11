"""Walled (channel) shallow water: linear assembly + skew-adjointness.

The nonlinear (Sadourny) walled coverage lives in test_sadourny.py.
"""
import numpy as np

from fridom.spatial.bc import BC
from fridom.spatial.spaces.nodal import NodeSet
from fridom.model.energy import shallowwater_energy_weights

from .conftest import N, gaussian_bump, make_grid, make_model

CSQR = 0.7


def make_walled_model(**kwargs):
    """Return a linear channel model: x periodic, y walled."""
    kwargs.setdefault("csqr", CSQR)
    return make_model(make_grid(periodic_y=False), advection=False,
                      **kwargs)


def random_states(model, count, seed=7):
    """Capture `count` random prognostic states off one model."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(count):
        model.set_fields(
            u=rng.standard_normal(model.state["u"].shape),
            v=rng.standard_normal(model.state["v"].shape),
            p=rng.standard_normal(model.state["p"].shape))
        states.append(model.state)
    return states


def m_inner(a, b, csqr=CSQR):
    """Compute the measure-weighted energy inner product ``<a, b>_M``.

    Component weights ``diag(1, 1, 1/c^2)`` times each component's
    own per-axis metric measure (dual cell widths on faces — the
    walled y axis weights v's interior faces correctly).
    """
    weights = shallowwater_energy_weights(1.0 / csqr)
    total = 0.0
    for name, weight in weights.items():
        prod = np.asarray(a[name].data) * np.asarray(b[name].data)
        for axis in ("x", "y"):
            prod = prod * np.asarray(a[name].measure(axis).data)
        total += weight * float(prod.sum())
    return total


# ================================================================
#  Assembly: topology-derived wall spaces
# ================================================================
def test_walled_model_assembles_with_the_derived_wall_spaces():
    model = make_walled_model()
    my = model.grid.factors[1]
    # v: Dirichlet on its own bounded component axis (impermeable
    # walls), interior faces only — N-1 DOFs, interned identity
    v_y = model.state["v"].function_space.bare.factor("y")
    assert v_y is my.nodal(NodeSet.INNER, bc=BC.DIRICHLET)
    assert model.state["v"].shape == (N, N - 1)
    # u, p stay BC-free centre along y
    for name in ("u", "p"):
        y = model.state[name].function_space.bare.factor("y")
        assert y is my.nodal(NodeSet.CENTER, bc=BC.NONE)


def test_walled_tendency_lands_on_the_state_spaces():
    model = make_walled_model()
    (z,) = random_states(model, 1)
    tendency = model.tendency(z)
    for name in ("u", "v", "p"):
        assert (tendency[name].function_space.bare
                is z[name].function_space.bare)


# ================================================================
#  The M-skew-adjointness of the linearized operator
# ================================================================
def test_walled_linear_operator_is_m_skew_adjoint():
    # <z1, L z2>_M = -<L z1, z2>_M under the measure-weighted energy
    # metric: gravity + Coriolis + walls, discretely antisymmetric
    model = make_walled_model()
    z1, z2 = random_states(model, 2)
    lz1 = model.tendency(z1)
    lz2 = model.tendency(z2)
    residual = m_inner(z1, lz2) + m_inner(lz1, z2)
    scale = (np.sqrt(m_inner(z1, z1) * m_inner(lz2, lz2))
             + np.sqrt(m_inner(lz1, lz1) * m_inner(z2, z2)))
    assert abs(residual) / scale < 1e-14
    # the quadratic form vanishes for real states: <z, Lz>_M = 0
    assert abs(m_inner(z1, lz1)) / m_inner(z1, z1) < 1e-14


# ================================================================
#  A short linear run: stable, energy bounded
# ================================================================
def test_walled_linear_run_is_stable_with_bounded_energy():
    model = make_walled_model(dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.05))
    e0 = 0.5 * m_inner(model.state, model.state)
    energies = []
    for _ in range(20):
        model.advance(5)
        energies.append(0.5 * m_inner(model.state, model.state))
    energies = np.asarray(energies)
    assert np.isfinite(energies).all()
    # the exact invariant of the skew-adjoint operator: AB3 leaves
    # only a bounded time-discretization drift
    assert np.all(np.abs(energies - e0) / e0 < 5e-3)
