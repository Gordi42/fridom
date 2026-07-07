"""Sadourny advection: the csqr-field behaviour and conservation."""
import numpy as np

import fridom.framework2 as fr
from fridom.shallowwater2 import params as sw_params

from .conftest import (
    gaussian_bump,
    make_grid,
    make_model,
    total_energy,
)


# ================================================================
#  The csqr-FIELD delta (§8.8): Sadourny reads state["csqr"]
# ================================================================
def test_csqr_is_a_state_field_not_a_scalar():
    model = make_model(csqr=1.5)
    c = model.state["csqr"]
    # a one-DOF Profile() field (constant depth) that broadcasts to the
    # nodal join in the tendency terms — still a field, not a scalar
    assert isinstance(c, fr.grid.ScalarField)
    assert c.function_space.bare is (
        fr.Profile().resolve(model.grid))
    np.testing.assert_allclose(np.asarray(c.data), 1.5)


def test_csqr_scalar_is_published_for_host_reads():
    model = make_model(csqr=2.25)
    assert float(model.parameters[sw_params.CSQR]) == 2.25


def test_advection_changes_the_solution_vs_linear():
    grid = make_grid()
    nonlin = make_model(grid, rossby_number=0.5, advection=True)
    linear = make_model(grid, rossby_number=0.5, advection=False)
    nonlin.set_fields(p=gaussian_bump(amp=0.2),
                      u=lambda x, y: 0.2 * np.sin(2 * np.pi * y) + 0.0 * x)
    linear.set_fields(p=gaussian_bump(amp=0.2),
                      u=lambda x, y: 0.2 * np.sin(2 * np.pi * y) + 0.0 * x)
    nonlin.advance(40)
    linear.advance(40)
    # the nonlinear advection leaves a measurable imprint
    diff = np.abs(np.asarray(nonlin.state["p"].data)
                  - np.asarray(linear.state["p"].data)).max()
    assert diff > 1e-4


# ================================================================
#  Conservation: mass exact, energy/enstrophy bounded
# ================================================================
def test_mass_exact_under_full_dynamics():
    model = make_model(rossby_number=0.5)
    model.set_fields(p=gaussian_bump(amp=0.15),
                     v=lambda x, y: 0.1 * np.cos(2 * np.pi * x) + 0.0 * y)
    mass0 = float(model.state["p"].integrate().data.ravel()[0])
    model.advance(50)
    mass1 = float(model.state["p"].integrate().data.ravel()[0])
    assert abs(mass1 - mass0) < 1e-12


def test_energy_bounded_with_a_balanced_start():
    model = make_model(rossby_number=0.3, dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.08),
                     u=lambda x, y: 0.05 * np.sin(2 * np.pi * y) + 0.0 * x,
                     v=lambda x, y: 0.05 * np.sin(2 * np.pi * x) + 0.0 * y)
    e0 = total_energy(model)
    peak = 0.0
    for _ in range(20):
        model.advance(5)
        peak = max(peak, abs(total_energy(model) - e0) / e0)
    assert peak < 5e-3


def test_potential_enstrophy_stays_bounded():
    model = make_model(rossby_number=0.4, dt=2e-3)
    model.set_fields(p=gaussian_bump(amp=0.1),
                     u=lambda x, y: 0.1 * np.sin(2 * np.pi * y) + 0.0 * x)

    def enstrophy(m):
        # 0.5 * integral zeta^2 (NE corner) — a bounded invariant
        z = m.state.rel_vort
        return float((0.5 * z * z).integrate().data.ravel()[0])

    q0 = enstrophy(model)
    model.advance(80)
    q1 = enstrophy(model)
    assert np.isfinite(q1)
    assert q1 < 5.0 * (q0 + 1e-9)   # no runaway enstrophy growth
