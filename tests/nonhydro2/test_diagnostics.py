"""FV-family acceptance for the nonhydro2 diagnostics (F1 gate).

The diagnostics module (``nh.diagnostics``) is family-agnostic: it
converts staggered quantities onto the pressure cell with ``.to`` and
``.diff``. These tests run it **unchanged** on a finite-volume state
(FV-D2 option A: velocities face-normal, scalars on ``CellAvg^3``) and
check the results against the nodal model on the same analytic fields.

Because the FV and nodal 2nd-order stencils are the same numbers
(scoping study, `design/plans/active/fv_nonhydro_scoping.md` §1), the
FV diagnostics are **bitwise identical** to the nodal ones on identical
(midpoint-sampled) input — the acceptance gate is parity, not an
error norm. ``ekin``/``epot`` need only F1's conversion rows; the
``linear_pot_vort`` vorticity is a staggered derivative, so it
additionally needs the C-grid ``diff`` overrides (the FV-D3 flip owned
by stage F3, exercised here as a test-local registry override built
from the *existing* ``FaceDifference``/``FluxDifference`` operators).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.params import (
    CORIOLIS_F0,
    SCALING_NONLINEARITY,
    STRATIFICATION_N2,
)
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.diagnostics import (
    DIAGNOSTICS,
    STRATIFICATION_DIAGNOSTICS,
)
from fridom.nonhydro2.params import ASPECT_RATIO
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.flux_diff import (
    FaceDifference,
    FluxDifference,
)
from fridom.spatial.spaces.tensor_product import TensorProductSpace

N = 16
LENGTH = 2 * np.pi
PARAMS = {
    ASPECT_RATIO: 0.5,
    CORIOLIS_F0: 1.3,
    STRATIFICATION_N2: 2.0,
    SCALING_NONLINEARITY: 0.1,
}
INITS = {
    "u": lambda x, y, z: jnp.sin(x) * jnp.cos(y) * jnp.cos(z),
    "v": lambda x, y, z: jnp.cos(x) * jnp.sin(2 * y) * jnp.cos(z),
    "w": lambda x, y, z: jnp.sin(z) * jnp.cos(x) * jnp.cos(y),
    "p": lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z),
    "b": lambda x, y, z: jnp.sin(x) * jnp.sin(y) * jnp.cos(z),
}


def _meshes():
    return tuple(
        IntervalMesh(N, (0.0, LENGTH), periodic=True, name=name)
        for name in ("x", "y", "z"))


def _prod(*factors):
    return TensorProductSpace.of(*factors)


def _state(grid, spaces):
    return VectorField({
        c: grid.create_field(spaces[c], init=INITS[c], name=c)
        for c in ("u", "v", "w", "p", "b")})


def _nodal_spaces(mx, my, mz):
    return {
        "u": _prod(mx.right, my.center, mz.center),
        "v": _prod(mx.center, my.right, mz.center),
        "w": _prod(mx.center, my.center, mz.right),
        "p": _prod(mx.center, my.center, mz.center),
        "b": _prod(mx.center, my.center, mz.center)}


def _fv_spaces(mx, my, mz):
    # FV-D2 option A: face-normal velocities, scalars on CellAvg^3
    return {
        "u": _prod(mx.right, my.cell_avg, mz.cell_avg),
        "v": _prod(mx.cell_avg, my.right, mz.cell_avg),
        "w": _prod(mx.cell_avg, my.cell_avg, mz.right),
        "p": _prod(mx.cell_avg, my.cell_avg, mz.cell_avg),
        "b": _prod(mx.cell_avg, my.cell_avg, mz.cell_avg)}


def _nodal_state():
    grid = Grid(_meshes())
    return grid, _state(grid, _nodal_spaces(*grid.factors))


def _fv_state(*, cgrid_diff=False):
    meshes = _meshes()
    grid = Grid(meshes)
    if cgrid_diff:
        # the FV-D3 C-grid flip, test-local: the pressure gradient on
        # CellAvg staggers to the face (FaceDifference) and the flux
        # divergence back (FluxDifference) — existing operators, so the
        # vorticity's staggered .diff lands on a common face space
        overrides = {}
        for mesh in meshes:
            overrides[("diff", mesh.cell_avg)] = FaceDifference()
            overrides[("diff", mesh.right)] = FluxDifference()
        grid.merge_overrides(overrides)
    return grid, _state(grid, _fv_spaces(*grid.factors))


# ================================================================
#  b_total: the anomaly plus the ConstantStratification background
# ================================================================
def test_b_total_runs_on_fv_and_matches_nodal():
    diag = STRATIFICATION_DIAGNOSTICS["b_total"]
    _, nodal = _nodal_state()
    _, fv = _fv_state()
    out_nodal = diag(nodal, PARAMS)
    out_fv = diag(fv, PARAMS)
    # lands on the buoyancy space of each family
    assert out_nodal.function_space is nodal["b"].function_space
    assert out_fv.function_space is fv["b"].function_space
    z = np.asarray(nodal["b"].nodes("z").data)
    expected = np.asarray(nodal["b"].data) + PARAMS[STRATIFICATION_N2] * z
    assert np.allclose(np.asarray(out_nodal.data), expected)
    assert np.allclose(np.asarray(out_fv.data),
                       np.asarray(out_nodal.data))
    assert out_nodal.name == "b_total"
    assert out_nodal.xr.attrs["units"] == "m/s^2"


def test_b_total_is_bound_by_the_stratification_module():
    grid = Grid(_meshes())
    model = nh.Model(
        grid=grid, core=nh.Core(aspect_ratio=0.5),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        buoyancy=nh.ConstantStratification(n2=4.0),
        advection=None,
        time_stepper=AdamBashforth(1e-3, order=2))
    total = model.diagnostics.b_total()
    z = np.asarray(model.state["b"].nodes("z").data)
    assert np.allclose(np.asarray(total.data), 4.0 * z)
    assert model.units.factor("b_total") == 1.0


def test_b_total_nondimensional_background_is_n2_z_in_physical_units():
    # Rotational frame: at rest b_total = (eps/Fr^2) z (the advection
    # carries eps, so the background gradient is N^2_eff / eps), and
    # the b_total unit row U^2/(eps delta L) turns it into N^2 z with
    # the Froude definition N = U/(Fr H), H = delta L and z = H z
    length, speed, delta = 2.0e3, 0.5, 0.5
    rossby, froude = 0.25, 0.125
    model = nh.Model(
        grid=Grid(_meshes()), core=nh.Core(aspect_ratio=delta),
        scaling=fr.scaling.Rotational(L=length, U=speed),
        coriolis=nh.FPlaneCoriolis(rossby_number=rossby),
        buoyancy=nh.ConstantStratification(froude_number=froude),
        advection=None,
        time_stepper=AdamBashforth(1e-3, order=2))
    total = model.diagnostics.b_total()
    z = np.asarray(model.state["b"].nodes("z").data)
    assert np.allclose(np.asarray(total.data), rossby / froude**2 * z)
    height = delta * length
    n_freq = speed / (froude * height)
    physical = model.units.factor("b_total") * np.asarray(total.data)
    assert np.allclose(physical, n_freq**2 * height * z)


# ================================================================
#  ekin / epot / etot — need only F1's conversion rows
# ================================================================
@pytest.mark.parametrize("name", ["ekin", "epot", "etot"])
def test_energy_diagnostics_run_on_fv_and_match_nodal(name):
    diag = DIAGNOSTICS[name]
    _, nodal = _nodal_state()
    _, fv = _fv_state()
    out_nodal = diag(nodal, PARAMS)
    out_fv = diag(fv, PARAMS)
    # lands on the pressure cell of each family
    assert out_nodal.function_space is nodal["p"].function_space
    assert out_fv.function_space is fv["p"].function_space
    # 2nd-order consistent — in fact bitwise, per scoping §1
    assert jnp.allclose(np.asarray(out_nodal.data),
                        np.asarray(out_fv.data),
                        rtol=0.0, atol=1e-12)


def test_etot_is_the_sum_of_the_two_energies():
    # the total is exactly ekin + epot on the same cell, so a user no
    # longer needs a derived= callable to write the wave energy out
    _, nodal = _nodal_state()
    total = DIAGNOSTICS["etot"](nodal, PARAMS)
    parts = (DIAGNOSTICS["ekin"](nodal, PARAMS).data
             + DIAGNOSTICS["epot"](nodal, PARAMS).data)
    assert total.function_space is nodal["p"].function_space
    np.testing.assert_array_equal(np.asarray(total.data),
                                  np.asarray(parts))


# ================================================================
#  linear_pot_vort — additionally needs the staggered diff (G5/F3)
# ================================================================
def test_pot_vort_needs_the_staggered_diff_on_fv():
    # without the C-grid diff flip the collocated FV derivative does
    # not stagger, so the vorticity pair never reaches a common edge
    # and is a strict-algebra error: linear_pot_vort is blocked on G5
    # (F3), not on G3/G4. The retag onto the shared vorticity edge is
    # the first step to notice, and it names the offending axis.
    _, fv = _fv_state(cgrid_diff=False)
    with pytest.raises(SpaceMismatchError,
                       match=r"CellAvg\(x\) and Right\(x\)"):
        DIAGNOSTICS["linear_pot_vort"](fv, PARAMS)


def test_pot_vort_runs_on_fv_cgrid_and_matches_nodal():
    diag = DIAGNOSTICS["linear_pot_vort"]
    _, nodal = _nodal_state()
    _, fv = _fv_state(cgrid_diff=True)
    out_nodal = diag(nodal, PARAMS)
    out_fv = diag(fv, PARAMS)
    assert out_fv.function_space is fv["p"].function_space
    assert jnp.allclose(np.asarray(out_nodal.data),
                        np.asarray(out_fv.data),
                        rtol=0.0, atol=1e-12)


# ================================================================
#  linear_pot_vort on a walled horizontal
# ================================================================
@pytest.mark.parametrize(
    ("periodic_x", "periodic_y"),
    [pytest.param(False, True, id="channel-x"),
     pytest.param(True, False, id="channel-y"),
     pytest.param(False, False, id="box-xy")])
def test_pot_vort_runs_on_a_walled_horizontal(periodic_x, periodic_y):
    # regression: the vorticity pair was subtracted at whatever spaces
    # the two differences happened to land on, and a staggered first
    # difference emits a BC-free face while the *other* velocity's
    # factor still carries its wall tag. The two therefore disagreed on
    # every walled axis and the subtraction raised -- linear_pot_vort
    # was periodic-horizontal only. Both differences now retag onto the
    # shared free-slip edge (nh.State.rel_vort_z).
    grid = Grid((
        IntervalMesh(8, (0.0, LENGTH), periodic=periodic_x, name="x"),
        IntervalMesh(8, (0.0, LENGTH), periodic=periodic_y, name="y"),
        IntervalMesh(8, (0.0, LENGTH), periodic=False, name="z")))
    model = nh.Model(
        grid=grid, core=nh.Core(),
        coriolis=nh.FPlaneCoriolis(f0=1.3),
        buoyancy=nh.ConstantStratification(n2=2.0),
        time_stepper=AdamBashforth(1e-3, order=3))
    rng = np.random.default_rng(5)
    model.set_fields(**{name: rng.standard_normal(model.state[name].shape)
                        for name in ("u", "v", "w", "b")})
    q = model.diagnostics.linear_pot_vort()
    assert q.function_space is model.state["p"].function_space
    assert np.isfinite(np.asarray(q.data)).all()
