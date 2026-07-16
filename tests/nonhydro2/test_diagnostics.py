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

from fridom.model.params import (
    CORIOLIS_F0,
    SCALING_ROSSBY,
    STRATIFICATION_N2,
)
from fridom.nonhydro2.diagnostics import DIAGNOSTICS
from fridom.nonhydro2.params import DSQR
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
    DSQR: 0.25,
    CORIOLIS_F0: 1.3,
    STRATIFICATION_N2: 2.0,
    SCALING_ROSSBY: 0.1,
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
#  ekin / epot — need only F1's conversion rows
# ================================================================
@pytest.mark.parametrize("name", ["ekin", "epot"])
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


# ================================================================
#  linear_pot_vort — additionally needs the staggered diff (G5/F3)
# ================================================================
def test_pot_vort_needs_the_staggered_diff_on_fv():
    # without the C-grid diff flip the collocated FV derivative does
    # not stagger, so v.diff("x") - u.diff("y") is a strict-algebra
    # error: linear_pot_vort is blocked on G5 (F3), not on G3/G4
    _, fv = _fv_state(cgrid_diff=False)
    with pytest.raises(SpaceMismatchError, match="combine spaces"):
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
