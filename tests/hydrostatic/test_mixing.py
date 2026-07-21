"""Hydrostatic integration tests for the vertical-mixing closure.

Covers the composition of the mixing tridiagonal solve with the
implicit free-surface constraint under CNAB2 (both effects present,
treedef stable across the run), the u/v/b target resolution (the
diagnosed ``w`` is never a mixing target), and a short EXPLICIT-path
integration under Adam-Bashforth staying finite.
"""
import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
import fridom.hydrostatic as hy
from fridom.model.closures.vertical_mixing import VerticalMixing
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.operators.integrate import Integral

IM = fr.spatial.meshes.IntervalMesh
NX, NZ, DEPTH = 4, 16, 1.0


# ================================================================
#  Builders
# ================================================================
def make_grid(nx=NX, nz=NZ, depth=DEPTH):
    """Return a doubly-periodic (x, y), bounded-z channel grid."""
    return fr.spatial.Grid((
        IM(nx, (0.0, 1.0), periodic=True, name="x"),
        IM(nx, (0.0, 1.0), periodic=True, name="y"),
        IM(nz, (0.0, depth), periodic=False, name="z")))


def depth_mean_divergence(state):
    """Max |Integral_z(du/dx + dv/dy)| — the barotropic divergence."""
    div = state["u"].diff("x") + state["v"].diff("y")
    return float(jnp.max(jnp.abs(Integral()["z"](div).data)))


def randomize(model, names, seed=1):
    """Seed the model state with random data on the given components."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        name: rng.standard_normal(model.state[name].shape)
        for name in names})


# ================================================================
#  Composition with the implicit free surface under CNAB2
# ================================================================
def test_mixing_with_implicit_free_surface_under_cnab2():
    dt = 0.02
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=100.0),
        time_stepper=fr.model.time_steppers.CNAB2(dt),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ImplicitFreeSurface(),
        advection=False,
        modules_extra=(VerticalMixing(kv=0.03, kb=0.05),))
    randomize(model, ("u", "v", "b", "ps"))

    bmax0 = float(jnp.max(jnp.abs(model.state["b"].data)))
    div0 = depth_mean_divergence(model.state)
    treedef0 = jax.tree_util.tree_structure(
        model._stepper.init(model.state))

    model.run(steps=5, progress=False)

    treedef1 = jax.tree_util.tree_structure(
        model._stepper.init(model.state))
    bmax1 = float(jnp.max(jnp.abs(model.state["b"].data)))
    div1 = depth_mean_divergence(model.state)

    # (a) mixing active: buoyancy has changed from its initial max
    assert abs(bmax1 - bmax0) > 1e-6
    # (b) surface active: the barotropic divergence is REDUCED (an
    # epsilon=1 implicit free surface reduces it, does not zero it)
    assert div1 < div0
    # treedef stability across the advance (a bitwise-restartable carry)
    assert treedef0 == treedef1


# ================================================================
#  Targets u, v, b (the diagnosed w is not a mixing target)
# ================================================================
def test_mixing_targets_are_the_velocities_and_buoyancy():
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=100.0),
        time_stepper=fr.model.time_steppers.CNAB2(0.02),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False,
        modules_extra=(VerticalMixing(kv=0.03, kb=0.05),))
    merged = model._artifacts.schedule.implicit_merged
    assert len(merged) == 1
    operator, _slot = merged[0]
    assert operator.fields == ("u", "v", "b")
    assert "w" not in operator.fields


# ================================================================
#  A short EXPLICIT-path integration stays finite
# ================================================================
def test_explicit_mixing_run_stays_finite_under_adam_bashforth():
    dt = 0.001
    model = hy.Model(
        grid=make_grid(),
        core=hy.Core(gravity=100.0),
        time_stepper=AdamBashforth(dt, order=3),
        coriolis=hy.FPlaneCoriolis(f0=0.5),
        stratification=hy.ConstantStratification(n2=1.0),
        free_surface=hy.ExplicitFreeSurface(),
        advection=False,
        modules_extra=(VerticalMixing(
            kv=0.03, kb=0.05, treatment=fr.model.EXPLICIT),))
    randomize(model, ("u", "v", "b"), seed=2)
    model.run(steps=50, progress=False)
    for name in ("u", "v", "b"):
        assert np.all(np.isfinite(np.asarray(model.state[name].data)))
