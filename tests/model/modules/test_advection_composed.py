"""Flux-form advection on a composed mapped + immersed grid (M4, MI-D5).

The proof that ``_FluxFormAdvection`` composes its two weightings — the
immersed open-area fraction ``alpha`` on the physical face flux and the
J-weighted mapped divergence — for the first time (they have always
been wired but never both fired). ``CenteredAdvection`` on a genuine cut
chart conserves ``sum_c theta_c J_c V_c q_c`` to machine zero, preserves
a uniform tracer exactly, and reduces to the pure mapped advective run
on an all-wet chart. The biased schemes stay mapped-rejecting.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
N = 8
DT = 0.01


def depth(x):
    """Smooth terrain depth H(x)."""
    return 1.0 + 0.25 * jnp.sin(x)


def cut(x, y, z):  # noqa: ARG001
    """Return a smooth immersed obstacle carving partial cells."""
    return jnp.clip((z - 0.15 - 0.1 * jnp.sin(x)) * 6 + 0.5, 0.0, 1.0)


def allwet(x, y, z):
    """Return an all-wet indicator (theta == 1 everywhere)."""
    return x * 0.0 + y * 0.0 + z * 0.0 + 1.0


def composed_grid(n=N, init=depth, ind=cut, order=4):
    """Terrain chart zp = z H(x) + an immersed cut (both descriptors)."""
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": init})
    return Grid((mx, my, mz), mapping=mapping,
                immersed=ImmersedDomain(ind, order=order, min_fraction=0.1))


def composed_model(grid, **kw):
    """Return a composed nh.Model (advection + a converging PCG)."""
    return nh.Model(
        grid=grid,
        core=nh.Core(
            aspect_ratio=(0.5) ** 0.5,
            pressure_iterations=30,
            pressure_tolerance=None,
            pressure_preconditioner="multigrid"),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        advection=True,
        **kw)


def set_random(model, seed=0):
    """Seed a random (u, v, w, b) state on the model's spaces."""
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(model.state[c].data.shape)
        for c in ("u", "v", "w", "b")})


# ================================================================
#  Conservation: sum theta J V q to machine zero (>= 10 steps)
# ================================================================
def test_theta_j_weighted_tracer_is_conserved_to_machine_zero():
    # n2 = 0, so advection alone governs b; on a chart .integrate() is
    # J-weighted, so (theta * b).integrate() is exactly sum theta J V b
    grid = composed_grid()
    model = composed_model(grid)
    set_random(model, seed=1)
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def total():
        return float(jnp.sum((theta * model.state["b"]).integrate().data))

    before = total()
    model.advance(12)
    assert not model.panicked
    after = total()
    assert abs(after - before) <= 1e-12 * max(abs(before), 1.0)


# ================================================================
#  Cross-gate: the varying-H cross term still conserves (the M4 fix)
# ================================================================
def test_conservation_holds_with_active_metric_cross_terms():
    # the freestream / constancy proof in conservative form: with a
    # genuinely varying H(x) the mapped cross term is active, and the
    # composed flux-form conserves sum theta J V q to machine zero only
    # because the reduced cross flux is gated by the base-face open
    # fraction (the MI-D5 cross-cell closure, _mapped_fv_cross). Left
    # un-gated the cross leaks at wet/dry vertical interfaces (the drift
    # this pins against).
    grid = composed_grid(init=lambda x: 1.0 + 0.35 * jnp.sin(x))
    model = composed_model(grid)
    set_random(model, seed=5)
    theta = grid.immersed.fraction(model.state["b"].function_space)
    total = lambda: float(  # noqa: E731
        jnp.sum((theta * model.state["b"]).integrate().data))
    before = total()
    model.advance(10)
    assert not model.panicked
    assert abs(total() - before) <= 1e-12 * max(abs(before), 1.0)


# ================================================================
#  All-wet chart == pure mapped advective run
# ================================================================
def test_all_wet_chart_advection_matches_the_pure_mapped_run():
    # with alpha == 1 the composed advection is the mapped advection: an
    # all-wet composed model and a pure mapped model (same IC) agree
    grid_c = composed_grid(ind=allwet)
    model_c = composed_model(grid_c)
    mx = IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(N, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(N, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": depth})
    grid_m = Grid((mx, my, mz), mapping=mapping)
    model_m = nh.Model(
        grid=grid_m,
        core=nh.Core(
            aspect_ratio=(0.5) ** 0.5,
            pressure_iterations=30,
            pressure_tolerance=None,
            pressure_preconditioner="multigrid"),
        time_stepper=AdamBashforth(DT, order=3),
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        advection=True)
    rng = np.random.default_rng(3)
    fields = {c: rng.standard_normal(model_c.state[c].data.shape)
              for c in ("u", "v", "w", "b")}
    model_c.set_fields(**fields)
    model_m.set_fields(**fields)
    model_c.advance(8)
    model_m.advance(8)
    assert not model_c.panicked
    for c in ("u", "v", "w", "b"):
        a = np.asarray(model_c.state[c].data)
        b = np.asarray(model_m.state[c].data)
        scale = max(np.abs(b).max(), 1.0)
        assert np.abs(a - b).max() <= 1e-9 * scale, c


# ================================================================
#  Biased / WENO stay mapped-rejecting on a composed grid
# ================================================================
@pytest.mark.parametrize(
    "advection", [nh.UpwindAdvection(3), nh.WENOAdvection(5)])
def test_biased_advection_still_rejects_a_composed_grid(advection):
    # the biased face reconstructions are computational-coordinate rows
    # (a taught error on any mapped column, composed or not)
    grid = composed_grid()
    with pytest.raises(NotImplementedError,
                       match="does not support mapped grids"):
        nh.Model(
            grid=grid,
            core=nh.Core(pressure_preconditioner="multigrid"),
            time_stepper=AdamBashforth(DT, order=3),
            stratification=nh.ConstantStratification(n2=1.0),
            advection=advection)
