"""Gates for the immersed (cut-cell) nonhydro2 model (stage I2).

The end-to-end model gates: an auto-FV immersed model with the shared
``MaskState`` keeps dry DOFs dead (gate g), a face-aligned {0, 1} box
matches the walled FV model to machine zero (gate c, staircase
equivalence), an all-wet immersed model reproduces the unimmersed run
(gate d), the theta-weighted tracer is conserved to machine zero
(gate f), and the family / advection taught errors are pinned (gate h).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro2 as nh
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.nonhydro2.modules.core import resolve_model_family
from fridom.nonhydro2.modules.immersed_pressure import (
    ImmersedPressureSolver,
)
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.spaces.average import AverageSpace

TWO_PI = 2.0 * np.pi


def _periodic(n=12, length=TWO_PI):
    return tuple(
        IntervalMesh(n, (0.0, length), periodic=True, name=nm)
        for nm in ("x", "y", "z"))


def _is_fv(model):
    return any(
        isinstance(f, AverageSpace)
        for f in model.state["w"].function_space.bare.factors)


def _warm_start_box_model():
    """Build a small immersed FV model with a random provisional IC."""
    n = 12
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = Grid(_periodic(n, length=TWO_PI),
                immersed=ImmersedDomain(box))
    model = nh.Model(grid=grid, dt=0.01, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0),
                     pressure_iterations=40)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    return model


def test_warm_start_matches_zero_start_over_a_run(monkeypatch):
    # Phase E (GE-1) at the model level: warm start is always-on, so a
    # forced zero-start (x0 dropped in the immersed projection) is the
    # reference. Over a multi-step run the two land on the same state
    # to the pressure-tolerance level — warm start never changes the
    # converged solution, only the achieved iteration count.
    warm = _warm_start_box_model()
    warm.advance(3)
    # force the zero start: re-trace after clearing the compilation
    # cache so the monkeypatched project (x0 stripped) is picked up
    jax.clear_caches()
    orig_project = ImmersedPressureSolver.project
    monkeypatch.setattr(
        ImmersedPressureSolver, "project",
        lambda self, vel, x0=None: orig_project(self, vel))  # noqa: ARG005
    cold = _warm_start_box_model()
    cold.advance(3)
    assert not warm.panicked
    assert not cold.panicked
    for name in ("u", "v", "w", "b"):
        w = np.asarray(warm.state[name].data)
        c = np.asarray(cold.state[name].data)
        scale = float(np.abs(c).max())
        assert float(np.abs(w - c).max()) <= 1e-6 * scale


# ================================================================
#  Family policy (IP-D7)
# ================================================================
def test_immersed_grid_auto_flips_to_fv():
    grid = Grid(_periodic(), immersed=ImmersedDomain(
        lambda x, y, z: x * 0.0 + 1.0))  # noqa: ARG005
    assert resolve_model_family(None, grid) == "fv"


def test_explicit_nodal_on_immersed_is_a_taught_error():
    grid = Grid(_periodic(), immersed=ImmersedDomain(
        lambda x, y, z: x * 0.0 + 1.0))  # noqa: ARG005
    with pytest.raises(NotImplementedError, match="finite-volume"):
        resolve_model_family("nodal", grid)


def test_immersed_model_installs_maskstate_and_is_fv():
    grid = Grid(_periodic(), immersed=ImmersedDomain(
        lambda x, y, z: x * 0.0 + 1.0))  # noqa: ARG005
    model = nh.Model(grid=grid, dt=0.02, advection=False,
                     coriolis=FPlaneCoriolis(f0=1.0))
    assert _is_fv(model)
    assert any(type(m).__name__ == "MaskState" for m in model.modules)


def test_mapped_plus_immersed_fv_is_a_taught_error():
    # the one FV combination iteration 2 does not serve: a grid that
    # declares both a terrain-following mapped column and an immersed
    # domain (plan §6) — the mapped and masked PCGs are not composed
    from fridom.spatial.coordinate_mapping import (  # noqa: PLC0415
        CoordinateMapping,
    )
    mapping = CoordinateMapping(
        maps={"zp": lambda z, h: z * h},
        params={"h": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    grid = Grid(
        (IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="x"),
         IntervalMesh(8, (0.0, TWO_PI), periodic=True, name="y"),
         IntervalMesh(8, (0.0, 1.0), periodic=False, name="z")),
        mapping=mapping,
        immersed=ImmersedDomain(
            lambda x, y, z: x * 0.0 + 1.0))  # noqa: ARG005
    with pytest.raises(NotImplementedError, match="both"):
        resolve_model_family("fv", grid)


@pytest.mark.parametrize(
    "advection", [nh.UpwindAdvection(3), nh.WENOAdvection(5)])
def test_biased_advection_on_immersed_binds_and_steps(advection):
    # the biased schemes gained the mask-keyed graded closure (GA-D6):
    # an immersed nonhydro2 model with UpwindAdvection/WENOAdvection binds
    # (the mask path replaces the wall closure) and steps finite, keeping
    # the dry-DOF hygiene the fraction weighting guarantees. A fresh grid
    # per scheme — the order-5 halo (3) exceeds the order-3 one (2), so a
    # shared frozen grid would fault the "most demanding model first" rule
    n = 12
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = Grid(_periodic(n), immersed=ImmersedDomain(box))
    model = nh.Model(grid=grid, dt=0.01, advection=advection,
                     coriolis=FPlaneCoriolis(f0=1.0),
                     pressure_iterations=15)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    model.advance(6)
    assert not model.panicked
    for name in ("u", "v", "w", "b"):
        field = model.state[name]
        mask = grid.immersed.mask(field.function_space)
        dry = np.asarray(field.data) * (1.0 - np.asarray(mask.data))
        assert np.abs(dry).max() == 0.0


# ================================================================
#  Gate g: dry-DOF hygiene over a multi-step run
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_dry_dofs_stay_exactly_zero(advection):
    n = 12
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = Grid(_periodic(n, length=TWO_PI),
                immersed=ImmersedDomain(box))
    model = nh.Model(grid=grid, dt=0.01, advection=advection,
                     coriolis=FPlaneCoriolis(f0=1.0),
                     pressure_iterations=25)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    model.advance(8)
    assert not model.panicked
    immersed = grid.immersed
    for name in ("u", "v", "w", "b"):
        field = model.state[name]
        mask = immersed.mask(field.function_space)
        dry = np.asarray(field.data) * (1.0 - np.asarray(mask.data))
        assert np.abs(dry).max() == 0.0


# ================================================================
#  Gate f: theta-weighted buoyancy conserved to machine zero
# ================================================================
def test_theta_weighted_buoyancy_is_conserved_to_machine_zero():
    n = 12
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = Grid(_periodic(n), immersed=ImmersedDomain(box))
    # n2=0: the buoyancy has no -N^2 w source, so advection alone
    # governs it — a pure conservation check
    model = nh.Model(
        grid=grid, dt=0.01, advection=True,
        coriolis=FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=25)
    rng = np.random.default_rng(3)
    model.set_fields(
        b=rng.standard_normal(model.state["b"].data.shape),
        **{k: 0.2 * rng.standard_normal(model.state[k].data.shape)
           for k in ("u", "v", "w")})
    theta = grid.immersed.fraction(model.state["b"].function_space)

    def total():
        return float(jnp.sum(
            (theta * model.state["b"]).integrate().data))

    before = total()
    model.advance(20)
    assert not model.panicked
    after = total()
    assert abs(after - before) <= 1e-13 * max(abs(before), 1.0)


# ================================================================
#  Gate d: all-wet immersed reproduces the unimmersed run
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_all_wet_immersed_matches_unimmersed(advection):
    n = 10
    meshes = lambda: (  # noqa: E731
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"))
    allwet = ImmersedDomain(lambda x, y, z: x * 0.0 + 1.0)  # noqa: ARG005
    im = nh.Model(grid=Grid(meshes(), immersed=allwet), dt=0.02,
                  advection=advection, coriolis=FPlaneCoriolis(f0=1.0),
                  pressure_iterations=3)
    un = nh.Model(grid=Grid(meshes()), dt=0.02, advection=advection,
                  coriolis=FPlaneCoriolis(f0=1.0))
    rng = np.random.default_rng(7)
    ic = {k: 0.3 * rng.standard_normal(im.state[k].data.shape)
          for k in ("u", "v", "w", "b")}
    im.set_fields(**ic)
    un.set_fields(**ic)
    im.advance(12)
    un.advance(12)
    for k in ("u", "v", "w", "b", "p"):
        diff = np.abs(np.asarray(im.state[k].data)
                      - np.asarray(un.state[k].data)).max()
        assert diff < 1e-11, (k, diff)


# ================================================================
#  Gate c: staircase equivalence (immersed box vs walled FV model)
# ================================================================
@pytest.mark.parametrize("advection", [False, True])
def test_face_aligned_box_matches_the_walled_fv_model(advection):
    # a face-aligned {0, 1} box (cells 3..9) in a 12^3 periodic grid
    # of unit spacing vs the walled FV model on the 6^3 wet box: the
    # wet-region trajectories agree to the CG residual (measured
    # ~1e-16 — bit-comparable to walls) over 12 jitted steps.
    npg, lo = 12, 3
    box = lambda x, y, z: (  # noqa: E731
        (x > lo) & (x < 9) & (y > lo) & (y < 9)
        & (z > lo) & (z < 9)).astype(float)
    imm = nh.Model(
        grid=Grid(tuple(
            IntervalMesh(npg, (0.0, 12.0), periodic=True, name=nm)
            for nm in ("x", "y", "z")),
            immersed=ImmersedDomain(box)),
        dt=0.02, advection=advection, coriolis=FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=60)
    wal = nh.Model(
        grid=Grid(tuple(
            IntervalMesh(6, (3.0, 9.0), periodic=False, name=nm)
            for nm in ("x", "y", "z"))),
        dt=0.02, advection=advection, coriolis=FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        pressure_iterations=1, family="fv")
    rng = np.random.default_rng(11)
    shapes = {k: wal.state[k].data.shape for k in ("u", "v", "w")}
    ic = {k: 0.1 * rng.standard_normal(shapes[k])
          for k in ("u", "v", "w")}
    bic = 0.1 * rng.standard_normal((6, 6, 6))
    wal.set_fields(b=bic, **ic)
    full = {}
    for k in ("u", "v", "w"):
        arr = np.zeros((npg, npg, npg))
        s = shapes[k]
        arr[lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]] = ic[k]
        full[k] = arr
    barr = np.zeros((npg, npg, npg))
    barr[lo:9, lo:9, lo:9] = bic
    imm.set_fields(b=barr, **full)
    imm.advance(12)
    wal.advance(12)
    for name in ("u", "v", "w", "b"):
        s = wal.state[name].data.shape
        sub = np.asarray(imm.state[name].data)[
            lo:lo + s[0], lo:lo + s[1], lo:lo + s[2]]
        diff = np.abs(sub - np.asarray(wal.state[name].data)).max()
        assert diff < 1e-10, (name, diff)


# ================================================================
#  Multigrid pressure preconditioner (B3/B4) — production wiring
# ================================================================
def test_multigrid_preconditioner_model_runs_end_to_end():
    # the pressure_preconditioner='multigrid' knob threads the factory
    # -> DynamicalCore -> ImmersedPressureSolver and assembles the
    # V-cycle on the FROZEN grid (the coarse-sibling hierarchy) — a full
    # immersed model steps finite and non-panicked (a bounded-z box so
    # the vertical-line smoother has a Neumann column)
    n = 12
    meshes = (
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x"),
        IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y"),
        IntervalMesh(8, (0.0, 1.0), periodic=False, name="z"))
    box = lambda x, y, z: (  # noqa: E731
        (x > 1.0) & (x < 5.0) & (y > 1.0) & (y < 5.0)
        & (z > 0.2) & (z < 0.8)).astype(float)
    grid = Grid(meshes, immersed=ImmersedDomain(box))
    model = nh.Model(
        grid=grid, dt=0.01, advection=False,
        coriolis=FPlaneCoriolis(f0=1.0), pressure_iterations=25,
        pressure_preconditioner="multigrid", multigrid_levels=3)
    rng = np.random.default_rng(0)
    model.set_fields(**{
        k: 0.2 * rng.standard_normal(model.state[k].data.shape)
        for k in ("u", "v", "w", "b")})
    model.advance(3)
    assert not model.panicked
    for name in ("u", "v", "w", "p"):
        assert bool(jnp.isfinite(model.state[name].data).all())
