"""Advection on mapped and stretched grids: physical flux divergence."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom as fr
from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
    _BiasedFaceReconstruction,
    _CenteredFaceInterpolation,
)
from fridom.model.modules.moving_geometry import MovingGeometry
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.nonhydro2.modules.stratification import (
    ConstantStratification,
)
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)

L = 2 * np.pi
NY = 8
DT = 0.01


# ================================================================
#  Helpers
# ================================================================
def make_grid(nx, lx=L, ny=NY):
    return Grid((
        IntervalMesh(nx, (0.0, lx), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(ny, (0.0, L), name="z"),
    ))


def make_model(nx, advection, *, lx=L, stratified=True, ro=1.0):
    modules = [DynamicalCore(rossby_number=ro)]
    if stratified:
        modules.append(ConstantStratification(n2=1.0))
    modules.append(advection)
    return FrModel(grid=make_grid(nx, lx=lx),
                   modules=tuple(modules),
                   time_stepper=AdamBashforth(DT, order=3))


def centers(nx, lx=L):
    return (np.arange(nx) + 0.5) * (lx / nx)


def faces(nx, lx=L):
    return (np.arange(nx) + 1.0) * (lx / nx)


def advection_tendency(model, cls):
    return model.tendency(model.state, constraints=False,
                          filter=fr.model.term_predicates.owned_by(cls))


def set_random_state(model, seed):
    rng = np.random.default_rng(seed)
    model.set_fields(**{
        c: rng.standard_normal(model.state[c].data.shape)
        for c in ("u", "v", "w", "b")})


# ================================================================
#  Mapped grids: the physical flux divergence (stage C4)
# ================================================================
H0 = 0.7


def depth(x):
    """Smooth periodic water depth H(x) (20% slope)."""
    return 1.0 + 0.2 * jnp.sin(x)


def make_mapped_grid(n, init=depth, ny=4, periodic_column=False):
    """Terrain-following grid ``zp = z * H(x)`` (z in [0, 1])."""
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": init})
    return Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=periodic_column,
                     name="z"),
    ), mapping=mapping)


def make_mapped_model(n, advection, *, init=depth, ny=4):
    return FrModel(
        grid=make_mapped_grid(n, init=init, ny=ny),
        modules=(DynamicalCore(),
                 ConstantStratification(n2=0.0),
                 advection),
        time_stepper=AdamBashforth(DT, order=3))


def test_mapped_flat_advection_matches_the_flat_grid():
    # constant H: every slope metric is exactly zero and the column
    # scaling folds to 1/H0, so the mapped flux divergence equals
    # the flat grid's advection tendency to rounding (measured: u,
    # v, w bitwise; b one ulp)
    n = 8
    flat = FrModel(
        grid=Grid((
            IntervalMesh(n, (0.0, L), name="x"),
            IntervalMesh(n, (0.0, L), name="y"),
            IntervalMesh(n, (0.0, H0), periodic=False, name="z"),
        )),
        modules=(DynamicalCore(), ConstantStratification(n2=0.0),
                 CenteredAdvection()),
        time_stepper=AdamBashforth(DT, order=3))
    mapped = make_mapped_model(
        n, CenteredAdvection(), init=lambda x: H0 + 0.0 * x, ny=n)
    hor = centers(n)
    ver = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    fields = {
        "u": 0.3 + 0.1 * np.sin(y),
        "v": 0.2 * np.cos(x),
        "b": np.sin(x) * np.cos(np.pi * z),
    }
    flat.set_fields(**fields)
    mapped.set_fields(**fields)
    tf = advection_tendency(flat, CenteredAdvection)
    tm = advection_tendency(mapped, CenteredAdvection)
    for c in ("u", "v", "w", "b"):
        a = np.asarray(tf[c].data)
        b = np.asarray(tm[c].data)
        scale = max(np.abs(a).max(), 1e-30)
        assert np.abs(a - b).max() <= 1e-14 * scale, c


def test_mapped_transport_converges_at_second_order():
    # uniform physical flow over the sloped column: b = cos(x) zp
    # depends on the PHYSICAL height, so the honest transport is
    # -U db/dx|_zp = U sin(x) zp; the computational derivative
    # alone would be off at O(1). Measured errors 1.43e-2, 3.75e-3,
    # 9.54e-4 at n = 16, 32, 64 — orders 1.93, 1.98.
    U = 0.4
    errors = []
    for n in (16, 32, 64):
        model = make_mapped_model(n, CenteredAdvection())
        hor = centers(n)
        hory = centers(4)
        ver = (np.arange(n) + 0.5) / n
        x, _, z = np.meshgrid(hor, hory, ver, indexing="ij")
        zp = z * depth(x)
        model.set_fields(u=U + 0 * x, b=np.cos(x) * zp)
        tau = advection_tendency(model, CenteredAdvection)
        exact = U * np.sin(x) * zp
        errors.append(
            np.abs(np.asarray(tau["b"].data) - exact).max())
    orders = np.log2(np.asarray(errors[:-1])
                     / np.asarray(errors[1:]))
    assert np.all(orders > 1.8)


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
def test_mapped_grid_is_a_taught_error_for_biased_schemes(cls):
    # a PERIODIC mapped column isolates the mapped rejection from
    # the walled one: the biased reconstructions are computational-
    # coordinate rows — future work
    grid = make_mapped_grid(8, periodic_column=True)
    with pytest.raises(NotImplementedError,
                       match=r"does not support mapped grids"
                             r".*CenteredAdvection"):
        FrModel(grid=grid,
                modules=(DynamicalCore(), cls(3)),
                time_stepper=AdamBashforth(DT, order=3))


# ================================================================
#  Stretched meshes: taught rejection of the biased schemes at bind
# ================================================================
def wavy_map(s):
    """Smooth wavy stretching of the unit computational interval."""
    return s + 0.1 * jnp.sin(2.0 * jnp.pi * s) / (2.0 * jnp.pi)


def make_stretched_grid(n=8, ny=NY):
    """Periodic grid whose z factor is a stretched (mapped) mesh."""
    return Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        MappedIntervalMesh(n, (0.0, 1.0), wavy_map, periodic=True,
                           name="z"),
    ))


@pytest.mark.parametrize("cls", [UpwindAdvection, WENOAdvection])
@pytest.mark.parametrize("order", [3, 5])
def test_stretched_mesh_is_a_taught_error_for_biased_schemes(
        cls, order):
    # a plain MappedIntervalMesh declares NO CoordinateMapping, so
    # column_corrections is empty: before the mapped_factor() guard
    # the biased schemes bound happily here and silently dropped to
    # 2nd order (measured: upwind-5 and weno-5 both 5.0 -> 2.0)
    grid = make_stretched_grid()
    with pytest.raises(
            NotImplementedError,
            match=r"does not support stretched \(mapped\) meshes"
                  r".*'z'.*uniform-offset.*silently drop to 2nd "
                  r"order.*CenteredAdvection"):
        FrModel(grid=grid,
                modules=(DynamicalCore(), cls(order)),
                time_stepper=AdamBashforth(DT, order=3))


def test_stretched_mesh_binds_the_centered_scheme():
    # the guard must not over-fire: the centered scheme's two-point
    # stencils divide by the codomain measure field (order 2) and
    # stay grounded on a stretched mesh. (A full nh Model on a plain
    # stretched axis is a separate deferral — the pressure solver
    # wants the spectral transform MappedIntervalMesh refuses (C2) —
    # so this exercises the module's own bind seam, as the
    # two-mapped-columns test does.)
    module = CenteredAdvection()
    module._bind_mapping(make_stretched_grid())  # no raise
    assert module.extra_halo is None  # no mapped column here


@pytest.mark.parametrize("op", [
    pytest.param(_BiasedFaceReconstruction(3, "left", "weno"),
                 id="biased"),
    pytest.param(_CenteredFaceInterpolation(4), id="centered-face"),
])
def test_biased_face_kernels_reject_a_stretched_factor(op):
    # the operator-level twin of the bind guard (direct misuse)
    mesh = MappedIntervalMesh(8, (0.0, 1.0), wavy_map,
                              periodic=True, name="z")
    with pytest.raises(SpaceMismatchError,
                       match=r"uniform-mesh only.*uniform-offset"):
        op.codomain(mesh.center)


def test_two_mapped_columns_are_a_taught_error():
    # two single-base analytic maps (parameter-free H keeps their
    # coupled coordinate sets disjoint) exceed the stage-C4 support
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H,
              "yp": lambda y, YN: y * YN},
        params={"H": lambda: 0.8, "YN": lambda: 0.9})
    grid = Grid((
        IntervalMesh(8, (0.0, L), name="x"),
        IntervalMesh(8, (0.0, 1.0), name="y"),
        IntervalMesh(8, (0.0, 1.0), name="z"),
    ), mapping=mapping)
    module = CenteredAdvection()
    with pytest.raises(NotImplementedError,
                       match="exactly one mapped column"):
        module._bind_mapping(grid)


def test_mapped_halo_substitute_and_flat_none():
    # the mapped flux divergence multiplies grid.metric coefficients
    # the halo tracer cannot follow: two cells per coordinate; the
    # flat path declares nothing (fully halo-traced, pre-C4)
    module = CenteredAdvection()
    make_mapped_model(8, module)
    assert module.extra_halo is not None
    assert dict(module.extra_halo.widths) == {
        "x": 2, "y": 2, "z": 2}
    flat_module = CenteredAdvection()
    make_model(8, flat_module)
    assert flat_module.extra_halo is None


def test_mapped_background_split_telescopes():
    # both background call sites on the mapped grid: the two-term
    # sum at Ro = 1 equals the no-background module on the combined
    # advecting velocity (the centered hooks are linear in the
    # velocity). The identity holds for the components whose
    # ADVECTED field carries no background (v, w, b): the split
    # transports the perturbation u itself (old-stack convention),
    # so the u rows advect different fields by design — asserted
    # finite only.
    n = 8

    def u_bg(y):
        return 1.0 + 0.5 * np.sin(y)

    module = CenteredAdvection(background={"u": u_bg})
    model = make_mapped_model(n, module, ny=n)
    set_random_state(model, seed=21)
    total = advection_tendency(model, CenteredAdvection)

    combined = make_mapped_model(n, CenteredAdvection(), ny=n)
    hor = centers(n)
    y = np.meshgrid(faces(n), hor, (np.arange(n) + 0.5) / n,
                    indexing="ij")[1]
    state = model.state
    combined.set_fields(
        u=np.asarray(state["u"].data) + u_bg(y),
        v=np.asarray(state["v"].data),
        w=np.asarray(state["w"].data),
        b=np.asarray(state["b"].data))
    want = advection_tendency(combined, CenteredAdvection)
    for c in ("v", "w", "b"):
        scale = max(np.abs(np.asarray(want[c].data)).max(), 1e-30)
        np.testing.assert_allclose(
            np.asarray(total[c].data), np.asarray(want[c].data),
            rtol=0, atol=1e-13 * scale)
    assert np.isfinite(np.asarray(total["u"].data)).all()


def test_mapped_advection_reads_the_current_geometry():
    # the dynamic-params seam: a MovingGeometry frozen at a depth
    # DIFFERENT from the grid's static default drives the advection
    # metrics — the tendency matches a static grid built at that
    # depth, and differs from the static-default tendency
    def other(x):
        return 1.0 + 0.1 * jnp.cos(2.0 * x)

    n = 8
    moving = FrModel(
        grid=make_mapped_grid(n),
        modules=(DynamicalCore(), ConstantStratification(n2=0.0),
                 CenteredAdvection(),
                 MovingGeometry(
                     {"H": lambda x, t: other(x) + 0.0 * t})),
        time_stepper=AdamBashforth(DT, order=3))
    static = make_mapped_model(n, CenteredAdvection(), init=other)
    default = make_mapped_model(n, CenteredAdvection())
    hor = centers(n)
    hory = centers(4)
    ver = (np.arange(n) + 0.5) / n
    x, _, z = np.meshgrid(hor, hory, ver, indexing="ij")
    fields = {"u": 0.4 + 0 * x, "b": np.cos(x) * z}
    for model in (moving, static, default):
        model.set_fields(**fields)
    got = advection_tendency(moving, CenteredAdvection)
    want = advection_tendency(static, CenteredAdvection)
    other_t = advection_tendency(default, CenteredAdvection)
    np.testing.assert_allclose(
        np.asarray(got["b"].data), np.asarray(want["b"].data),
        rtol=0, atol=1e-14)
    assert not np.allclose(np.asarray(got["b"].data),
                           np.asarray(other_t["b"].data))

