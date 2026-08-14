"""Advection on mapped and stretched grids: physical flux divergence."""

import jax
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
    _safe_ratio,
)
from fridom.model.modules.moving_geometry import MovingGeometry
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.nonhydro2.modules.core import Core
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


def make_model(nx, advection, *, lx=L, stratified=True, eps=None):
    """Assemble the advection module; ``eps`` selects nondim (fr.scaling).

    ``eps=None`` is the dimensional assembly (the de-scaled advection
    carries zero scaling ops). A float assembles the NONDIMENSIONAL
    variant through the InternalWave frame: the stratification owns
    the epsilon leaf (froude_number=eps) and the scaling-neutral
    advection adopts the variant at bind, so the tendency carries one
    outer epsilon (and, with a background, U + eps*u' inside).
    """
    modules = [Core()]
    scaling = None
    if eps is not None:
        modules.append(ConstantStratification(froude_number=eps))
        scaling = fr.scaling.InternalWave()
    elif stratified:
        modules.append(ConstantStratification(n2=1.0))
    modules.append(advection)
    return FrModel(grid=make_grid(nx, lx=lx),
                   modules=tuple(modules),
                   time_stepper=AdamBashforth(DT, order=3),
                   scaling=scaling)


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
        modules=(Core(),
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
        modules=(Core(), ConstantStratification(n2=0.0),
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
    # alone would be off at O(1). Measured interior errors 1.34e-2,
    # 3.64e-3, 9.39e-4 at n = 16, 32, 64 — orders 1.88, 1.95.
    #
    # The order is measured over the INTERIOR rows because this
    # manufactured state is not admissible AT the sloping lid: with
    # w = 0 its contravariant column flux Omega = w - Z_x u equals
    # -Z_x U != 0 there, i.e. the "exact" solution transports b
    # straight through the rigid lid. The wall closure structurally
    # imposes the zero advective wall flux (Omega q)|_wall = 0 (A0,
    # `_mapped_nodal_cross`), so the top row measures that closure,
    # not the manufactured interior solution. The admissible
    # (Omega == 0) counterpart is
    # test_mapped_column_flux_closes_on_the_contravariant_flux.
    #
    # This restriction is not a weakening: before A0 the assertion ran
    # over the FULL domain and passed, top row included (measured
    # 1.43e-2, 3.75e-3, 9.54e-4 — 2nd order), which is exactly the
    # symptom. A scheme with no wall closure reproduces a reference
    # solution that flows through the wall, so the old full-domain
    # assertion was pinning the ABSENCE of the closure. The interior
    # numbers are unchanged by A0 to three digits (pre-fix 1.3362e-2,
    # 3.6337e-3, 9.3861e-4), so the interior rows carry the same
    # evidence they always did; the wall row is covered by the two
    # tests below.
    U = 0.4
    errors, tops = [], []
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
        err = np.abs(np.asarray(tau["b"].data) - exact)
        errors.append(err[:, :, 1:-1].max())
        tops.append(err[:, :, -1].max())
    orders = np.log2(np.asarray(errors[:-1])
                     / np.asarray(errors[1:]))
    assert np.all(orders > 1.8)
    # the error constant, not only the order (a regression that keeps
    # 2nd order but inflates the constant would slip past `orders`)
    assert errors[0] < 2e-2
    # and the excluded top row really is the wall closure refusing the
    # inadmissible reference: its "error" GROWS like the missing wall
    # flux Omega q / (J dz), i.e. like 1/h (measured 1.16, 2.50, 5.09)
    assert tops[2] > tops[1] > tops[0] > 1.0


# ================================================================
#  The sloping-wall column flux closure (A0)
# ================================================================
def omega_free_flow(n, ny=4):
    r"""Analytic (u, w, b) with a vanishing contravariant column flux.

    The stream function ``psi = f(z)``, ``f(z) = sin(pi z)``, has
    ``f(0) = f(1) = 0``, so both walls of ``zp = z H(x)`` are exact
    streamlines: ``u = f'(z)/H``, ``w = f'(z) z H'/H`` is physically
    divergence-free AND its contravariant column flux
    ``Omega = w - (dzp/dx) u`` vanishes IDENTICALLY. Advecting
    ``b = zp`` then has the closed form tendency ``-w``.
    """
    dx = L / n
    xc = (np.arange(n) + 0.5) * dx
    xr = (np.arange(n) + 1.0) * dx
    zc = (np.arange(n) + 0.5) / n
    zi = np.arange(1, n) / n                  # Inner(z): interior faces
    ones = np.ones((1, ny, 1))

    def h(x):
        return 1.0 + 0.2 * np.sin(x)

    def hp(x):
        return 0.2 * np.cos(x)

    x, z = np.meshgrid(xr, zc, indexing="ij")
    u = (np.pi * np.cos(np.pi * z) / h(x))[:, None, :] * ones
    x, z = np.meshgrid(xc, zi, indexing="ij")
    w = (np.pi * np.cos(np.pi * z) * z * hp(x) / h(x))[:, None, :] * ones
    x, z = np.meshgrid(xc, zc, indexing="ij")
    b = (z * h(x))[:, None, :] * ones
    exact = -(np.pi * np.cos(np.pi * z) * z
              * hp(x) / h(x))[:, None, :] * ones
    return u, w, b, exact


def test_mapped_column_flux_closes_on_the_contravariant_flux():
    # A0's mechanism, isolated. The wall-normal velocity of a
    # terrain-following column is the CONTRAVARIANT flux
    # Omega = w - Z_x u, not the Cartesian w: at a sloping wall
    # w = Z_x u != 0. The pre-A0 product-rule column correction
    # -(Z_x/J) I(d_z F_x) could not cancel the `axis == base` term's
    # structural wall zero, so it left the wall flux Z_x u q
    # unbalanced — an O(1/h) term in the boundary-adjacent row
    # (measured top-row errors 10.3, 20.4, 41.0 at n = 16, 32, 64,
    # i.e. DIVERGING, against a 2nd-order interior). The J-weighted
    # flux form D_b(Z_i I_b(F_i)) closes on Omega instead: the top
    # row now converges (0.283, 0.149, 0.0758).
    tops, interiors = [], []
    for n in (16, 32):
        model = make_mapped_model(n, CenteredAdvection())
        u, w, b, exact = omega_free_flow(n)
        model.set_fields(u=u, v=0.0 * b, w=w, b=b)
        tau = advection_tendency(model, CenteredAdvection)
        err = np.abs(np.asarray(tau["b"].data) - exact).max(axis=(0, 1))
        tops.append(err[-1])
        interiors.append(err[1:-1].max())
    # the sloping (top) wall row converges instead of diverging
    # (measured ratio 0.53; before the fix it was 1.98)
    assert tops[1] < 0.7 * tops[0]
    assert tops[0] < 1.0
    # ... and the interior keeps its 2nd order
    assert interiors[1] < 0.35 * interiors[0]


def test_mapped_advection_preserves_a_constant_tracer():
    # WHY the mapped nodal divergence is the J-weighted flux form and
    # not the (equally 2nd-order) product-rule spelling it replaced:
    # advection must use the SAME discrete divergence the projection
    # drives to zero, or a discretely non-divergent velocity injects a
    # spurious source into every constant field. tau(b == 1) is
    # -Div_adv(v), so on a projected velocity it must vanish at the CG
    # residual. The flux form is exactly MappedPressureSolver's own
    # divergence, so it does (measured 1.2e-7 at n = 16, 3.7e-7 at
    # n = 32 — the projection tolerance, and it FALLS when the solve is
    # tightened). The product-rule spelling was a different operator:
    # O(h^2) apart in the interior and O(1/h) apart in the wall row,
    # measured 13.0 (n = 16) and 30.8 (n = 32) here, i.e. growing.
    for n, bound in ((16, 1e-4), (32, 1e-4)):
        model = make_mapped_model(n, CenteredAdvection())
        set_random_state(model, seed=5)
        model.advance(steps=1)          # projects u, v, w
        model.set_fields(b=1.0)
        tau = advection_tendency(model, CenteredAdvection)
        scale = float(np.abs(np.asarray(model.state["u"].data)).max())
        assert np.abs(np.asarray(tau["b"].data)).max() < bound * scale


def test_mapped_advection_preserves_a_uniform_free_stream():
    # the discrete metric identity (free-stream preservation) the flux
    # form needs and gets: for a CONSTANT flux the coupled-axis term is
    # (q/J)[D_i(J) - D_b(Z_i)], which vanishes only if the two discrete
    # metrics are compatible. They are, because grid.metric derives a
    # parameter field's slope with the registry `diff` rows rather than
    # analytically — so a uniform u over the bump leaves a constant
    # tracer untouched to machine zero in the interior, exactly as the
    # product-rule spelling did (nothing was traded away for the wall
    # closure). The wall row is deliberately NOT machine zero: with
    # w = 0 the state drives Omega = -Z_x u != 0 through the lid, and
    # the tendency there is the divergence the projection would remove.
    for n in (16, 32):
        model = make_mapped_model(n, CenteredAdvection())
        model.set_fields(u=0.4, v=0.0, w=0.0, b=1.0)
        tau = advection_tendency(model, CenteredAdvection)
        rows = np.abs(np.asarray(tau["b"].data)).max(axis=(0, 1))
        assert rows[1:-1].max() < 1e-13
        assert rows[0] < 1e-13


def test_mapped_advection_holds_a_steady_flow_over_the_bump():
    # A0 regression, the suite gap it named: nothing integrated a
    # mapped grid forward in time with advection and a moving flow.
    # A uniform u = 0.4 over the 20% bump relaxes to the steady
    # potential flow (continuity speed-up 0.4 / 0.8 = 0.5 at the
    # crest). Before the contravariant wall closure the boundary row
    # grew without bound at every dt, resolution and slope tried
    # (measured here: |u|max 0.61 by step 20, 1.18 by step 40, 14.0
    # by step 60, all at iz = 15, the sloping lid; at n = 32,
    # dt = 0.02 it was 2.9e3 by step 25 and non-finite by step 31).
    model = make_mapped_model(16, CenteredAdvection())
    model.set_fields(u=0.4)
    model.advance(steps=60)
    u = np.abs(np.asarray(model.state["u"].data))
    assert np.isfinite(u).all()
    assert u.max() < 0.6


def test_mapped_run_grad_matches_fd():
    # the differentiability policy on the changed step-path code: the
    # cross column flux difference rides jax.grad through a short
    # mapped run (Model.propagator, the public surface) and matches a
    # central finite difference to rtol 1e-4
    model = make_mapped_model(8, CenteredAdvection())
    model.set_fields(u=0.4)
    run = model.propagator(wrt=("u",), steps=3)
    u0 = model._carry.state["u"].storage

    def loss(field):
        out = run((field,))
        return sum(jnp.sum(f.data ** 2) for f in out.state)

    grad = np.asarray(jax.grad(loss)(u0))
    assert np.all(np.isfinite(grad))
    assert np.abs(grad).max() > 0.0

    direction = jnp.asarray(
        np.random.default_rng(11).standard_normal(u0.shape),
        dtype=u0.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u0 + eps * direction))
          - float(loss(u0 - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)


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
                modules=(Core(), cls(3)),
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
                modules=(Core(), cls(order)),
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
        modules=(Core(), ConstantStratification(n2=0.0),
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


# ================================================================
#  Differentiability: the coupled-axis slope factor Z/J is VJP-sealed
# ================================================================
def test_mapped_slope_ratio_is_vjp_sealed():
    # the coupled-axis slope coefficient of the nodal mapped divergence
    # divides two metric fields; on the bounded (terrain) column the
    # denominator J = dzp/dz is zero-filled in the never-valid storage
    # padding, so the raw quotient is a masked singularity whose reverse
    # pass is 0/0 -> NaN. _safe_ratio seals it: forward is bitwise the
    # naive quotient on every valid cell, and jax.grad through it is
    # finite where the raw divide poisons the padding.
    n = 8
    model = make_mapped_model(n, CenteredAdvection(), ny=n)
    grid = model.grid
    space = model.state["b"].function_space
    slope = grid.metric(space, "dzp_dx")
    jac = grid.metric(space, "dzp_dz")

    guarded = _safe_ratio(slope, jac)
    naive = slope / jac
    valid = np.asarray(jac.storage != 0.0)
    # (1) forward bitwise unchanged on every valid cell
    assert np.array_equal(
        np.asarray(guarded.storage)[valid],
        np.asarray(naive.storage)[valid])
    # the guard fires only on the never-valid padding
    assert np.all(np.isfinite(np.asarray(guarded.storage)))
    assert np.any(~np.isfinite(np.asarray(naive.storage)))

    # (2) reverse-mode: grad of a quadratic loss through the ratio times
    # a differentiated field is finite (the padding cotangent 0 * NaN is
    # sealed to 0), where the naive divide leaves NaN in the padding
    base = jnp.asarray(
        np.random.default_rng(4).standard_normal(slope.storage.shape))

    def make_loss(ratio):
        def loss(x):
            v = slope.with_storage(x)
            return jnp.sum((ratio(slope, jac) * v).data ** 2)
        return loss

    gg = np.asarray(jax.grad(make_loss(_safe_ratio))(base))
    gn = np.asarray(jax.grad(make_loss(lambda a, b: a / b))(base))
    assert np.all(np.isfinite(gg))
    assert np.any(~np.isfinite(gn))

    # (3) forward mode stays finite (never foreclose jvp)
    tangent = jnp.asarray(
        np.random.default_rng(5).standard_normal(slope.storage.shape))
    _, jvp = jax.jvp(lambda x: _safe_ratio(slope.with_storage(x),
                                           jac).data,
                     (slope.storage,), (tangent,))
    assert np.all(np.isfinite(np.asarray(jvp)))


def test_mapped_advection_tendency_grad_matches_fd():
    # jax.grad of a quadratic loss through the nodal mapped advection
    # tendency (a terrain grid, velocities + tracer -- the u rows cross
    # the guarded Z/J slope factor) is finite and matches a central
    # finite difference; one jvp stays finite. The smallest grid that
    # carries the cross/slope term (a sloped column, coupled x-axis).
    n = 6
    model = make_mapped_model(n, CenteredAdvection(), ny=4)
    set_random_state(model, seed=3)
    flt = fr.model.term_predicates.owned_by(CenteredAdvection)
    state = model.state
    leaves, treedef = jax.tree_util.tree_flatten(state)
    u_leaf = state["u"].storage
    (idx,) = [i for i, ref in enumerate(leaves) if ref is u_leaf]

    def loss(x):
        new = list(leaves)
        new[idx] = x
        spliced = jax.tree_util.tree_unflatten(treedef, new)
        tend = model.tendency(spliced, constraints=False, filter=flt)
        return sum(jnp.sum(f.data ** 2) for f in tend)

    grad = np.asarray(jax.grad(loss)(u_leaf))
    assert np.all(np.isfinite(grad))
    assert np.abs(grad).max() > 0.0

    rng = np.random.default_rng(6)
    direction = jnp.asarray(rng.standard_normal(u_leaf.shape),
                            dtype=u_leaf.dtype)
    directional = float(jnp.vdot(jnp.asarray(grad), direction))
    eps = 1e-4
    fd = (float(loss(u_leaf + eps * direction))
          - float(loss(u_leaf - eps * direction))) / (2.0 * eps)
    assert directional == pytest.approx(fd, rel=1e-4)

    _, jvp = jax.jvp(loss, (u_leaf,), (direction,))
    assert np.isfinite(float(jvp))

