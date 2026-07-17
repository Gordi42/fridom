"""Tests for fridom.spatial.operators.cumulative.

Covers the ``CumulativeIntegral`` running-integral operator (stage H1):
the discrete fundamental theorem (the matching staggered difference
recovers the integrand), telescoping (the far-boundary value equals
``Integral``), the FV cell-average exactness, the co-located
(``center``) consistency form, the ``jacobian=`` chart weighting, the
periodic taught error, and the decomposed-axis reshard path. Also
exercises the ``("cumint", ...)`` seeded registry rows.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.operators.finite_difference import (
    FiniteDifference,
)
from fridom.spatial.operators.flux_diff import FluxDifference
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.interp import LinearInterp
from fridom.spatial.operators.verbs import cumint
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.constant import ConstantSpace


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mz():
    return IntervalMesh(16, (0.0, 2.0), periodic=False, name="z")


@pytest.fixture
def grid(mz):
    return Grid((mz,), device_ids=(0,))


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


def _poly(z):
    # a quadratic: exact antiderivative for the exactness checks
    return 3.0 * z**2 - 2.0 * z + 1.0


def _poly_int(z):
    # antiderivative of _poly with P(0) = 0
    return z**3 - z**2 + z


# ================================================================
#  Static surface, construction, interning
# ================================================================
def test_dispatch_kind():
    assert CumulativeIntegral().dispatch_kind == "cumint"


def test_defaults_are_up_face_plain():
    op = CumulativeIntegral()
    assert op.direction == "up"
    assert op.target == "face"
    assert op.jacobian is None


def test_requirements_are_axis_local(mz):
    req = CumulativeIntegral().requirements(mz.center)
    assert req.halo == 0
    assert req.layout == "local"


def test_constructor_validates_direction():
    with pytest.raises(ValueError, match="direction"):
        CumulativeIntegral(direction="sideways")


def test_constructor_validates_target():
    with pytest.raises(ValueError, match="target"):
        CumulativeIntegral(target="edge")


def test_constructor_validates_jacobian():
    with pytest.raises(TypeError, match="jacobian"):
        CumulativeIntegral(jacobian=())
    with pytest.raises(TypeError, match="jacobian"):
        CumulativeIntegral(jacobian=(1, 2))


def test_families_intern_separately():
    assert CumulativeIntegral() is CumulativeIntegral("up", "face")
    assert CumulativeIntegral("down") is CumulativeIntegral("down")
    assert CumulativeIntegral("up") is not CumulativeIntegral("down")
    assert CumulativeIntegral(target="center") is not (
        CumulativeIntegral(target="face"))
    assert CumulativeIntegral(jacobian=("u", "v")) is (
        CumulativeIntegral(jacobian=("u", "v")))
    assert CumulativeIntegral(jacobian=("u",)) is not (
        CumulativeIntegral())


# ================================================================
#  Codomain
# ================================================================
def test_codomain_face(mz):
    op = CumulativeIntegral(target="face")
    assert op.codomain(mz.center) is mz.outer
    assert op.codomain(mz.cell_avg) is mz.outer
    assert op.codomain(mz.center.as_complex()) is (
        mz.outer.as_complex())


def test_codomain_center_is_colocated(mz):
    op = CumulativeIntegral(target="center")
    assert op.codomain(mz.center) is mz.center
    assert op.codomain(mz.cell_avg) is mz.cell_avg


def test_codomain_identity_on_constant(mz):
    assert CumulativeIntegral().codomain(mz.constant) is mz.constant


def test_codomain_rejects_coefficient(mx):
    fourier = mx.fourier(origin=mx.center)
    with pytest.raises(SpaceMismatchError, match="transform"):
        CumulativeIntegral().codomain(fourier)


def test_codomain_rejects_face_families(mz):
    with pytest.raises(SpaceMismatchError, match="center-valued"):
        CumulativeIntegral().codomain(mz.outer)
    with pytest.raises(SpaceMismatchError, match="center-valued"):
        CumulativeIntegral().codomain(mz.face_avg)


def test_codomain_rejects_periodic_axis_naming_it(mx):
    with pytest.raises(SpaceMismatchError, match=r"cumint along 'x'"):
        CumulativeIntegral().codomain(mx.center)


# ================================================================
#  Nodal fundamental theorem + telescoping (both directions)
# ================================================================
def test_up_face_lands_on_the_both_boundary_faces(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    assert w.function_space.bare is mz.outer
    assert w.shape == (17,)
    # seeded zero at the bottom face
    assert jnp.allclose(w.data[0], 0.0)


def test_ftc_up_recovers_integrand(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    back = FiniteDifference()["z"](w)          # Outer -> Center
    assert back.function_space.bare is mz.center
    assert jnp.allclose(back.data, f.data, atol=1e-12)


def test_ftc_down_recovers_negated_integrand(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    w = CumulativeIntegral(direction="down", target="face")["z"](f)
    assert jnp.allclose(w.data[-1], 0.0)       # seeded zero at the top
    back = FiniteDifference()["z"](w)
    # the derivative of an integral taken from the upper limit is -f
    assert jnp.allclose(back.data, -f.data, atol=1e-12)


def test_telescopes_to_integral_up(grid, mz):
    # the accumulated increments are exactly the increments Integral
    # sums, so telescoping is exact even where the nodal midpoint rule
    # is not the analytic integral (both share the same quadrature)
    f = grid.create_field(mz.center, init=_poly)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    total = f.integrate("z").data.reshape(())
    assert jnp.allclose(w.data[-1], total, atol=1e-13)


def test_telescopes_to_integral_down(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    w = CumulativeIntegral(direction="down", target="face")["z"](f)
    total = f.integrate("z").data.reshape(())
    assert jnp.allclose(w.data[0], total, atol=1e-13)


def test_running_integral_is_exact_for_a_linear_integrand(grid, mz):
    # the nodal midpoint rule is exact on linears, so the whole face
    # profile matches the analytic antiderivative (not just the far
    # value); a quadratic keeps only the exact FTC/telescoping above
    g = grid.create_field(mz.center, init=lambda z: 2.0 * z + 0.5)
    wg = CumulativeIntegral(direction="up", target="face")["z"](g)
    faces = jnp.linspace(0.0, 2.0, 17)
    exact = faces**2 + 0.5 * faces          # antiderivative of 2z+0.5
    assert jnp.allclose(wg.data, exact, atol=1e-12)


# ================================================================
#  Multi-axis: the other factors ride along
# ================================================================
def test_partial_cumint_keeps_the_other_factors():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    mz = IntervalMesh(8, (0.0, 2.0), periodic=False, name="z")
    grid = Grid((mx, mz), device_ids=(0,))
    f = grid.create_field(mx.center * mz.center,
                          init=lambda x, z: z + 0.0 * x)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    factors = w.function_space.bare.factors
    assert factors[0] is mx.center
    assert factors[1] is mz.outer
    assert w.shape == (8, 9)


# ================================================================
#  FV family (cell-average cumulative sum is exact)
# ================================================================
def _cell_averages(mesh, antideriv):
    edges = jnp.linspace(0.0, 2.0, mesh.n_cells + 1)
    return jnp.diff(antideriv(edges)) / mesh.dx


def test_fv_face_ftc_via_flux_difference(grid, mz):
    avg = _cell_averages(mz, _poly_int)          # exact averages of _poly
    f = grid.create_field(mz.cell_avg, data=avg)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    assert w.function_space.bare is mz.outer
    back = FluxDifference()["z"](w)              # Outer -> CellAvg (exact)
    assert back.function_space.bare is mz.cell_avg
    assert jnp.allclose(back.data, f.data, atol=1e-12)


def test_fv_face_is_the_exact_running_integral(grid, mz):
    avg = _cell_averages(mz, _poly_int)
    f = grid.create_field(mz.cell_avg, data=avg)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    faces = jnp.linspace(0.0, 2.0, mz.n_cells + 1)
    # the cell-average cumulative sum is EXACT: face j == P(z_j) - P(0)
    assert jnp.allclose(w.data, _poly_int(faces), atol=1e-12)


def test_fv_telescopes_to_integral(grid, mz):
    avg = _cell_averages(mz, _poly_int)
    f = grid.create_field(mz.cell_avg, data=avg)
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    assert jnp.allclose(w.data[-1], f.integrate("z").data.reshape(()),
                        atol=1e-13)


# ================================================================
#  The co-located (center) form: the pyOM half-cell midpoint
# ================================================================
def test_center_form_is_colocated(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    p = CumulativeIntegral(direction="down", target="center")["z"](f)
    assert p.function_space.bare is mz.center
    assert p.shape == (16,)


def test_center_form_is_the_midpoint_of_the_face_form(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    face = CumulativeIntegral(direction="up", target="face")["z"](f)
    center = CumulativeIntegral(direction="up", target="center")["z"](f)
    midpoint = 0.5 * (face.data[:-1] + face.data[1:])
    assert jnp.allclose(center.data, midpoint, atol=1e-14)


def test_center_form_balance_diff_equals_interp(grid, mz):
    # the discrete hydrostatic balance: FiniteDifference(center) ==
    # LinearInterp(integrand) on the inner faces (down => negated)
    f = grid.create_field(mz.center, init=_poly)
    p = CumulativeIntegral(direction="down", target="center")["z"](f)
    dp = FiniteDifference()["z"](p)             # Center -> Inner
    bi = LinearInterp()["z"](f)                 # Center -> Inner
    assert dp.function_space.bare is mz.inner
    assert jnp.allclose(dp.data, -bi.data, atol=1e-12)


def test_fv_center_form_is_the_exact_cell_average(grid, mz):
    # the running integral of a piecewise-constant (cell-avg) integrand
    # is piecewise-linear, so its cell average IS the face midpoint
    avg = _cell_averages(mz, _poly_int)
    f = grid.create_field(mz.cell_avg, data=avg)
    center = CumulativeIntegral(target="center")["z"](f)
    assert center.function_space.bare is mz.cell_avg
    faces = jnp.linspace(0.0, 2.0, mz.n_cells + 1)
    exact = 0.5 * (_poly_int(faces[:-1]) + _poly_int(faces[1:]))
    assert jnp.allclose(center.data, exact, atol=1e-12)


# ================================================================
#  Complex integrands
# ================================================================
def test_complex_integrand_accumulates_both_parts(grid, mz):
    real = grid.create_field(mz.center, init=_poly)
    imag = grid.create_field(mz.center, init=lambda z: z)
    f = real + 1j * imag
    assert f.function_space.bare.factors[0].scalars is Scalars.COMPLEX
    w = CumulativeIntegral(direction="up", target="face")["z"](f)
    wr = CumulativeIntegral(direction="up", target="face")["z"](real)
    wi = CumulativeIntegral(direction="up", target="face")["z"](imag)
    assert jnp.allclose(w.data.real, wr.data, atol=1e-13)
    assert jnp.allclose(w.data.imag, wi.data, atol=1e-13)


# ================================================================
#  Mapped / stretched meshes (measure-handled; jacobian=None)
# ================================================================
def _tanh_map(s):
    return jnp.tanh(2.0 * s) / jnp.tanh(2.0)


def test_ftc_on_a_mapped_mesh():
    mesh = MappedIntervalMesh(24, (0.0, 1.0), _tanh_map, name="v")
    grid = Grid((mesh,), device_ids=(0,))
    f = grid.create_field(mesh.center, init=lambda v: 2.0 * v + 0.5)
    w = CumulativeIntegral(direction="up", target="face")["v"](f)
    back = FiniteDifference()["v"](w)
    assert jnp.allclose(back.data, f.data, atol=1e-11)


def test_telescopes_on_a_mapped_mesh():
    mesh = MappedIntervalMesh(24, (0.0, 1.0), _tanh_map, name="v")
    grid = Grid((mesh,), device_ids=(0,))
    f = grid.create_field(mesh.center, init=lambda v: 2.0 * v + 0.5)
    w = CumulativeIntegral(direction="up", target="face")["v"](f)
    assert jnp.allclose(w.data[-1],
                        f.integrate("v").data.reshape(()), atol=1e-12)


# ================================================================
#  The sqrt_g Jacobian rows on chart grids (mirrors Integral)
# ================================================================
def _torus_grid(n=8, ring=2.0, minor=0.5):
    mu = IntervalMesh(n, (0.0, 2.0 * jnp.pi), name="u")
    mv = IntervalMesh(n, (0.0, 2.0 * jnp.pi), periodic=False, name="v")
    mapping = CoordinateMapping(chart={"X": lambda u, v: (
        (ring + minor * jnp.cos(v)) * jnp.cos(u),
        (ring + minor * jnp.cos(v)) * jnp.sin(u),
        minor * jnp.sin(v))})
    return Grid((mu, mv), mapping=mapping, device_ids=(0,)), mu, mv


def test_chart_row_carries_the_jacobian():
    grid, _mu, mv = _torus_grid()
    row = grid.dispatch.resolve("cumint", mv.center)
    assert row.jacobian == ("u", "v")


def test_chart_running_integral_telescopes_to_the_weighted_integral():
    grid, mu, mv = _torus_grid()
    row = grid.dispatch.resolve("cumint", mv.center)
    f = grid.create_field(mu.center * mv.center,
                          init=lambda u, v: jnp.cos(v) + 0.0 * u)
    w = row["v"](f)
    # the far-boundary value == the sqrt_g-weighted Integral("v")
    telescope = f.integrate("v")
    assert jnp.allclose(w.data[:, -1], telescope.data.squeeze(),
                        atol=1e-12)


def test_chart_running_integral_ftc_carries_sqrt_g():
    # differencing the physical running integral returns f * sqrt_g
    grid, mu, mv = _torus_grid()
    row = grid.dispatch.resolve("cumint", mv.center)
    f = grid.create_field(mu.center * mv.center,
                          init=lambda u, v: jnp.cos(v) + 0.0 * u)
    w = row["v"](f)
    back = FiniteDifference()["v"](w)
    sqrt_g = grid.metric(mu.center * mv.center, "sqrt_g")
    assert jnp.allclose(back.data, f.data * sqrt_g.data, atol=1e-11)


def test_born_constant_chart_factor_gets_no_jacobian():
    # a field constant along u cannot resolve sqrt_g: plain measure
    grid, mu, mv = _torus_grid()
    row = grid.dispatch.resolve("cumint", mv.center)
    f = grid.create_field(mu.constant * mv.center,
                          init=lambda v: 1.0 + 0.0 * v)
    w = row["v"](f)
    # plain computational running integral: far value == length in v
    assert jnp.allclose(w.data.squeeze()[-1], 2.0 * jnp.pi, atol=1e-12)


def test_lone_factor_off_the_chart_uses_the_plain_measure():
    # a lone-factor field cannot resolve the 2D sqrt_g (KeyError path)
    grid, _mu, mv = _torus_grid()
    row = grid.dispatch.resolve("cumint", mv.center)
    f = grid.create_field(mv.center, init=lambda v: 1.0 + 0.0 * v)
    w = row["v"](f)
    assert jnp.allclose(w.data[-1], 2.0 * jnp.pi, atol=1e-12)


# ================================================================
#  Analytic maps= terrain columns: the jacobian= column Jacobian
# ================================================================
def _depth(x):
    return 1.0 + 0.2 * jnp.sin(x)


def _stretch(s):
    return s + 0.15 * jnp.sin(2.0 * np.pi * s) / (2.0 * np.pi)


def _terrain_grid(n, stretched):
    # z_p = sigma * H(x): the mapped physical coordinate "zp" is not a
    # mesh axis; its base column "sigma" may itself be stretched
    mx = IntervalMesh(n, (0.0, 2.0 * np.pi), periodic=True, name="x")
    if stretched:
        ms = MappedIntervalMesh(n, (0.0, 1.0), _stretch,
                                periodic=False, name="sigma")
    else:
        ms = IntervalMesh(n, (0.0, 1.0), periodic=False,
                          name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": _depth})
    return Grid((mx, ms), mapping=mapping, device_ids=(0,)), mx, ms


@pytest.mark.parametrize("stretched", [
    pytest.param(False, id="uniform-sigma"),
    pytest.param(True, id="stretched-sigma"),
])
def test_maps_jacobian_running_integral_converges(stretched):
    # p_hyd(z) = -int_z^H cos(z') dz' = sin(z) - sin(H), with the
    # increment weighted by the column Jacobian d<zp>_d<sigma> == H;
    # the down/center running integral converges at 2nd order
    errors = []
    for n in (16, 32, 64):
        grid, mx, ms = _terrain_grid(n, stretched)
        space = mx.center * ms.center
        b = grid.create_field(
            space, init=lambda x, sigma: jnp.cos(sigma * _depth(x)))
        op = CumulativeIntegral(direction="down", target="center",
                                jacobian=("zp",))["sigma"]
        p = -np.asarray(op(b).data)
        x = np.asarray(grid.evaluation_nodes(space, "x").data)
        s = np.asarray(grid.evaluation_nodes(space, "sigma").data)
        depth = 1.0 + 0.2 * np.sin(x)
        exact = np.sin(s * depth) - np.sin(depth)
        errors.append(float(np.max(np.abs(p - exact))))
    orders = np.log2(np.asarray(errors[:-1])
                     / np.asarray(errors[1:]))
    assert bool(np.all(orders > 1.7))


def test_maps_jacobian_telescopes_to_the_weighted_integral():
    # the far-boundary value of the running integral == the
    # jacobian-weighted Integral of the same field (physical column)
    grid, mx, ms = _terrain_grid(24, stretched=True)
    space = mx.center * ms.center
    b = grid.create_field(space,
                          init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    run = CumulativeIntegral(direction="up", target="face",
                             jacobian=("zp",))["sigma"](b)
    total = Integral(jacobian=("zp",))["sigma"](b)
    assert jnp.allclose(run.data[:, -1], total.data.squeeze(),
                        atol=1e-12)


def test_maps_jacobian_is_no_longer_a_silent_noop():
    # the historical trap: jacobian=("zp",) once was bitwise equal to
    # the unweighted running integral (a silent no-op)
    grid, mx, ms = _terrain_grid(16, stretched=False)
    space = mx.center * ms.center
    b = grid.create_field(space,
                          init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    weighted = CumulativeIntegral(
        direction="down", jacobian=("zp",))["sigma"](b)
    plain = CumulativeIntegral(direction="down")["sigma"](b)
    assert not jnp.allclose(weighted.data, plain.data)


def test_maps_jacobian_bogus_name_raises():
    grid, mx, ms = _terrain_grid(8, stretched=False)
    space = mx.center * ms.center
    b = grid.create_field(space,
                          init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    with pytest.raises(ValueError, match="not a chart coordinate"):
        CumulativeIntegral(jacobian=("bogus",))["sigma"](b)


def test_maps_jacobian_base_axis_name_raises():
    # "sigma" is the base axis, not the chart coordinate "zp"
    grid, mx, ms = _terrain_grid(8, stretched=False)
    space = mx.center * ms.center
    b = grid.create_field(space,
                          init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    with pytest.raises(ValueError, match="not a chart coordinate"):
        CumulativeIntegral(jacobian=("sigma",))["sigma"](b)


def test_maps_jacobian_running_integral_is_differentiable():
    # the down/center running integral is the hydrostatic p_hyd path:
    # grad of a quadratic loss through it is finite and matches a
    # central finite difference (differentiability policy)
    grid, mx, ms = _terrain_grid(8, stretched=False)
    space = mx.center * ms.center
    op = CumulativeIntegral(direction="down", target="center",
                            jacobian=("zp",))["sigma"]

    def loss(data):
        b = grid.create_field(space, data=data)
        return (op(b).data ** 2).sum()

    rng = np.random.default_rng(0)
    x0 = jnp.asarray(rng.standard_normal(space.shape))
    direction = jnp.asarray(rng.standard_normal(space.shape))
    ad = float(jnp.vdot(jax.grad(loss)(x0), direction))
    eps = 1e-4
    fd = float((loss(x0 + eps * direction)
                - loss(x0 - eps * direction)) / (2.0 * eps))
    assert np.isfinite(ad)
    assert np.isclose(ad, fd, rtol=1e-4)


# ================================================================
#  Seeded registry rows / dispatch
# ================================================================
def test_seeded_rows_are_the_default_up_face(grid, mz):
    assert grid.dispatch.resolve("cumint", mz.center) is (
        CumulativeIntegral())
    assert grid.dispatch.resolve("cumint", mz.cell_avg) is (
        CumulativeIntegral())
    assert grid.dispatch.resolve(
        "cumint", mz.center.as_complex()) is CumulativeIntegral()


def test_chartless_row_has_no_jacobian(grid, mz):
    row = grid.dispatch.resolve("cumint", mz.center)
    assert row.jacobian is None


def test_verb_resolves_and_applies(grid, mz):
    f = grid.create_field(mz.center, init=_poly)
    w = cumint["z"](f)
    assert w.function_space.bare is mz.outer
    assert jnp.allclose(w.data[0], 0.0)


# ================================================================
#  Decomposition: the integration axis is resharded local and back
# ================================================================
@pytest.mark.multi_device
def test_decomposed_axis_matches_single_device():
    def build(ids):
        meshes = (
            IntervalMesh(16, (0.0, 1.0), periodic=True, name="x"),
            IntervalMesh(16, (0.0, 2.0), periodic=True, name="y"),
            IntervalMesh(16, (0.0, 3.0), periodic=False, name="z"))
        return Grid(meshes, device_ids=ids)

    many = build(tuple(range(jax.device_count())))
    one = build((0,))
    data = jnp.asarray(
        np.random.default_rng(0).standard_normal((16, 16, 16)))
    space_many = (many.factors[0].center * many.factors[1].center
                  * many.factors[2].center)
    space_one = (one.factors[0].center * one.factors[1].center
                 * one.factors[2].center)
    f_many = many.create_field(space_many, data=data)
    f_one = one.create_field(space_one, data=data)

    # force the integration axis onto a device axis
    z_sharded = many.decomposition.layout_for(("x", "y"))
    assert not z_sharded.is_local("z")
    f_many = f_many.reshard(z_sharded)

    op = CumulativeIntegral(direction="down", target="face")
    w_many = op["z"](f_many)
    w_one = op["z"](f_one)
    # the result is handed back in the operand's layout
    assert w_many.function_space.layout == z_sharded
    assert np.allclose(np.asarray(w_many.data), np.asarray(w_one.data),
                       atol=1e-12)
    # the discrete FTC survives the reshard round-trip
    back = FiniteDifference()["z"](w_many)
    assert np.allclose(np.asarray(back.data),
                       -np.asarray(f_many.data), atol=1e-11)


@pytest.mark.multi_device
def test_center_target_is_colocated_under_decomposition():
    def build(ids):
        meshes = (
            IntervalMesh(16, (0.0, 1.0), periodic=True, name="x"),
            IntervalMesh(16, (0.0, 3.0), periodic=False, name="z"))
        return Grid(meshes, device_ids=ids)

    many = build(tuple(range(jax.device_count())))
    one = build((0,))
    data = jnp.asarray(
        np.random.default_rng(1).standard_normal((16, 16)))
    f_many = many.create_field(
        many.factors[0].center * many.factors[1].center, data=data)
    f_one = one.create_field(
        one.factors[0].center * one.factors[1].center, data=data)
    f_many = f_many.reshard(many.decomposition.layout_for(("x",)))

    op = CumulativeIntegral(direction="down", target="center")
    p_many = op["z"](f_many)
    p_one = op["z"](f_one)
    assert p_many.function_space.bare is (
        many.factors[0].center * many.factors[1].center)
    assert np.allclose(np.asarray(p_many.data),
                       np.asarray(p_one.data), atol=1e-12)


def test_binding_a_constant_axis_is_the_identity():
    mx = IntervalMesh(8, (0.0, 1.0), name="x")
    mz = IntervalMesh(8, (0.0, 2.0), periodic=False, name="z")
    grid = Grid((mx, mz), device_ids=(0,))
    g = grid.create_field(mx.constant * mz.center, init=lambda z: z)
    # the base short-circuits a ConstantSpace axis (rules 3.3)
    assert CumulativeIntegral()["x"](g) is g
    assert isinstance(g.function_space.bare.factors[0], ConstantSpace)
