"""Tests for fridom.spatial.operators.integrate.

Also covers the two seams this cluster wires for it: the
``grid.measure`` accessor and the ``f.integrate`` / ``f.mean``
forwarders on ``ScalarField``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.spatial.bc import BC
from fridom.spatial.coordinate_mapping import (
    CoordinateMapping,
)
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import (
    MappedIntervalMesh,
)
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodeSet


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def integral():
    return Integral()


# ================================================================
#  Static surface and codomain
# ================================================================
def test_dispatch_kind(integral):
    assert integral.dispatch_kind == "integrate"


def test_requirements(integral, mx):
    req = integral.requirements(mx.center)
    assert req.halo == 0
    assert req.layout == "any"
    assert req.collective is True


def test_codomain(integral, mx):
    assert integral.codomain(mx.center) is mx.constant
    assert integral.codomain(mx.cell_avg) is mx.constant
    assert integral.codomain(mx.constant) is mx.constant
    assert integral.codomain(mx.center.as_complex()) is (
        mx.constant.as_complex())


def test_codomain_rejects_coefficient_factors(integral, mx):
    fourier = mx.fourier(origin=mx.center)
    with pytest.raises(SpaceMismatchError, match="transform back"):
        integral.codomain(fourier)


def test_codomain_rejects_non_factor_domains(integral, mx, my):
    with pytest.raises(SpaceMismatchError,
                       match="nodal and average"):
        integral.codomain(mx.center * my.center)


# ================================================================
#  grid.measure (the quadrature-weight seam)
# ================================================================
def test_measure_is_the_uniform_cell_width(mx):
    grid = Grid((mx,))
    w = grid.measure(mx.center)
    assert w.function_space.bare is mx.center
    assert jnp.allclose(w.data, jnp.full(8, mx.dx))
    assert grid.measure(mx.cell_avg).data.shape == (8,)
    assert jnp.allclose(grid.measure(mx.right).data, mx.dx)


def test_measure_halves_the_boundary_dual_cells(my):
    grid = Grid((my,))
    w = grid.measure(my.outer)
    dy = my.dx
    expected = jnp.full(9, dy).at[0].set(dy / 2).at[-1].set(dy / 2)
    assert jnp.allclose(w.data, expected)
    # the trapezoid weights sum to the domain length
    assert jnp.allclose(w.data.sum(), 2.0)
    # face-family interior measures stay full dual cells
    assert jnp.allclose(grid.measure(my.inner).data, dy)
    assert grid.measure(my.face_avg).data.shape == (7,)


def test_measure_tags_the_querying_space(mx, my):
    grid = Grid((mx, my))
    w = grid.measure(mx.center * my.center, name="x")
    factors = w.function_space.bare.factors
    assert factors[0] is mx.center
    assert isinstance(factors[1], ConstantSpace)
    assert w.name == "dx"
    # broadcasts against the querying space
    f = grid.create_field(mx.center * my.center)
    assert (f * w).function_space.bare is mx.center * my.center


def test_measure_name_resolution_errors(mx, my):
    grid = Grid((mx, my))
    with pytest.raises(ValueError, match="ambiguous"):
        grid.measure(mx.center * my.center)
    with pytest.raises(ValueError, match="constant"):
        grid.measure(mx.constant * my.center, name="x")
    with pytest.raises(ValueError, match="coefficient"):
        grid.measure(mx.fourier(origin=mx.center) * my.center,
                     name="x")


def test_measure_drops_bc_constrained_boundary_dofs(my):
    grid = Grid((my,))
    space = my.nodal(NodeSet.OUTER, bc=BC.DIRICHLET)
    w = grid.measure(space)
    # both boundary DOFs are dropped, like the space shape drops them
    assert w.data.shape == space.shape
    assert jnp.allclose(w.data, my.dx)


def test_measure_on_chebyshev_awaits_clenshaw_curtis():
    cheb = ChebyshevMesh(8, (0.0, 1.0), name="s")
    grid = Grid((cheb,))
    with pytest.raises(NotImplementedError, match="Clenshaw"):
        grid.measure(cheb.outer)


# ================================================================
#  Integral application
# ================================================================
def test_integral_of_sin_vanishes_on_a_periodic_axis(mx):
    grid = Grid((mx,))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    total = f.integrate("x")
    assert total.function_space.bare is mx.constant
    assert jnp.allclose(total.data[0], 0.0, atol=1e-12)


def test_integral_is_exact_on_cell_averages(my):
    # exact cell averages of y^2 over [0, 2]: sum(avg * dy) = 8/3
    grid = Grid((my,))
    edges = jnp.linspace(0.0, 2.0, 9)
    averages = (edges[1:] ** 3 - edges[:-1] ** 3) / (3 * my.dx)
    f = grid.create_field(my.cell_avg, data=averages)
    assert jnp.allclose(f.integrate("y").data[0], 8.0 / 3.0)


def test_integral_is_the_trapezoid_rule_on_outer(my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y)
    # trapezoid is exact on linears: int_0^2 3y dy = 6
    assert jnp.allclose(f.integrate("y").data[0], 6.0)


def test_integral_result_has_default_metadata(mx):
    grid = Grid((mx,))
    f = grid.create_field(name="q", units="kg")
    assert f.integrate("x").name == "unnamed"  # new quantity


# ================================================================
#  f.integrate forwarder
# ================================================================
def test_partial_reduction_keeps_the_other_factors(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x) + y)
    fx = f.integrate("x")
    factors = fx.function_space.bare.factors
    assert isinstance(factors[0], ConstantSpace)
    assert factors[1] is my.center
    assert fx.shape == (1, 8)


def test_no_names_reduces_every_factor(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(init=lambda x, y: 1.0 + 0.0 * x * y)
    total = f.integrate()
    assert all(isinstance(factor, ConstantSpace)
               for factor in total.function_space.bare.factors)
    assert jnp.allclose(total.data, 1.0 * 2.0)  # the domain volume


def test_reduction_along_a_constant_factor_is_identity(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(mx.center * my.constant)
    assert f.integrate("y") is f


def test_integrate_broadcasts_back(mx):
    # f - f.integrate("x") stays in the strict algebra (rules 3.3)
    grid = Grid((mx,))
    f = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x) + 5.0)
    residual = f - f.integrate("x")
    assert residual.function_space.bare is mx.center
    assert jnp.allclose(residual.integrate("x").data[0], 0.0,
                        atol=1e-12)


def test_integrate_validates_and_dedupes_names(mx):
    grid = Grid((mx,))
    f = grid.create_field()
    with pytest.raises(ValueError, match="no factors named"):
        f.integrate("z")
    once = f.integrate("x")
    twice = f.integrate("x", "x")
    assert jnp.allclose(once.data, twice.data)


def test_integrate_coefficient_factor_says_transform_back(mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.fourier(origin=mx.center))
    with pytest.raises(DispatchError, match="transform back"):
        f.integrate("x")


# ================================================================
#  f.mean forwarder
# ================================================================
def test_mean_of_a_constant_is_the_constant(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(init=lambda x, y: 3.0 + 0.0 * x * y)
    assert jnp.allclose(f.mean().data, 3.0)


def test_mean_removes_the_oscillation(mx):
    grid = Grid((mx,))
    f = grid.create_field(
        init=lambda x: jnp.sin(2 * jnp.pi * x) + 4.0)
    assert jnp.allclose(f.mean("x").data[0], 4.0, atol=1e-12)


def test_mean_uses_the_trapezoid_measure_on_outer(my):
    grid = Grid((my,))
    f = grid.create_field(my.outer, init=lambda y: 3.0 * y)
    # mean of 3y over [0, 2] is 3 (trapezoid exact on linears)
    assert jnp.allclose(f.mean("y").data[0], 3.0)


def test_mean_on_an_all_constant_space_is_identity(mx):
    grid = Grid((mx,))
    f = grid.create_field(mx.constant)
    assert f.mean() is f


# ================================================================
#  Mapped meshes: stretched quadrature weights (stage C0)
# ================================================================
def _tanh_map(s):
    return jnp.tanh(2.0 * s) / jnp.tanh(2.0)


def test_integral_is_exact_on_mapped_cell_averages():
    mesh = MappedIntervalMesh(8, (0.0, 1.0), _tanh_map, name="v")
    grid = Grid((mesh,))
    # true cell averages of f(v) = v: antiderivative differences
    # over the stretched primal cells, divided by the cell widths
    faces = _tanh_map(jnp.arange(9) / 8)
    w = grid.measure(mesh.cell_avg, name="v")
    averages = jnp.diff(faces**2 / 2.0) / w.data
    f = grid.create_field(mesh.cell_avg, data=averages)
    assert jnp.allclose(f.integrate("v").data[0], 0.5)


def test_stretched_outer_weights_tile_the_domain():
    mesh = MappedIntervalMesh(8, (0.0, 1.0), _tanh_map, name="v")
    grid = Grid((mesh,))
    # the clipped dual measures tile [0, 1] exactly (telescoping),
    # so constants integrate exactly; unlike the uniform trapezoid
    # the stretched nodes are not dual-cell midpoints, so linears
    # are only 2nd-order convergent (covered below)
    f = grid.create_field(mesh.outer, init=lambda v: 3.0 + 0.0 * v)
    assert jnp.allclose(f.integrate("v").data[0], 3.0)


def test_nodal_integral_converges_on_a_mapped_mesh():
    errors = []
    for n in (16, 32):
        mesh = MappedIntervalMesh(n, (0.0, 1.0), _tanh_map,
                                  name="v")
        grid = Grid((mesh,))
        f = grid.create_field(mesh.center,
                              init=lambda v: jnp.sin(jnp.pi * v))
        errors.append(abs(float(f.integrate("v").data[0])
                          - 2.0 / jnp.pi))
    assert errors[0] / errors[1] > 3.0


# ================================================================
#  The sqrt_g Jacobian rows on chart grids (stage C2)
# ================================================================
def torus_grid(n=8, ring=2.0, minor=0.5):
    mu = IntervalMesh(n, (0.0, 2.0 * jnp.pi), name="u")
    mv = IntervalMesh(n, (0.0, 2.0 * jnp.pi), name="v")
    mapping = CoordinateMapping(chart={"X": lambda u, v: (
        (ring + minor * jnp.cos(v)) * jnp.cos(u),
        (ring + minor * jnp.cos(v)) * jnp.sin(u),
        minor * jnp.sin(v))})
    return Grid((mu, mv), mapping=mapping), mu, mv


def test_jacobian_constructor_validates():
    with pytest.raises(TypeError, match="jacobian"):
        Integral(jacobian=())
    with pytest.raises(TypeError, match="jacobian"):
        Integral(jacobian=(1, 2))


def test_jacobian_families_intern_separately():
    assert Integral() is Integral(jacobian=None)
    assert Integral(jacobian=("u", "v")) is Integral(
        jacobian=("u", "v"))
    assert Integral(jacobian=("u", "v")) is not Integral()


def test_area_integral_carries_sqrt_g_once():
    # the torus area is 4 pi^2 R r, exact for the constant field:
    # sqrt_g enters on the first chart reduction only
    ring, minor = 2.0, 0.5
    grid, mu, mv = torus_grid(ring=ring, minor=minor)
    one = grid.create_field(
        mu.center * mv.center, init=lambda u, v: 1.0 + 0 * u + 0 * v)
    area = one.integrate()
    exact = 4.0 * jnp.pi**2 * ring * minor
    assert jnp.allclose(area.data.squeeze(), exact)


def test_partial_chart_reduction_is_the_weighted_density():
    # integrate("u") of f == int f sqrt_g du: a v-dependent density
    ring, minor = 2.0, 0.5
    grid, mu, mv = torus_grid(ring=ring, minor=minor)
    f = grid.create_field(
        mu.center * mv.center,
        init=lambda u, v: jnp.cos(v) + 0 * u)
    density = f.integrate("u")
    v = grid.evaluation_nodes(density.function_space, "v").data
    exact = (2.0 * jnp.pi * jnp.cos(v)
             * minor * (ring + minor * jnp.cos(v)))
    assert jnp.allclose(density.data, exact)


def test_born_constant_chart_factor_gets_no_jacobian():
    # a field constant along u cannot resolve sqrt_g: it contracts
    # against the computational measure only (module docstring)
    grid, mu, mv = torus_grid()
    f = grid.create_field(
        mu.constant * mv.center, init=lambda v: 1.0 + 0 * v)
    total = f.integrate()
    assert jnp.allclose(total.data.squeeze(), 2.0 * jnp.pi)


def test_chartless_grids_keep_the_plain_integral(mx):
    grid = Grid((mx,))
    row = grid.dispatch.resolve("integrate", mx.center)
    assert row is Integral()
    assert row.jacobian is None


def test_fields_off_the_chart_meshes_use_the_plain_measure():
    # a lone-factor field cannot resolve the 2D sqrt_g: the chart
    # row falls back to the computational measure
    grid, mu, _mv = torus_grid()
    f = grid.create_field(mu.center, init=lambda u: 1.0 + 0 * u)
    assert jnp.allclose(f.integrate("u").data.squeeze(),
                        2.0 * jnp.pi)


# ================================================================
#  Analytic maps= terrain columns: the jacobian= column Jacobian
# ================================================================
def _depth(x):
    return 1.0 + 0.2 * jnp.sin(x)


def _terrain_grid(n):
    # z_p = sigma * H(x): a single-base terrain-following column whose
    # mapped physical coordinate "zp" is not a mesh axis
    mx = IntervalMesh(n, (0.0, 2.0 * jnp.pi), name="x")
    ms = IntervalMesh(n, (0.0, 1.0), periodic=False, name="sigma")
    mapping = CoordinateMapping(
        maps={"zp": lambda sigma, H: sigma * H},
        params={"H": _depth})
    return Grid((mx, ms), mapping=mapping), mx, ms


def test_maps_jacobian_matches_the_module_side_weighting():
    # Integral(jacobian=("zp",)) is exactly cumint of the column
    # Jacobian d<zp>_d<sigma> onto the increment (route b == route a)
    grid, mx, ms = _terrain_grid(16)
    space = mx.center * ms.center
    u = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    gate = Integral(jacobian=("zp",))["sigma"](u)
    # the module-side reference hand-weights, then reduces with the RAW
    # computational Integral() (the seeded verb now Jacobian-weights on
    # this terrain grid, so `.integrate` would double-count)
    manual = Integral()["sigma"](
        u * grid.metric(space.bare, "dzp_dsigma"))
    # route b (the wired seam) computes the same weighted integral as
    # route a (module-side weighting); the two differ only by
    # floating-point reassociation (weight*metric multiply order and,
    # sharded, the reduction), so allclose not bitwise
    assert jnp.allclose(gate.data, manual.data, atol=1e-14)


def test_maps_jacobian_weighted_column_converges_to_the_physical():
    # int_0^H (sigma H)^2 dz_p = H^3 / 3; the weighted reduction
    # converges to it at 2nd order (chart-coordinate spelling "zp")
    errors = []
    for n in (16, 32, 64):
        grid, mx, ms = _terrain_grid(n)
        space = mx.center * ms.center
        u = grid.create_field(
            space, init=lambda x, sigma: (sigma * _depth(x)) ** 2)
        column = Integral(jacobian=("zp",))["sigma"](u)
        x = np.asarray(
            grid.evaluation_nodes(column.function_space, "x").data)
        exact = (1.0 + 0.2 * np.sin(x.ravel())) ** 3 / 3.0
        errors.append(float(np.max(np.abs(
            np.asarray(column.data).ravel() - exact))))
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert bool(np.all(orders > 1.7))


def test_maps_jacobian_is_no_longer_a_silent_noop():
    # the historical trap: jacobian=("zp",) once equalled the
    # unweighted reduction (a silent no-op); it now weights by H. The
    # unweighted reduction is the RAW computational Integral() (the
    # seeded `u.integrate("sigma")` verb is itself Jacobian-weighted
    # now, so it equals `weighted`).
    grid, mx, ms = _terrain_grid(16)
    space = mx.center * ms.center
    u = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    weighted = Integral(jacobian=("zp",))["sigma"](u)
    plain = Integral()["sigma"](u)
    assert not jnp.allclose(weighted.data, plain.data)
    # and the seeded verb IS the weighted seam, bitwise
    assert jnp.array_equal(u.integrate("sigma").data, weighted.data)


# ================================================================
#  The seeded f.integrate() verb is physical on a maps= grid
#  (the physical-integral-default flip)
# ================================================================
def test_seeded_integrate_partial_matches_the_jacobian_seam():
    # f.integrate("sigma") dispatches the seeded Jacobian-weighted row,
    # bitwise the raw Integral(jacobian=("zp",)) seam
    grid, mx, ms = _terrain_grid(16)
    space = mx.center * ms.center
    f = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.1 * x)
    assert jnp.array_equal(
        f.integrate("sigma").data,
        Integral(jacobian=("zp",))["sigma"](f).data)


def test_seeded_full_integrate_is_the_physical_volume_integral():
    # f.integrate() == the raw computational integral of f * column
    # Jacobian (the physical integral), not the plain sigma-measure sum
    grid, mx, ms = _terrain_grid(16)
    space = mx.center * ms.center
    f = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.3 * jnp.sin(x))
    physical = f.integrate()
    jw = f * grid.metric(space.bare, "dzp_dsigma")
    manual = Integral()["x"](Integral()["sigma"](jw))
    assert jnp.allclose(physical.data, manual.data, atol=1e-13)
    # genuinely different from the plain computational reduction
    plain = Integral()["x"](Integral()["sigma"](f))
    assert not jnp.allclose(physical.data, plain.data)


def test_seeded_integrate_reduces_the_base_axis_first():
    # Hazard 1: the joint reduction reorders the single-base column base
    # axis ("sigma") ahead of its parameter axis ("x"), so the column
    # Jacobian is evaluated while x is still alive; collapsing x first
    # in a separate call is a taught error, never a silent number
    grid, mx, ms = _terrain_grid(16)
    space = mx.center * ms.center
    f = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.2 * x)
    joint = f.integrate("x", "sigma")
    safe = f.integrate("sigma").integrate("x")
    assert jnp.array_equal(joint.data, safe.data)
    # user order does not matter: the verb reorders base-first
    assert jnp.array_equal(f.integrate("sigma", "x").data, joint.data)
    with pytest.raises(ValueError, match="reduce the base axis"):
        f.integrate("x").integrate("sigma")


def test_maps_jacobian_bogus_name_raises():
    grid, mx, ms = _terrain_grid(8)
    space = mx.center * ms.center
    u = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    with pytest.raises(ValueError, match="not a chart coordinate"):
        Integral(jacobian=("bogus",))["sigma"](u)


def test_maps_jacobian_base_axis_name_raises():
    # spelling the *base* axis ("sigma") rather than the mapped
    # physical coordinate ("zp") no longer half-matches: a taught
    # error naming the available chart coordinate, not "unknown
    # metric 'sqrt_g'"
    grid, mx, ms = _terrain_grid(8)
    space = mx.center * ms.center
    u = grid.create_field(
        space, init=lambda x, sigma: jnp.cos(sigma) + 0.0 * x)
    with pytest.raises(ValueError, match="not a chart coordinate"):
        Integral(jacobian=("sigma",))["sigma"](u)


def test_maps_jacobian_on_a_grid_without_a_mapping_raises(mx, my):
    grid = Grid((mx, my))
    f = grid.create_field(mx.center * my.center,
                          init=lambda x, y: 1.0 + 0 * x + 0 * y)
    with pytest.raises(ValueError, match="no chart coordinates"):
        Integral(jacobian=("zp",))["x"](f)


def test_maps_jacobian_weighted_integral_is_differentiable():
    # the reduction sits on the hydrostatic p_hyd column path: grad
    # of a quadratic loss through the weighted integral is finite and
    # matches a central finite difference (differentiability policy)
    grid, mx, ms = _terrain_grid(8)
    space = mx.center * ms.center
    op = Integral(jacobian=("zp",))["sigma"]

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
