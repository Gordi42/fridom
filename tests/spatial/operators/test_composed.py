"""Tests for fridom.spatial.operators.composed."""
import jax.numpy as jnp
import pytest

from fridom.spatial.coordinate_mapping import (
    CoordinateMapping,
)
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.base import (
    Identity,
    Operator,
    OperatorSum,
    ScaledOperator,
    SeparableComposite,
    Zero,
)
from fridom.spatial.operators.composed import (
    BlockMatrix,
    Curl,
    Diag,
    Divergence,
    Gradient,
    Laplacian,
    LowerIndex,
    MetricCurl,
    MetricDivergence,
    MetricGradient,
    MetricLaplacian,
    RaiseIndex,
    VarianceRetag,
)
from fridom.spatial.operators.mapped import MetricScaled
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.scalars import Variance


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def my():
    return IntervalMesh(8, (0.0, 1.0), name="y")


@pytest.fixture
def mz():
    return IntervalMesh(8, (0.0, 1.0), name="z")


@pytest.fixture
def grid2(mx, my):
    return Grid((mx, my))


@pytest.fixture
def p(grid2):
    return grid2.create_field(
        init=lambda x, y: jnp.sin(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y))


# ================================================================
#  Factories and builder surface
# ================================================================
def test_factories_return_kind_tagged_builders():
    assert Gradient().dispatch_kind == "grad"
    assert Divergence().dispatch_kind == "div"
    assert Curl().dispatch_kind == "curl"
    assert Laplacian().dispatch_kind == "laplacian"
    assert Gradient().order is None
    assert Laplacian(order=2).order == 2
    assert Laplacian().metric is None
    assert Laplacian(metric={"z": 2.0}).metric == {"z": 2.0}


def test_builders_validate_the_order():
    with pytest.raises(ValueError, match="even order"):
        Gradient(order=3)


def test_unexpanded_builders_have_no_signature(mx):
    with pytest.raises(DispatchError, match="expand"):
        Gradient().codomain(mx.center)


def test_builders_are_seeded_kind_only(grid2, mx):
    for kind in ("grad", "div", "curl", "laplacian"):
        op = grid2.dispatch.resolve(kind, mx.center)
        assert op.dispatch_kind == kind


def test_builders_validate_the_operand_kind(p, grid2, mx, my):
    u = grid2.create_field(mx.right * my.center, name="u")
    v = grid2.create_field(mx.center * my.right, name="v")
    vec = VectorField([u, v])
    with pytest.raises(TypeError, match="scalar field"):
        Gradient()(vec)
    with pytest.raises(TypeError, match="VectorField"):
        Divergence()(p)
    with pytest.raises(TypeError, match="tuple of component"):
        Divergence().expand(p.function_space.bare, grid2.dispatch)


# ================================================================
#  Gradient
# ================================================================
def test_grad_lands_on_the_staggered_vector(p, mx, my):
    g = Gradient()(p)
    assert isinstance(g, VectorField)
    assert g.component_names == ("x", "y")
    assert g["x"].function_space.bare is mx.right * my.center
    assert g["y"].function_space.bare is mx.center * my.right


def test_grad_on_one_axis_is_the_scalar_derivative(mx):
    grid = Grid((mx,))
    f = grid.create_field(init=lambda x: jnp.sin(2 * jnp.pi * x))
    g = Gradient()(f)
    assert g.function_space.bare is mx.right
    assert jnp.allclose(g.data, f.diff("x").data)


def test_grad_matches_the_dispatched_diff(p):
    g = Gradient()(p)
    assert jnp.allclose(g["x"].data, p.diff("x").data)
    assert jnp.allclose(g["y"].data, p.diff("y").data)


# ================================================================
#  Divergence and Laplacian
# ================================================================
def test_div_of_grad_equals_laplacian_exactly(p):
    d = Divergence()(Gradient()(p))
    lap = Laplacian()(p)
    assert d.function_space.bare is lap.function_space.bare
    assert jnp.allclose(d.data, lap.data)


def test_laplacian_converges_at_second_order():
    errors = []
    for n in (16, 32):
        mesh = IntervalMesh(n, (0.0, 1.0), name="x")
        grid = Grid((mesh,))
        f = grid.create_field(
            init=lambda x: jnp.sin(2 * jnp.pi * x))
        lap = Laplacian()(f)
        exact = -(2 * jnp.pi) ** 2 * f.data
        errors.append(jnp.abs(lap.data - exact).max())
    assert errors[0] / errors[1] > 3.0


def test_laplacian_expands_to_a_sum_of_chains(p, grid2):
    block = Laplacian().expand(p.function_space.bare,
                               grid2.dispatch)
    assert isinstance(block, BlockMatrix)
    assert len(block.rows) == 1
    entry = block.rows[0][0]
    assert isinstance(entry, OperatorSum)
    assert len(entry.terms) == 2
    # per-axis second derivatives are the D5 SeparableComposites,
    # bound to their axis (operator_algebra_merge.md B1)
    assert all(isinstance(term, SeparableComposite)
               for term in entry.terms)
    assert {term.bound_axis for term in entry.terms} == {"x", "y"}
    # per-axis second derivative Center -> Right -> Center: two-sided
    # accounting composes [0,+1] and [-1,0] to the window [-1,+1],
    # width 1 (not the scalar sum 2)
    assert block.requirements(p.function_space.bare).halo == 1


def test_div_validates_the_component_count(grid2, mx, my):
    u = grid2.create_field(mx.right * my.center, name="u")
    with pytest.raises(SpaceMismatchError, match="one component"):
        Divergence()(VectorField([u]))


def test_pinned_order_matches_the_dispatched_default(p):
    default = Laplacian()(p)
    pinned = Laplacian(order=2)(p)
    assert jnp.allclose(default.data, pinned.data)


# ================================================================
#  Curl
# ================================================================
def test_curl_of_a_gradient_vanishes(p, mx, my):
    c = Curl()(Gradient()(p))
    assert c.function_space.bare is mx.right * my.right
    assert jnp.allclose(c.data, 0.0, atol=1e-12)


def test_curl_3d_produces_the_dual_staggered_vector(mx, my, mz):
    grid = Grid((mx, my, mz))
    u = grid.create_field(mx.right * my.center * mz.center,
                          name="u")
    v = grid.create_field(mx.center * my.right * mz.center,
                          name="v")
    w = grid.create_field(mx.center * my.center * mz.right,
                          name="w")
    c = Curl()(VectorField([u, v, w]))
    assert isinstance(c, VectorField)
    assert c.component_names == ("x", "y", "z")
    assert c["x"].function_space.bare is (
        mx.center * my.right * mz.right)


def test_curl_3d_solid_rotation(mx, my, mz):
    # (u, v, w) = (-y, x, 0) has curl (0, 0, 2)
    grid = Grid((mx, my, mz))
    u = grid.create_field(
        mx.right * my.center * mz.center, name="u",
        init=lambda x, y, z: -y + 0.0 * x * z)
    v = grid.create_field(
        mx.center * my.right * mz.center, name="v",
        init=lambda x, y, z: x + 0.0 * y * z)
    w = grid.create_field(
        mx.center * my.center * mz.right, name="w")
    c = Curl()(VectorField([u, v, w]))
    # the linear fields are not periodic: check the interior only
    interior = (slice(1, -1),) * 3
    assert jnp.allclose(c["x"].data[interior], 0.0, atol=1e-12)
    assert jnp.allclose(c["y"].data[interior], 0.0, atol=1e-12)
    assert jnp.allclose(c["z"].data[interior], 2.0)


def test_curl_needs_two_or_three_axes(mx):
    grid = Grid((mx,))
    u = grid.create_field(mx.right, name="u")
    with pytest.raises(SpaceMismatchError, match="2 or 3 axes"):
        Curl()(VectorField([u]))


# ================================================================
#  Diag (the pressure Laplacian's diagonal metric)
# ================================================================
def test_diag_builds_the_diagonal_block():
    axes = ("x", "y", "z")
    diag = Diag({"z": 2.0}, axes=axes)
    assert isinstance(diag, BlockMatrix)
    assert diag.output_names == axes
    ident = Identity()
    for i in range(len(axes)):
        for j in range(len(axes)):
            entry = diag.rows[i][j]
            if i != j:
                assert isinstance(entry, Zero)
            elif axes[i] == "z":
                # a non-unit weight is the scaled identity
                assert isinstance(entry, ScaledOperator)
                assert entry.coeff == 2.0
                assert entry.target is ident
            else:
                # a unit weight is the structural neutral Identity
                assert entry is ident


def test_diag_accepts_a_zero_d_array_weight():
    diag = Diag({"z": jnp.asarray(3.0)}, axes=("x", "y", "z"))
    entry = diag.rows[2][2]
    assert isinstance(entry, ScaledOperator)
    assert float(entry.coeff) == 3.0


def test_diag_default_weight_fills_absent_axes():
    diag = Diag({}, axes=("x", "y"), default=5.0)
    assert diag.rows[0][0].coeff == 5.0
    assert diag.rows[1][1].coeff == 5.0
    assert isinstance(diag.rows[0][1], Zero)


def test_diag_single_axis_has_no_output_names():
    diag = Diag({}, axes=("x",))
    assert diag.output_names is None
    assert diag.rows[0][0] is Identity()


def test_diag_needs_at_least_one_axis():
    with pytest.raises(SpaceMismatchError, match="component axis"):
        Diag({}, axes=())


def _periodic_grid():
    gx = IntervalMesh(8, (0.0, 2 * jnp.pi), periodic=True, name="x")
    gy = IntervalMesh(8, (0.0, 2 * jnp.pi), periodic=True, name="y")
    return Grid((gx, gy))


def _div_grad(grid):
    f = grid.create_field(
        init=lambda x, y: jnp.sin(x) * jnp.cos(y))
    bare = f.function_space.bare
    grad = Gradient().expand(bare, grid.dispatch)
    mid = grad.codomain(bare)
    mid = mid if isinstance(mid, tuple) else (mid,)
    div = Divergence().expand(mid, grid.dispatch)
    coeff = grid.dispatch.resolve("transform", bare).codomain(bare)
    return div, grad, coeff


def test_div_identity_diag_grad_is_the_plain_laplacian():
    # the all-unit Diag is the identity metric: Div @ Diag @ Grad has
    # exactly the plain div @ grad symbol (Identity threads the spaces)
    grid = _periodic_grid()
    div, grad, coeff = _div_grad(grid)
    weighted = div @ Diag({}, axes=("x", "y")) @ grad
    assert isinstance(weighted, BlockMatrix)
    entry = weighted.rows[0][0]
    assert isinstance(entry, OperatorSum)
    got = entry.eigenvalues(grid, coeff)
    plain = (div @ grad).rows[0][0].eigenvalues(grid, coeff)
    assert jnp.array_equal(got.data, plain.data)


def test_div_weighted_diag_grad_scales_the_matching_axis():
    # a weight w on axis y scales exactly that axis' Laplacian term
    grid = _periodic_grid()
    div, grad, coeff = _div_grad(grid)
    w = 4.0
    plain = (div @ grad).rows[0][0]
    x_term = next(t for t in plain.terms if t.bound_axis == "x")
    y_term = next(t for t in plain.terms if t.bound_axis == "y")
    expected = (x_term.eigenvalues(grid, coeff)
                + w * y_term.eigenvalues(grid, coeff))
    got = (div @ Diag({"y": w}, axes=("x", "y")) @ grad
           ).rows[0][0].eigenvalues(grid, coeff)
    assert jnp.allclose(got.data, expected.data)


# ================================================================
#  BlockMatrix (iteration-1 subset)
# ================================================================
def test_block_validation():
    ident = Identity()
    with pytest.raises(ValueError, match="at least one entry"):
        BlockMatrix(())
    with pytest.raises(ValueError, match="equal lengths"):
        BlockMatrix(((ident,), (ident, ident)))
    with pytest.raises(TypeError, match="operators"):
        BlockMatrix(((ident, 3.0),))
    with pytest.raises(ValueError, match="structural zeros"):
        BlockMatrix(((Zero(), Zero()),))
    with pytest.raises(ValueError, match="output name"):
        BlockMatrix(((ident,), (ident,)))


def test_block_codomain_and_arity(mx, my):
    ident = Identity()
    block = BlockMatrix(((ident, ident),))
    assert block.codomain(mx.center, mx.center) is mx.center
    with pytest.raises(SpaceMismatchError, match="columns"):
        block.codomain(mx.center)
    with pytest.raises(SpaceMismatchError, match="share a codomain"):
        block.codomain(mx.center, my.center)


def test_block_structure_accessors(mx):
    ident = Identity()
    zero = Zero()
    block = BlockMatrix(((ident, zero), (zero, ident)),
                        output_names=("a", "b"))
    assert block.output_names == ("a", "b")
    assert block.rows[0] == (ident, zero)
    # Zero entries are skipped by the requirements accumulation
    assert block.requirements(mx.center).halo == 0


def test_block_matmul_with_a_plain_operator_composes():
    ident = Identity()
    block = BlockMatrix(((ident, ident),))
    # non-block operands fall back to the base algebra (Identity
    # elides, so the chain normalizes back to the block itself)
    assert (block @ Identity()) is block


def test_grad_needs_a_bindable_factor(grid2, mx, my):
    f = grid2.create_field(mx.constant * my.constant)
    with pytest.raises(SpaceMismatchError, match="bindable"):
        Gradient()(f)


def test_div_needs_a_shared_axis_family(grid2, mx, my):
    a = grid2.create_field(mx.right * my.center, name="a")
    b = grid2.create_field(mx.constant * my.center, name="b")
    with pytest.raises(SpaceMismatchError, match="axis family"):
        Divergence()(VectorField([a, b]))


def test_block_matmul_shape_mismatch():
    ident = Identity()
    row = BlockMatrix(((ident, ident),))
    with pytest.raises(SpaceMismatchError, match="shape mismatch"):
        row @ row


def test_block_matmul_absorbs_structural_zeros():
    ident = Identity()
    zero = Zero()
    left = BlockMatrix(((zero, ident),))
    right = BlockMatrix(((ident, zero), (zero, ident)),
                        output_names=("a", "b"))
    product = left @ right
    # zero chains drop from the entry sums; fully cancelled
    # entries stay structural zeros; identity chains normalize
    assert isinstance(product.rows[0][0], Zero)
    assert isinstance(product.rows[0][1], Identity)


def test_block_application_arity(grid2, mx, my):
    ident = Identity()
    block = BlockMatrix(((ident, ident),))
    f = grid2.create_field(mx.center * my.center)
    with pytest.raises(ValueError, match="component"):
        block(f)


# ================================================================
#  codomains (always-tuple accessor, R4)
# ================================================================
def test_codomains_wraps_a_single_space_on_the_base(mx):
    # the Operator base wraps a single-signature codomain in a tuple
    assert Identity().codomains(mx.center) == (mx.center,)


def test_codomains_returns_a_tuple_for_grad_and_laplacian(
        grid2, p, mx, my):
    bare = p.function_space.bare
    grad = Gradient().expand(bare, grid2.dispatch)
    # grad is multi-row: codomains is the true per-row tuple
    cod = grad.codomains(bare)
    assert isinstance(cod, tuple)
    assert cod == (mx.right * my.center, mx.center * my.right)
    # laplacian is 1x1: codomain is a bare space, codomains wraps it
    lap = Laplacian().expand(bare, grid2.dispatch)
    assert not isinstance(lap.codomain(bare), tuple)
    assert lap.codomains(bare) == (lap.codomain(bare),)


# ================================================================
#  scalar (1x1 block collapse, R5)
# ================================================================
def test_scalar_collapses_a_1x1_block(grid2, p):
    bare = p.function_space.bare
    lap = Laplacian().expand(bare, grid2.dispatch)
    assert lap.scalar() is lap.rows[0][0]


def test_scalar_rejects_a_non_1x1_block():
    ident = Identity()
    with pytest.raises(SpaceMismatchError, match="1x1"):
        BlockMatrix(((ident, ident),)).scalar()
    with pytest.raises(SpaceMismatchError, match="1x1"):
        BlockMatrix(((ident,), (ident,)),
                    output_names=("a", "b")).scalar()


# ================================================================
#  Weighted Laplacian (metric, R6) and grid-or-registry expand (R7)
# ================================================================
def test_laplacian_unweighted_is_unchanged(grid2, p):
    # metric=None expands byte-for-byte to the plain div @ grad
    bare = p.function_space.bare
    grad = Gradient().expand(bare, grid2.dispatch)
    div = Divergence().expand(grad.codomains(bare), grid2.dispatch)
    assert Laplacian().expand(bare, grid2.dispatch).scalar() is (
        (div @ grad).scalar())


def test_laplacian_metric_scales_the_matching_axis():
    grid = _periodic_grid()
    f = grid.create_field(
        init=lambda x, y: jnp.sin(x) * jnp.cos(y))
    bare = f.function_space.bare
    coeff = grid.dispatch.resolve("transform", bare).codomain(bare)
    w = 4.0
    weighted = Laplacian(metric={"y": w}).expand(bare, grid).scalar()
    plain = Laplacian().expand(bare, grid).scalar()
    x_term = next(t for t in plain.terms if t.bound_axis == "x")
    y_term = next(t for t in plain.terms if t.bound_axis == "y")
    # the weight scales exactly the y-axis Laplacian term
    expected = (x_term.eigenvalues(grid, coeff)
                + w * y_term.eigenvalues(grid, coeff))
    got = weighted.eigenvalues(grid, coeff)
    assert jnp.allclose(got.data, expected.data)
    # and the weighted symbol genuinely differs from the plain one
    assert not jnp.allclose(
        got.data, plain.eigenvalues(grid, coeff).data)


def test_laplacian_unit_metric_matches_the_unweighted_symbol():
    # a unit weight on an axis leaves that axis' term unchanged
    grid = _periodic_grid()
    f = grid.create_field(
        init=lambda x, y: jnp.sin(x) * jnp.cos(y))
    bare = f.function_space.bare
    coeff = grid.dispatch.resolve("transform", bare).codomain(bare)
    weighted = Laplacian(metric={"x": 1.0}).expand(
        bare, grid).scalar()
    plain = Laplacian().expand(bare, grid).scalar()
    assert jnp.allclose(weighted.eigenvalues(grid, coeff).data,
                        plain.eigenvalues(grid, coeff).data)


def test_expand_accepts_a_grid_or_a_registry(grid2, p):
    bare = p.function_space.bare
    from_grid = Gradient().expand(bare, grid2)
    from_reg = Gradient().expand(bare, grid2.dispatch)
    assert from_grid.rows == from_reg.rows
    # the laplacian path too (grid vs registry give the same block)
    assert Laplacian().expand(bare, grid2).scalar() is (
        Laplacian().expand(bare, grid2.dispatch).scalar())


# ================================================================
#  Metric-aware vector calculus (stage C2)
# ================================================================
TWO_PI = 2.0 * jnp.pi
RING, MINOR = 2.0, 0.7


@pytest.fixture
def mu():
    return IntervalMesh(16, (0.0, float(TWO_PI)), name="u")


@pytest.fixture
def mv():
    return IntervalMesh(16, (0.0, float(TWO_PI)), name="v")


@pytest.fixture
def chart_grid(mu, mv):
    mapping = CoordinateMapping(chart={"X": lambda u, v: (
        (RING + MINOR * jnp.cos(v)) * jnp.cos(u),
        (RING + MINOR * jnp.cos(v)) * jnp.sin(u),
        MINOR * jnp.sin(v))})
    return Grid((mu, mv), mapping=mapping)


@pytest.fixture
def cov_vec(chart_grid, mu, mv):
    uu = chart_grid.create_field(
        mu.right * mv.center,
        init=lambda u, v: jnp.sin(u + v)).with_variance(
            Variance.COVARIANT)
    vv = chart_grid.create_field(
        mu.center * mv.right,
        init=lambda u, v: jnp.cos(u - 2 * v)).with_variance(
            Variance.COVARIANT)
    return VectorField({"u": uu, "v": vv})


# ------------------------------------------------------------
#  VarianceRetag
# ------------------------------------------------------------
def test_variance_retag_validates():
    with pytest.raises(TypeError, match="Operator"):
        VarianceRetag("nope", Variance.COVARIANT)
    with pytest.raises(TypeError, match="Variance"):
        VarianceRetag(Identity(), "cov")


def test_variance_retag_codomain_and_requirements(mu, mv):
    space = (mu.right * mv.center).with_variance(Variance.COVARIANT)
    op = VarianceRetag(Identity(), Variance.CONTRAVARIANT)
    assert op.target is Identity()
    assert op.variance is Variance.CONTRAVARIANT
    assert op.codomain(space) is space.with_variance(
        Variance.CONTRAVARIANT)
    assert op.requirements(space) == Identity().requirements(space)
    with pytest.raises(SpaceMismatchError, match="unary"):
        op.codomain(space, space)


def test_variance_retag_is_a_pure_claim(chart_grid, mu, mv):
    f = chart_grid.create_field(
        mu.center * mv.center, init=lambda u, v: jnp.sin(u) + 0 * v)
    out = VarianceRetag(Identity(), Variance.COVARIANT)(f)
    assert out.function_space.variance is Variance.COVARIANT
    assert out._data is f._data
    strip = VarianceRetag(Identity(), None)(out)
    assert strip.function_space.variance is None
    # retagging onto the current claim is the identity
    assert VarianceRetag(Identity(), None)(f) is f


# ------------------------------------------------------------
#  Builder validation surface
# ------------------------------------------------------------
def test_metric_builders_validate_coords():
    with pytest.raises(TypeError, match="coords"):
        MetricGradient(("u",))
    with pytest.raises(TypeError, match="coords"):
        MetricDivergence(("u", "u"))
    with pytest.raises(TypeError, match="coords"):
        RaiseIndex((1, 2))


def test_metric_builders_have_no_unexpanded_signature(mu, mv):
    with pytest.raises(DispatchError, match="expands"):
        MetricGradient(("u", "v")).codomain(mu.center * mv.center)


def test_metric_builder_arity_validation(mu, mv, chart_grid):
    space = mu.center * mv.center
    with pytest.raises(TypeError, match="one scalar operand"):
        MetricGradient(("u", "v")).expand((space,), chart_grid)
    with pytest.raises(TypeError, match="tuple of component"):
        MetricDivergence(("u", "v")).expand(space, chart_grid)


def test_metric_builder_operand_type_errors(chart_grid, mu, mv,
                                            cov_vec):
    f = chart_grid.create_field(
        mu.center * mv.center, init=lambda u, v: 0 * u + 0 * v)
    with pytest.raises(TypeError, match="scalar field"):
        MetricGradient(("u", "v"))(cov_vec)
    with pytest.raises(TypeError, match="VectorField"):
        MetricDivergence(("u", "v"))(f)


def test_metric_grad_requires_untagged_scalar(chart_grid, mu, mv):
    tagged = (mu.center * mv.center).with_variance(
        Variance.COVARIANT)
    with pytest.raises(SpaceMismatchError, match="untagged scalar"):
        MetricGradient(("u", "v")).expand(tagged, chart_grid)


def test_metric_entries_pin_the_chart_axes(chart_grid, mu, mv):
    lone = mu.center * mv.constant
    with pytest.raises(SpaceMismatchError, match="chart"):
        MetricGradient(("u", "v")).expand(lone, chart_grid)


def test_metric_div_demands_contravariant_components(
        chart_grid, cov_vec):
    domains = tuple(c.function_space.bare for c in cov_vec)
    with pytest.raises(SpaceMismatchError, match="raise the index"):
        MetricDivergence(("u", "v")).expand(domains, chart_grid)


def test_metric_curl_demands_covariant_components(chart_grid, mu,
                                                  mv):
    con = tuple(s.with_variance(Variance.CONTRAVARIANT) for s in
                ((mu.right * mv.center).bare,
                 (mu.center * mv.right).bare))
    with pytest.raises(SpaceMismatchError, match="lower the index"):
        MetricCurl(("u", "v")).expand(con, chart_grid)


def test_metric_curl_is_two_dimensional(chart_grid, mu, mv):
    mw = IntervalMesh(16, (0.0, 1.0), name="w")
    spaces = tuple(
        s.with_variance(Variance.COVARIANT) for s in (
            mu.right * mv.center * mw.center,
            mu.center * mv.right * mw.center,
            mu.center * mv.center * mw.right))
    with pytest.raises(SpaceMismatchError, match="2D scalar"):
        MetricCurl(("u", "v", "w")).expand(spaces, chart_grid)


# ------------------------------------------------------------
#  Expansion structure (metric names on the right spaces)
# ------------------------------------------------------------
def test_metric_grad_expands_to_tagged_diff_entries(chart_grid, mu,
                                                    mv):
    space = mu.center * mv.center
    block = MetricGradient(("u", "v")).expand(space, chart_grid)
    assert isinstance(block, BlockMatrix)
    assert block.output_names == ("u", "v")
    entry = block.rows[0][0]
    assert isinstance(entry, VarianceRetag)
    assert entry.variance is Variance.COVARIANT
    assert entry.target is chart_grid.dispatch.resolve(
        "diff", mu.center)["u"]
    assert block.codomains(space) == (
        (mu.right * mv.center).with_variance(Variance.COVARIANT),
        (mu.center * mv.right).with_variance(Variance.COVARIANT))


def test_metric_div_expands_the_flux_form(chart_grid, mu, mv):
    domains = tuple(
        s.with_variance(Variance.CONTRAVARIANT) for s in (
            (mu.right * mv.center).bare,
            (mu.center * mv.right).bare))
    block = MetricDivergence(("u", "v")).expand(domains, chart_grid)
    assert isinstance(block, BlockMatrix)
    (row,) = block.rows
    for entry in row:
        assert isinstance(entry, VarianceRetag)
        assert entry.variance is None
        outer = entry.target
        assert isinstance(outer, MetricScaled)
        assert outer.numerator is None
        assert outer.denominator == "sqrt_g"
    assert block.codomain(*domains) is (mu.center * mv.center).bare


def test_raise_index_diagonal_flag_drops_cross_terms(chart_grid, mu,
                                                     mv):
    domains = tuple(
        s.with_variance(Variance.COVARIANT) for s in (
            (mu.right * mv.center).bare,
            (mu.center * mv.right).bare))
    dense = RaiseIndex(("u", "v")).expand(domains, chart_grid)
    sparse = RaiseIndex(("u", "v"), diagonal=True).expand(
        domains, chart_grid)
    assert not RaiseIndex(("u", "v")).diagonal
    assert RaiseIndex(("u", "v"), diagonal=True).diagonal
    assert isinstance(dense.rows[0][1], VarianceRetag)
    assert isinstance(sparse.rows[0][1], Zero)
    diag = sparse.rows[0][0]
    assert diag.variance is Variance.CONTRAVARIANT
    assert diag.target.numerator == "inv_g_uu"
    cross = dense.rows[0][1].target
    assert isinstance(cross, MetricScaled)
    assert cross.numerator == "inv_g_uv"


def test_bounded_chart_cross_term_teaches_the_diagonal_fix():
    # across a wall there is no interpolation row for the cross term;
    # the raw DispatchError never names the fix, so raise/lower re-
    # raise a taught error pointing at orthogonal=True / diagonal=True
    # (chart-ergonomics E2)
    mlon = IntervalMesh(16, (0.0, float(TWO_PI)), name="lon")
    mlat = IntervalMesh(8, (-1.0, 1.0), periodic=False, name="lat")
    grid = Grid((mlon, mlat), mapping=CoordinateMapping(chart={
        "X": lambda lon, lat: (jnp.cos(lat) * jnp.cos(lon),
                               jnp.cos(lat) * jnp.sin(lon),
                               jnp.sin(lat))}))
    domains = tuple(
        s.with_variance(Variance.COVARIANT) for s in (
            (mlon.right * mlat.center).bare,
            (mlon.center * mlat.right).bare))
    with pytest.raises(DispatchError,
                       match=r"orthogonal=True.*diagonal=True"):
        RaiseIndex(("lon", "lat")).expand(domains, grid)
    contra = tuple(
        s.with_variance(Variance.CONTRAVARIANT) for s in (
            (mlon.right * mlat.center).bare,
            (mlon.center * mlat.right).bare))
    with pytest.raises(DispatchError, match=r"lower_index.*lon<->lat"):
        LowerIndex(("lon", "lat")).expand(contra, grid)


def test_raise_lower_round_trip_is_identity(chart_grid, cov_vec):
    raised = chart_grid.dispatch.resolve(
        "raise_index", cov_vec[0].function_space.bare)(cov_vec)
    for component in raised:
        assert component.function_space.variance is (
            Variance.CONTRAVARIANT)
    back = chart_grid.dispatch.resolve(
        "lower_index", cov_vec[0].function_space.bare)(raised)
    for name in cov_vec.component_names:
        assert back[name].function_space is (
            cov_vec[name].function_space)
        assert jnp.allclose(back[name].data, cov_vec[name].data,
                            atol=1e-12)


def test_lower_index_demands_contravariant(chart_grid, cov_vec):
    domains = tuple(c.function_space.bare for c in cov_vec)
    with pytest.raises(SpaceMismatchError, match="contravariant"):
        LowerIndex(("u", "v")).expand(domains, chart_grid)


def test_metric_laplacian_composes_through_the_kinds(chart_grid, mu,
                                                     mv):
    # overriding the raise leg must propagate into the laplacian
    space = (mu.center * mv.center).bare
    dense = MetricLaplacian(("u", "v")).expand(space, chart_grid)
    chart_grid.merge_overrides({
        "raise_index": RaiseIndex(("u", "v"), diagonal=True)})
    sparse = MetricLaplacian(("u", "v")).expand(space, chart_grid)
    f = chart_grid.create_field(
        space, init=lambda u, v: jnp.sin(v) + jnp.cos(u))
    # the torus metric is diagonal: both expansions agree numerically
    assert jnp.allclose(dense(f).data, sparse(f).data, atol=1e-12)


def test_metric_laplacian_needs_expandable_legs(chart_grid, mu, mv):
    space = (mu.center * mv.center).bare
    chart_grid.merge_overrides({"grad": Identity()})
    with pytest.raises(DispatchError, match="expand"):
        MetricLaplacian(("u", "v")).expand(space, chart_grid)


def test_interp_onto_reports_unjoined_staggerings(chart_grid, mu,
                                                  mv):
    class StayPut(Operator):

        """Fake interpolate row that never moves the staggering."""

        def codomain(self, domain):
            return domain

        def __getitem__(self, axis):
            return self

        def __call__(self, f):  # pragma: no cover - never applied
            return f

    class Stay:
        def resolve(self, kind, space):
            if kind == "interpolate":
                return StayPut()
            return chart_grid.dispatch.resolve(kind, space)

    domains = tuple(
        s.with_variance(Variance.COVARIANT) for s in (
            (mu.right * mv.center).bare,
            (mu.center * mv.right).bare))
    with pytest.raises(SpaceMismatchError, match="interpolate rows"):
        RaiseIndex(("u", "v")).expand(domains, Stay())


def test_raise_index_on_collocated_components(chart_grid, mu, mv):
    # an A-grid vector: both components collocated at centers — the
    # cross-term chain is the bare Identity (no staggering hops)
    space = mu.center * mv.center
    uu = chart_grid.create_field(
        space, init=lambda u, v: jnp.sin(u + v)).with_variance(
            Variance.COVARIANT)
    vv = chart_grid.create_field(
        space, init=lambda u, v: jnp.cos(u - v)).with_variance(
            Variance.COVARIANT)
    vec = VectorField({"u": uu, "v": vv})
    raised = RaiseIndex(("u", "v"))(vec)
    # the torus metric is diagonal: u^u = inv_g_uu u_u pointwise
    inv_g_uu = chart_grid.metric(uu.function_space.bare,
                                 "inv_g_uu")
    assert jnp.allclose(raised["u"].data,
                        inv_g_uu.data * uu.data, atol=1e-12)


def test_lower_index_diagonal_flag():
    assert LowerIndex(("u", "v"), diagonal=True).diagonal
    assert not LowerIndex(("u", "v")).diagonal
