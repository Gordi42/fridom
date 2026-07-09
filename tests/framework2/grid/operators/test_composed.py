"""Tests for fridom.framework2.grid.operators.composed."""
import jax.numpy as jnp
import pytest

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.operators.base import (
    Identity,
    OperatorSum,
    ScaledOperator,
    SeparableComposite,
    Zero,
)
from fridom.framework2.grid.operators.composed import (
    BlockMatrix,
    Curl,
    Diag,
    Divergence,
    Gradient,
    Laplacian,
)
from fridom.framework2.grid.operators.registry import DispatchError


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
    # per-axis second derivatives: summed chain halo of 2
    assert block.requirements(p.function_space.bare).halo == 2


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
