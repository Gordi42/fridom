"""Tests for the operator algebra: laws, interning, codomains."""
import pytest

from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    Composite,
    EigenbasisError,
    Identity,
    OperatorSum,
    ScaledOperator,
    SeparableComposite,
    Zero,
    _merge_layout,
    resolve_codomain,
)


@pytest.fixture
def a(stagger_cls):
    return stagger_cls()


@pytest.fixture
def b(stagger_cls):
    return stagger_cls()


@pytest.fixture
def k(keep_cls):
    return keep_cls()


# ================================================================
#  Chain flattening and associativity
# ================================================================
def test_chains_flatten(a, b, k):
    chain = (a @ b) @ k
    assert chain.factors == (a, b, k)


def test_composition_is_associative_via_flattening(a, b, k):
    assert ((a @ b) @ k) is (a @ (b @ k))


def test_no_nested_composites(a, b, k):
    chain = a @ (b @ k)
    assert all(not isinstance(f, SeparableComposite | Composite)
               for f in chain.factors)


# ================================================================
#  Identity elision and Zero absorption/dropping
# ================================================================
def test_identity_is_a_singleton():
    assert Identity() is Identity()


def test_zero_is_a_singleton():
    assert Zero() is Zero()


def test_identity_elides_in_chains(a):
    assert (a @ Identity()) is a
    assert (Identity() @ a) is a
    assert (Identity() @ Identity()) is Identity()


def test_zero_absorbs_in_chains(a):
    assert (Zero() @ a) is Zero()
    assert (a @ Zero()) is Zero()


def test_zero_drops_in_sums(a):
    assert (a + Zero()) is a
    assert (Zero() + a) is a
    assert (Zero() + Zero()) is Zero()


def test_scaling_zero_is_zero():
    assert (2 * Zero()) is Zero()


# ================================================================
#  Interning, decision D6
# ================================================================
def test_chain_interning(a, b):
    assert (a @ b) is (a @ b)
    assert (a @ b) is not (b @ a)


def test_bound_variant_interning(a):
    assert a["x"] is a["x"]
    assert a["x"] is not a["y"]


def test_sum_interning(a, b):
    assert (a + b) is (a + b)


def test_binding_distributes_over_composition(a, b):
    assert (a @ b)["x"] is (a["x"] @ b["x"])


def test_bound_chain_flattens_back(a, b, k):
    assert ((a @ b)["x"] @ k["x"]) is (a @ b @ k)["x"]


# ================================================================
#  Separable vs whole-space composites (D5)
# ================================================================
def test_unbound_separable_chain_is_separable(a, b):
    chain = a @ b
    assert isinstance(chain, SeparableComposite)
    assert chain.bound_axis is None


def test_same_axis_chain_is_bound_separable(a, b):
    chain = a["x"] @ b["x"]
    assert isinstance(chain, SeparableComposite)
    assert chain.bound_axis == "x"
    assert chain.factors == (a, b)  # canonical: unbound factors


def test_mixed_axis_chain_is_whole_space(a, b):
    chain = a["x"] @ b["y"]
    assert isinstance(chain, Composite)
    assert chain.factors == (a["x"], b["y"])


def test_mixed_bound_unbound_chain_is_whole_space(a, b):
    assert isinstance(a["x"] @ b, Composite)


def test_whole_space_factor_forces_composite(a, whole_cls):
    assert isinstance(whole_cls() @ a, Composite)


def test_composites_are_not_bindable(a, b):
    with pytest.raises(TypeError, match="fixed signature"):
        (a["x"] @ b["y"])["x"]


# ================================================================
#  Codomain threading
# ================================================================
def test_separable_codomain_per_factor(mx, a):
    assert a.codomain(mx.center) is mx.right
    assert a.codomain(mx.right) is mx.center


def test_chain_codomain_threads(mx, a, b):
    assert (a @ b).codomain(mx.center) is mx.center
    assert (a @ b @ a).codomain(mx.center) is mx.right


def test_resolve_codomain_bound_product(mx, my, a):
    prod = mx.center * my.center
    assert resolve_codomain(a["x"], prod) is (mx.right * my.center)
    assert resolve_codomain(a["y"], prod) is (mx.center * my.right)


def test_resolve_codomain_strips_and_ignores_layout(mx, my, a):
    laid = (mx.center * my.center).with_layout(Layout({}))
    assert resolve_codomain(a["x"], laid) is (mx.right * my.center)


def test_resolve_codomain_composite_product(mx, my, a, b):
    prod = mx.center * my.center
    chain = a["x"] @ b["y"]
    assert resolve_codomain(chain, prod) is (mx.right * my.right)


def test_constant_factor_application_is_identity(mx, my, a):
    prod = mx.center * my.constant
    assert resolve_codomain(a["y"], prod) is prod


def test_unbound_resolves_the_sole_bindable_factor(mx, my, a):
    prod = mx.center * my.constant
    assert resolve_codomain(a, prod) is (mx.right * my.constant)


def test_unbound_on_ambiguous_product_raises(mx, my, a):
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_codomain(a, mx.center * my.center)


def test_bound_axis_not_in_operand_raises(mx, a):
    with pytest.raises(SpaceMismatchError, match="bound to axis"):
        resolve_codomain(a["y"], mx.center)


def test_identity_codomain_unchanged(mx):
    assert Identity().codomain(mx.center) is mx.center
    assert Zero().codomain(mx.center) is mx.center


# ================================================================
#  Sums and scalings
# ================================================================
def test_sum_flattens(a, b, k):
    total = (a + b) + k
    assert isinstance(total, OperatorSum)
    assert total.terms == (a, b, k)


def test_sum_codomain_common_signature(mx, a, b):
    assert (a + b).codomain(mx.center) is mx.right


def test_sum_codomain_mismatch_raises(mx, a, k):
    with pytest.raises(SpaceMismatchError, match="share a signature"):
        (a + k).codomain(mx.center)


def test_neg_and_sub_build_scaled_sums(a, b):
    neg = -a
    assert isinstance(neg, ScaledOperator)
    assert neg.coeff == -1
    assert neg.target is a
    diff = a - b
    assert isinstance(diff, OperatorSum)
    assert diff.terms[0] is a
    assert isinstance(diff.terms[1], ScaledOperator)
    assert diff.terms[1].target is b


def test_scalar_multiplication(a):
    scaled = 2.5 * a
    assert isinstance(scaled, ScaledOperator)
    assert scaled.coeff == 2.5
    assert scaled.target is a
    assert (a * 2.5).coeff == 2.5


def test_scaled_codomain_and_requirements(mx, a):
    scaled = 2 * a
    assert scaled.codomain(mx.center) is mx.right
    assert scaled.requirements(mx.center) == a.requirements(mx.center)


def test_scaled_field_coefficient_has_no_symbol(mx, a):
    scaled = object() * a  # any non-scalar coefficient
    with pytest.raises(EigenbasisError, match="translation"):
        scaled.eigenvalues(None, mx.center)


def test_pow_is_the_interned_chain(a):
    assert (a ** 2) is (a @ a)
    assert (a ** 1) is a
    assert (a ** 0) is Identity()


# ================================================================
#  Requirements accounting (rules 3.6)
# ================================================================
def test_chain_halo_sums(mx, stagger_cls):
    a, b = stagger_cls(halo=1), stagger_cls(halo=2)
    assert (a @ b).requirements(mx.center).halo == 3


def test_bound_chain_halo_sums_on_product(mx, my, stagger_cls):
    a, b = stagger_cls(halo=1), stagger_cls(halo=2)
    chain = (a @ b)["x"]
    prod = mx.center * my.center
    assert chain.requirements(prod).halo == 3


def test_whole_space_chain_halo_sums(mx, my, stagger_cls):
    a, b = stagger_cls(halo=1), stagger_cls(halo=2)
    chain = a["x"] @ b["y"]
    assert chain.requirements(mx.center * my.center).halo == 3


def test_sum_halo_maxes(mx, stagger_cls):
    a, b = stagger_cls(halo=1), stagger_cls(halo=3)
    assert (a + b).requirements(mx.center).halo == 3


def test_requirement_layout_conflicts_raise():
    assert _merge_layout("any", "local") == "local"
    assert _merge_layout("transpose", "any") == "transpose"
    with pytest.raises(ValueError, match="conflicting"):
        _merge_layout("local", "transpose")


# ================================================================
#  Symbols along chains stay optional capabilities
# ================================================================
def test_chain_eigenvalues_propagate_the_error(mx, a, b):
    with pytest.raises(EigenbasisError):
        (a @ b).eigenvalues(None, mx.center)
