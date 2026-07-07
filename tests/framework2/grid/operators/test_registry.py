"""Tests for OperatorRegistry: registration, resolution, merging."""
from fractions import Fraction

import pytest

from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    Dispatched,
    OperatorRequirements,
    SeparableComposite,
)
from fridom.framework2.grid.operators.registry import (
    DispatchError,
    LazyEntry,
    OperatorRegistry,
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
#  Mapping surface
# ================================================================
def test_exact_lookup_roundtrip(mx, a):
    reg = OperatorRegistry({("diff", mx.center): a})
    assert reg[("diff", mx.center)] is a
    assert ("diff", mx.center) in reg
    assert ("diff", mx.right) not in reg


def test_setitem_registers(mx, a):
    reg = OperatorRegistry()
    reg["reconstruct"] = a
    assert reg["reconstruct"] is a
    reg[("reconstruct", mx.center)] = a
    assert ("reconstruct", mx.center) in reg


def test_getitem_is_exact_no_precedence(mx, a):
    reg = OperatorRegistry({"diff": a})
    with pytest.raises(DispatchError, match="no exact entry"):
        reg[("diff", mx.center)]


def test_keys_are_normalized_to_bare_spaces(mx, a):
    reg = OperatorRegistry()
    laid = mx.center.with_layout(Layout({}))
    reg[("diff", laid)] = a
    assert ("diff", mx.center) in reg
    assert reg[("diff", mx.center)] is a


def test_bad_keys_and_values_raise(mx, a):
    reg = OperatorRegistry()
    with pytest.raises(TypeError, match="dispatch keys"):
        reg[42] = a
    with pytest.raises(TypeError, match="dispatch keys"):
        reg[("diff", "x")] = a
    with pytest.raises(TypeError, match="registry entries"):
        reg[("diff", mx.center)] = object()


def test_items_iterates_all_entries(mx, a, b):
    reg = OperatorRegistry({("diff", mx.center): a, "reconstruct": b})
    assert dict(reg.items()) == {
        ("diff", mx.center): a, "reconstruct": b}


def test_items_dedupes_with_precedence(mx, a, b):
    reg = OperatorRegistry({("diff", mx.center): a})
    merged = reg.merge({("diff", mx.center): b})
    assert dict(merged.items()) == {("diff", mx.center): b}


def test_registry_is_identity_hashed(a):
    reg1, reg2 = OperatorRegistry({"k": a}), OperatorRegistry({"k": a})
    assert reg1 == reg1  # noqa: PLR0124 — tests __eq__
    assert reg1 != reg2
    assert hash(reg1) == id(reg1)


# ================================================================
#  Factor-space resolution precedence
# ================================================================
def test_space_specific_beats_kind_only(mx, a, b):
    reg = OperatorRegistry({"diff": a, ("diff", mx.center): b})
    assert reg.resolve("diff", mx.center) is b
    assert reg.resolve("diff", mx.right) is a


def test_missing_entry_is_a_dispatch_error(mx):
    reg = OperatorRegistry()
    with pytest.raises(DispatchError, match="no operator registered"):
        reg.resolve("diff", mx.center)


def test_dispatch_error_is_a_key_error_not_a_space_error():
    assert issubclass(DispatchError, KeyError)
    assert not issubclass(DispatchError, SpaceMismatchError)


def test_resolution_strips_layouts(mx, a):
    reg = OperatorRegistry({("diff", mx.center): a})
    laid = mx.center.with_layout(Layout({}))
    assert reg.resolve("diff", laid) is a


def test_override_kind_beats_default_space_key(mx, a, b, k):
    reg = OperatorRegistry({("diff", mx.center): a, "diff": b})
    merged = reg.merge({"diff": k})
    assert merged.resolve("diff", mx.center) is k


def test_override_space_key_beats_override_kind(mx, a, b, k):
    reg = OperatorRegistry({"diff": a})
    merged = reg.merge({"diff": b, ("diff", mx.center): k})
    assert merged.resolve("diff", mx.center) is k
    assert merged.resolve("diff", mx.right) is b


# ================================================================
#  Product-key resolution (three forms)
# ================================================================
def test_product_form1_exact_product_key(mx, my, a, b):
    prod = mx.center * my.center
    reg = OperatorRegistry({
        ("multiply", prod): a,
        ("multiply", mx.center): b,
        ("multiply", my.center): b,
    })
    assert reg.resolve("multiply", prod) is a


def test_product_form2_same_instance_per_factor(mx, my, a):
    reg = OperatorRegistry({
        ("multiply", mx.center): a,
        ("multiply", my.center): a,
    })
    assert reg.resolve("multiply", mx.center * my.center) is a


def test_product_form2_drops_constant_factors(mx, my, a):
    reg = OperatorRegistry({("multiply", mx.center): a})
    assert reg.resolve("multiply", mx.center * my.constant) is a


def test_product_form2_mixed_instances_raise(mx, my, a, b):
    reg = OperatorRegistry({
        ("multiply", mx.center): a,
        ("multiply", my.center): b,
    })
    with pytest.raises(DispatchError, match="mixed product"):
        reg.resolve("multiply", mx.center * my.center)


def test_product_form3_kind_only_fallback(mx, my, a):
    reg = OperatorRegistry({"multiply": a})
    assert reg.resolve("multiply", mx.center * my.center) is a


def test_product_no_entry_raises(mx, my):
    reg = OperatorRegistry()
    with pytest.raises(DispatchError, match="no operator registered"):
        reg.resolve("multiply", mx.center * my.center)


# ================================================================
#  Merge purity and layering
# ================================================================
def test_merge_is_pure(mx, a, b):
    reg = OperatorRegistry({("diff", mx.center): a})
    merged = reg.merge({("diff", mx.center): b})
    assert reg.resolve("diff", mx.center) is a
    assert merged.resolve("diff", mx.center) is b


def test_merge_keeps_unoverridden_defaults(mx, a, b):
    reg = OperatorRegistry({("diff", mx.center): a})
    merged = reg.merge({"reconstruct": b})
    assert merged.resolve("diff", mx.center) is a
    assert merged.resolve("reconstruct", mx.center) is b


# ================================================================
#  Dispatched-hole resolution at merge (D4)
# ================================================================
def test_merge_resolves_holes_in_space_keyed_entries(mx, a, b):
    chain = a @ Dispatched("reconstruct")
    reg = OperatorRegistry({
        ("diff", mx.center): chain,
        ("reconstruct", mx.center): b,
    })
    merged = reg.merge({})
    resolved = merged[("diff", mx.center)]
    assert resolved is (a @ b)
    assert isinstance(resolved, SeparableComposite)
    # the pre-merge registry keeps the hole (purity)
    assert reg[("diff", mx.center)] is chain


def test_merge_hole_resolution_honors_overrides(mx, a, b, k):
    chain = a @ Dispatched("reconstruct")
    reg = OperatorRegistry({
        ("diff", mx.center): chain,
        ("reconstruct", mx.center): b,
    })
    merged = reg.merge({"reconstruct": k})
    assert merged[("diff", mx.center)] is (a @ k)


def test_merge_resolves_holes_with_pending_axis(mx, a, b):
    chain = (a @ Dispatched("reconstruct"))["x"]
    reg = OperatorRegistry({
        ("diff", mx.center): chain,
        ("reconstruct", mx.center): b,
    })
    merged = reg.merge({})
    assert merged[("diff", mx.center)] is (a @ b)["x"]


def test_merge_threads_the_chain_space(mx, a, b):
    # the hole sits mid-chain: its resolution space is the codomain
    # of the factor right of it (Center -> Right)
    chain = a @ Dispatched("reconstruct") @ a
    reg = OperatorRegistry({
        ("diff", mx.center): chain,
        ("reconstruct", mx.right): b,
    })
    merged = reg.merge({})
    assert merged[("diff", mx.center)] is (a @ b @ a)


def test_merge_missing_hole_target_raises(mx, a):
    reg = OperatorRegistry({
        ("diff", mx.center): a @ Dispatched("reconstruct"),
    })
    with pytest.raises(DispatchError, match="no operator registered"):
        reg.merge({})


def test_merge_cyclic_holes_raise(mx):
    reg = OperatorRegistry({
        ("a", mx.center): Dispatched("b"),
        ("b", mx.center): Dispatched("a"),
    })
    with pytest.raises(DispatchError, match="cyclic"):
        reg.merge({})


def test_merge_leaves_kind_only_holes_for_application(mx, a):
    hole = a @ Dispatched("reconstruct")
    reg = OperatorRegistry({"diff": hole, ("reconstruct", mx.center): a})
    merged = reg.merge({})
    assert merged["diff"] is hole  # resolves at application instead


def test_merge_resolves_scaled_and_sum_entries(mx, a, b):
    entry = 2.0 * Dispatched("reconstruct") + a
    reg = OperatorRegistry({
        ("diff", mx.center): entry,
        ("reconstruct", mx.center): b,
    })
    merged = reg.merge({})
    assert merged[("diff", mx.center)] is (2.0 * b + a)


# ================================================================
#  Lazy entries (transform rows; grid lifecycle step 1)
# ================================================================
def test_lazy_entry_materializes_once_on_resolve(mx, a):
    calls = []

    def factory():
        calls.append(1)
        return a

    row = LazyEntry(factory)
    reg = OperatorRegistry({("transform", mx.center): row,
                            ("transform", mx.right): row})
    # items() reports the entry without materializing
    assert dict(reg.items())[("transform", mx.center)] is row
    assert not calls
    assert row.requirements(mx.center).halo == 0
    resolved = reg.resolve("transform", mx.center)
    assert resolved is a
    # memoized: every key sharing the entry gets the same instance
    assert reg.resolve("transform", mx.right) is a
    assert reg[("transform", mx.center)] is a
    assert len(calls) == 1


def test_lazy_entry_declared_requirements(mx):
    row = LazyEntry(lambda: None, requirements=OperatorRequirements(
        halo=0, layout="transpose"))
    assert row.requirements(mx.center).layout == "transpose"


def test_lazy_entry_validates(mx):
    with pytest.raises(TypeError, match="factory"):
        LazyEntry("not callable")
    row = LazyEntry(lambda: "not an operator")
    reg = OperatorRegistry({("transform", mx.center): row})
    with pytest.raises(TypeError, match="operators"):
        reg.resolve("transform", mx.center)


def test_lazy_product_resolution_shares_the_instance(mx, my, a):
    row = LazyEntry(lambda: a)
    reg = OperatorRegistry({("multiply", mx.center): row,
                            ("multiply", my.center): row})
    # form 2: both factors materialize the identical instance
    assert reg.resolve("multiply", mx.center * my.center) is a


# ================================================================
#  Refined-mesh adoption (grid.md normative adoption paragraph)
# ================================================================
def test_refined_mesh_adopts_parent_rows(mx, a):
    reg = OperatorRegistry({
        ("multiply", mx.center): a,
        ("multiply", mx.cell_avg): a,
        ("multiply", mx.center.as_complex()): a,
    })
    fine = mx.refined(Fraction(3, 2))
    # the same operator instances as the parent rows
    assert reg.resolve("multiply", fine.center) is a
    assert reg.resolve("multiply", fine.cell_avg) is a
    assert reg.resolve("multiply", fine.center.as_complex()) is a
    # nested refinement walks the whole chain
    finer = fine.refined(2)
    assert reg.resolve("multiply", finer.center) is a
    # missing parent rows still raise
    with pytest.raises(DispatchError, match="no operator"):
        reg.resolve("divide", fine.center)


def test_refined_adoption_covers_averages(mx, a):
    reg = OperatorRegistry({("multiply", mx.cell_avg): a,
                            ("multiply", mx.face_avg): a})
    fine = mx.refined(2)
    assert reg.resolve("multiply", fine.face_avg) is a


def test_refined_adoption_stops_at_coefficient_factors(mx, a):
    reg = OperatorRegistry({
        ("multiply", mx.fourier(origin=mx.center)): a})
    fine = mx.refined(2)
    # coefficient rows are minted by their own transforms, never
    # adopted along the refinement chain
    with pytest.raises(DispatchError, match="no operator"):
        reg.resolve("multiply", fine.fourier(origin=fine.center))


def test_refined_adoption_covers_products(mx, my, a):
    reg = OperatorRegistry({("multiply", mx.center): a,
                            ("multiply", my.center): a})
    fine = mx.refined(2)
    assert reg.resolve("multiply", fine.center * my.center) is a


def test_exact_key_beats_adoption(mx, a, b):
    fine = mx.refined(2)
    reg = OperatorRegistry({("multiply", mx.center): a,
                            ("multiply", fine.center): b})
    assert reg.resolve("multiply", fine.center) is b
