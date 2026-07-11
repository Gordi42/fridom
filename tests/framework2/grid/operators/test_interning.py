"""Family-wide leaf-operator interning (D6, the ``@interned`` decorator).

Covers every leaf that ``operators.interned.interned`` was applied to:
(1) structural self-interning (``Cls(a) is Cls(a)``), (2) distinct keys
give distinct objects, (3) binding-safety for the separable leaves (the
``_rebind`` ``copy.copy`` hazard the ``__copy__`` seam fixes), and (4) a
``fr.utils.jaxify`` pytree round-trip. The three in-flight classes
(``WenoReconstruction`` / ``UpwindOne`` / ``Fallback``) keep their own
identity tests in ``test_weno.py`` / ``test_fallback.py``; ``UpwindOne``
also rides the separable table here since it now folds onto
``@interned``.
"""
import copy

import jax
import pytest

from fridom.framework2.grid.operators.fallback import UpwindOne
from fridom.framework2.grid.operators.finite_difference import (
    FiniteDifference,
)
from fridom.framework2.grid.operators.flux_diff import (
    DualFluxDifference,
    FaceDifference,
    FluxDifference,
)
from fridom.framework2.grid.operators.integrate import Integral
from fridom.framework2.grid.operators.interp import LinearInterp
from fridom.framework2.grid.operators.products import (
    Abs,
    CollocationProduct,
    Divide,
    Power,
)
from fridom.framework2.grid.operators.reconstruct import (
    LinearReconstruction,
)
from fridom.framework2.grid.operators.select import Where
from fridom.framework2.grid.operators.spectral import (
    PhaseShift,
    SincShift,
    SpectralDerivative,
)
from fridom.framework2.grid.spaces.nodal import NodeSet

# ================================================================
#  Specs: id, a builder for the interned instance, and a builder
#  for a DISTINCT-key sibling (None when the leaf has a single
#  structural key, i.e. its intern key is the empty tuple).
# ================================================================
SEPARABLE = [
    ("FiniteDifference",
     lambda: FiniteDifference(2), lambda: FiniteDifference(4)),
    ("LinearInterp",
     LinearInterp, lambda: LinearInterp(NodeSet.OUTER)),
    ("FluxDifference", FluxDifference, None),
    ("DualFluxDifference", DualFluxDifference, None),
    ("FaceDifference", FaceDifference, None),
    ("Integral", Integral, None),
    ("SpectralDerivative", SpectralDerivative, None),
    ("PhaseShift", PhaseShift, lambda: PhaseShift(NodeSet.LEFT)),
    ("SincShift", SincShift, lambda: SincShift(NodeSet.LEFT)),
    ("LinearReconstruction",
     LinearReconstruction,
     lambda: LinearReconstruction(NodeSet.OUTER)),
    ("UpwindOne",
     lambda: UpwindOne("left"), lambda: UpwindOne("right")),
]

WHOLE = [
    ("Where", Where, None),
    ("CollocationProduct", CollocationProduct, None),
    ("Divide", Divide, None),
    ("Power", Power, None),
    ("Abs", Abs, None),
]

ALL = SEPARABLE + WHOLE


def _ids(specs):
    return [s[0] for s in specs]


# ================================================================
#  (1) Structural self-interning
# ================================================================
@pytest.mark.parametrize(
    ("make_a", "make_b"), [(s[1], s[2]) for s in ALL], ids=_ids(ALL))
def test_self_interns_on_structure(make_a, make_b):  # noqa: ARG001
    # structurally-equal requests return the identical object (D6)
    assert make_a() is make_a()


# ================================================================
#  (2) Distinct keys give distinct objects
# ================================================================
_DISTINCT = [s for s in ALL if s[2] is not None]


@pytest.mark.parametrize(
    ("make_a", "make_b"),
    [(s[1], s[2]) for s in _DISTINCT], ids=_ids(_DISTINCT))
def test_distinct_keys_are_distinct_objects(make_a, make_b):
    a, b = make_a(), make_b()
    assert a is not b
    assert type(a) is type(b)


def test_cross_class_keys_never_collide():
    # empty-key leaves must not coalesce across classes (the class is
    # prepended to the interning key)
    assert FluxDifference() is not DualFluxDifference()
    assert FluxDifference() is not FaceDifference()
    assert Where() is not CollocationProduct()
    assert Divide() is not Power()


# ================================================================
#  (3) Binding-safety for the separable leaves (the _rebind
#      copy.copy hazard that __copy__ fixes)
# ================================================================
@pytest.mark.parametrize(
    ("make_a", "make_b"),
    [(s[1], s[2]) for s in SEPARABLE], ids=_ids(SEPARABLE))
def test_binding_does_not_corrupt_the_singleton(make_a, make_b):  # noqa: ARG001
    op = make_a()
    bound = op["x"]
    # the unbound interned singleton is untouched by binding
    assert op.bound_axis is None
    assert bound.bound_axis == "x"
    # binding is itself interned on (base, axis)
    assert op["x"] is op["x"]
    # the bound variant points back at the unbound original
    assert bound.unbound is op
    # copy.copy hands _rebind a fresh mutable clone, not the singleton
    clone = copy.copy(op)
    assert clone is not op
    assert type(clone) is type(op)


# ================================================================
#  (4) jaxify pytree round-trip (no regression: _tree_unflatten uses
#      object.__new__, bypassing the interning __new__)
# ================================================================
@pytest.mark.parametrize(
    ("make_a", "make_b"), [(s[1], s[2]) for s in ALL], ids=_ids(ALL))
def test_jaxify_round_trip(make_a, make_b):  # noqa: ARG001
    op = make_a()
    leaves, treedef = jax.tree_util.tree_flatten(op)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(op)
    # structural key survives the round-trip
    assert rebuilt._intern_key() == op._intern_key()
