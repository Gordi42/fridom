"""Tests for the Module base class (framework2/model/module.py)."""
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from fridom.framework.utils import jaxify
from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.errors import (
    ImmutableParameterError,
    TimeDependentParameterError,
)
from fridom.model.module import BindParameterView, Module
from fridom.model.parameters import (
    ParameterDeclaration,
    ParameterReference,
)
from fridom.model.roles import ADVECTED
from fridom.model.stages import Stage, StageKind, self_update
from fridom.model.terms import term
from fridom.model.time_dependent import Ramp, resolve_at
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Profile


# ================================================================
#  The toy module (API-sketch 7.2 style) and the duck-typed table
# ================================================================
@partial(jaxify, dynamic=("n2_input", "kh"))
class ToyStratification(Module):

    """Sketch-7.2-style module exercising the full capability menu."""

    def __init__(self, n2=1e-5, kh=1.0):
        self.n2_input = (n2 if isinstance(n2, Ramp)
                         else jnp.asarray(n2))
        self.kh = jnp.asarray(kh)

    field_declarations = property(lambda self: (  # noqa: ARG005
        FieldDeclaration.tracer("b", long_name="Buoyancy",
                                units="m/s^2"),
        FieldDeclaration("n2", space=Profile(),
                         lifecycle=Lifecycle.AUXILIARY,
                         units="1/s^2"),
    ))
    field_references = (FieldReference(
        "w", hint="buoyancy couples to vertical velocity"),)
    parameter_declarations = (ParameterDeclaration(
        "stratification.n2", attr="n2_input", units="1/s^2"),)
    parameter_references = (ParameterReference(
        "forcing.amp", default=0.0),)

    stages = (Stage(kind=StageKind.CONSTRAINT, fn="clamp_b"),)

    def bind(self, table):
        super().bind(table)
        self._advected = tuple(table.select(ADVECTED))
        self._amp0 = float(
            table.parameters.at_time(0.0)["forcing.amp"])

    @self_update
    def self_update(self, state, ctx):  # noqa: ARG002
        return {"n2": resolve_at(self.n2_input, ctx.clock.time)}

    @term(advances=("w",), linear=True)
    def buoyancy_force(self, state, ctx):
        return {"w": state["b"] / ctx.params["nonhydro.dsqr"]}

    @term(advances=("b",), linear=True)
    def restoring(self, state, ctx):  # noqa: ARG002
        return {"b": -(state["w"] * state["n2"])}

    def clamp_b(self, state, ctx):  # noqa: ARG002
        return {"b": state["b"]}


class DuckTable:

    """Duck-typed bind table (FieldTable + the parameters seam)."""

    def __init__(self, parameters=None, advected=("b",)):
        values = ({"forcing.amp": 0.0} if parameters is None
                  else parameters)
        self.parameters = BindParameterView(values)
        self._advected = tuple(advected)

    def select(self, role):
        return self._advected if role == ADVECTED else ()


class FailingTable(DuckTable):

    """A table whose role query fails (failed-bind tests)."""

    def select(self, role):  # noqa: ARG002
        raise RuntimeError("table exploded")


def make_ctx(time=0.0, params=None):
    return SimpleNamespace(
        params={"nonhydro.dsqr": 1.0, **(params or {})},
        clock=SimpleNamespace(time=jnp.asarray(time)))


# ================================================================
#  Capability menu defaults (the base is concrete)
# ================================================================
def test_base_is_concrete_with_empty_menu():
    module = Module()
    assert module.field_declarations == ()
    assert module.field_references == ()
    assert module.parameter_declarations == ()
    assert module.parameter_references == ()
    assert dict(module.dispatch) == {}
    assert module.extra_halo is None
    assert module.state_type is None
    assert module.stages == ()
    assert module.tendency_terms() == ()
    assert module.collected_stages() == ()


def test_toy_declaration_surfaces():
    module = ToyStratification()
    names = [decl.name for decl in module.field_declarations]
    assert names == ["b", "n2"]
    assert module.field_references[0].name == "w"
    assert module.parameter_declarations[0].attr == "n2_input"
    assert module.parameter_references[0].default == 0.0


# ================================================================
#  Term collection
# ================================================================
def test_term_collection_definition_order():
    module = ToyStratification()
    terms = module.tendency_terms()
    assert [t.name for t in terms] == ["buoyancy_force", "restoring"]
    assert terms[0].advances == ("w",)
    assert all(t.linear for t in terms)


def test_term_collection_is_deterministic():
    module = ToyStratification()
    assert module.tendency_terms() == module.tendency_terms()
    table = DuckTable()
    module.bind(table)
    # post-bind re-collection returns the same tuple (variants)
    assert module.tendency_terms() == module.tendency_terms()


def test_terms_inherited_first_in_subclass():
    class ExtendedToy(ToyStratification):
        @term(advances=("b",))
        def extra_forcing(self, state, ctx):  # noqa: ARG002
            return {"b": state["b"]}

    names = [t.name for t in ExtendedToy().tendency_terms()]
    assert names == ["buoyancy_force", "restoring", "extra_forcing"]


def test_term_fns_stored_unbound_and_callable_through_module():
    module = ToyStratification()
    terms = module.tendency_terms()
    for t in terms:
        assert getattr(t.fn, "__self__", None) is None
    state = {"b": jnp.asarray(2.0), "w": jnp.asarray(0.5),
             "n2": jnp.asarray(1e-5)}
    out = terms[0].fn(module, state, make_ctx())
    assert float(out["w"]) == 2.0
    out = terms[1].fn(module, state, make_ctx())
    assert float(out["b"]) == -0.5 * 1e-5


# ================================================================
#  Stage collection
# ================================================================
def test_stage_collection_kinds_and_order():
    module = ToyStratification()
    stages = module.collected_stages()
    assert [s.kind for s in stages] == [
        StageKind.SELF_UPDATE, StageKind.CONSTRAINT]
    assert [s.name for s in stages] == ["self_update", "clamp_b"]


def test_string_fn_resolved_to_unbound_method():
    module = ToyStratification()
    constraint = module.collected_stages()[1]
    assert constraint.fn is ToyStratification.clamp_b
    assert getattr(constraint.fn, "__self__", None) is None
    out = constraint.fn(module, {"b": jnp.asarray(1.5)}, make_ctx())
    assert float(out["b"]) == 1.5


def test_unknown_string_fn_raises():
    class Broken(Module):
        stages = (Stage(kind=StageKind.DIAGNOSE, fn="missing"),)

    with pytest.raises(AttributeError, match="missing"):
        Broken().collected_stages()


def test_bare_self_update_collected():
    class BareUpdate(Module):
        def self_update(self, state, ctx):  # noqa: ARG002
            return {}

    stages = BareUpdate().collected_stages()
    assert len(stages) == 1
    assert stages[0].kind is StageKind.SELF_UPDATE
    assert stages[0].reads == ()
    assert stages[0].fn is BareUpdate.self_update


def test_stage_collection_is_deterministic():
    module = ToyStratification()
    assert module.collected_stages() == module.collected_stages()


def test_self_update_hook_unbound_execution():
    ramp = Ramp(0.0, 1.0, period=10.0)
    module = ToyStratification(n2=ramp)
    stage = module.collected_stages()[0]
    out = stage.fn(module, {}, make_ctx(time=5.0))
    assert float(out["n2"]) == 0.5


# ================================================================
#  bind: freeze, once, immutability
# ================================================================
def test_bind_freezes_role_selection_to_tuples():
    module = ToyStratification()
    module.bind(DuckTable())
    assert module._advected == ("b",)
    assert module._amp0 == 0.0


def test_bind_runs_once():
    module = ToyStratification()
    module.bind(DuckTable())
    with pytest.raises(ImmutableParameterError, match="once"):
        module.bind(DuckTable())


def test_setattr_before_bind_is_allowed():
    module = ToyStratification()
    module.kh = jnp.asarray(3.0)
    assert float(module.kh) == 3.0


def test_setattr_after_bind_raises():
    module = ToyStratification()
    module.bind(DuckTable())
    with pytest.raises(ImmutableParameterError,
                       match="update_parameters"):
        module.kh = jnp.asarray(3.0)
    with pytest.raises(ImmutableParameterError):
        module.brand_new = 1


def test_delattr_after_bind_raises():
    module = ToyStratification()
    module.bind(DuckTable())
    with pytest.raises(ImmutableParameterError):
        del module.kh


def test_failed_bind_leaves_module_unbound():
    module = ToyStratification()
    with pytest.raises(RuntimeError, match="table exploded"):
        module.bind(FailingTable())
    module.kh = jnp.asarray(2.0)          # still mutable
    module.bind(DuckTable())              # retry succeeds
    assert module._advected == ("b",)


def test_override_with_super_call_freezes_once():
    class ExtendedBind(ToyStratification):
        def bind(self, table):
            super().bind(table)
            self._extra = ("precomputed",)

    module = ExtendedBind()
    module.bind(DuckTable())
    assert module._advected == ("b",)
    assert module._extra == ("precomputed",)
    with pytest.raises(ImmutableParameterError):
        module.bind(DuckTable())
    with pytest.raises(ImmutableParameterError):
        module._extra = ()


def test_base_module_bind_is_noop_but_freezes():
    module = Module()
    module.bind(DuckTable())
    with pytest.raises(ImmutableParameterError):
        module.anything = 1


# ================================================================
#  The bind-time parameter read gate
# ================================================================
def test_bare_ramp_read_raises():
    view = BindParameterView(
        {"forcing.amp": Ramp(0.0, 1.0, period=10.0), "plain": 4.0})
    with pytest.raises(TimeDependentParameterError,
                       match="at_time"):
        view["forcing.amp"]
    assert view["plain"] == 4.0


def test_at_time_zero_is_the_sanctioned_spelling():
    view = BindParameterView(
        {"forcing.amp": Ramp(1.0, 3.0, period=2.0), "plain": 4.0})
    values = view.at_time(1.0)
    assert float(values["forcing.amp"]) == 2.0
    assert values["plain"] == 4.0


def test_membership_and_iteration_do_not_evaluate():
    view = BindParameterView(
        {"forcing.amp": Ramp(0.0, 1.0, period=10.0)})
    assert "forcing.amp" in view
    assert sorted(view) == ["forcing.amp"]
    assert len(view) == 1
    with pytest.raises(KeyError):
        view["unknown.name"]


def test_bind_body_bare_read_raises_for_ramp():
    class BareReader(Module):
        parameter_references = (
            ParameterReference("forcing.amp", default=0.0),)

        def bind(self, table):
            self._amp = table.parameters["forcing.amp"]

    module = BareReader()
    module.bind(DuckTable({"forcing.amp": 0.5}))
    assert module._amp == 0.5

    ramped = BareReader()
    with pytest.raises(TimeDependentParameterError):
        ramped.bind(DuckTable(
            {"forcing.amp": Ramp(0.0, 1.0, period=10.0)}))
    ramped._probe = 1                     # failed bind: not frozen


def test_toy_binds_with_ramp_via_at_time():
    module = ToyStratification()
    module.bind(DuckTable(
        {"forcing.amp": Ramp(0.25, 1.0, period=10.0)}))
    assert module._amp0 == 0.25


# ================================================================
#  jaxify: carry round-trip, traces, recompiles
# ================================================================
def test_flatten_unflatten_round_trip_preserves_leaves_and_statics():
    module = ToyStratification(
        n2=Ramp(1e-6, 1e-5, period=3600.0), kh=2.0)
    module.bind(DuckTable())
    leaves, treedef = jax.tree_util.tree_flatten(module)
    # kh + the four Ramp leaves (v0, v1, t0, period)
    assert len(leaves) == 5
    clone = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(clone) is ToyStratification
    assert float(clone.kh) == 2.0
    assert isinstance(clone.n2_input, Ramp)
    assert clone._advected == ("b",)      # statics survive
    assert clone._amp0 == 0.0
    assert clone == module                # structural equality
    assert jax.tree_util.tree_structure(clone) == treedef
    # unflattened copies are transient in-trace objects: the
    # host-side freeze does not follow them through the carry
    clone.kh = jnp.asarray(3.0)


def test_equal_config_bound_instances_share_one_trace(
        compile_counter):
    m1 = ToyStratification(n2=1e-5, kh=2.0)
    m2 = ToyStratification(n2=1e-5, kh=2.0)
    m1.bind(DuckTable())
    m2.bind(DuckTable())

    @jax.jit
    def probe(module, x):
        return module.kh * x + resolve_at(module.n2_input, 0.5)

    x = jnp.arange(4.0)
    assert float(probe(m1, x)[1]) == pytest.approx(2.0 + 1e-5)
    compile_counter.reset()
    probe(m2, x)
    assert compile_counter.count == 0


def test_leaf_sweep_zero_recompiles(compile_counter):
    def bound(kh, v1):
        module = ToyStratification(
            n2=Ramp(0.0, v1, period=3600.0), kh=kh)
        module.bind(DuckTable())
        return module

    @jax.jit
    def probe(module, x):
        return module.kh * x + resolve_at(module.n2_input, 1800.0)

    x = jnp.arange(4.0)
    probe(bound(1.0, 1e-5), x)
    compile_counter.reset()
    for kh, v1 in ((2.0, 1e-5), (3.0, 2e-5), (0.5, 5e-4)):
        probe(bound(kh, v1), x)
    assert compile_counter.count == 0


def test_eq_ignored_attrs_keep_observers_out_of_equality(
        compile_counter):
    # host-side observers need BOTH mechanisms under the landed
    # jaxify: `_eq_ignored_attrs` exempts them from the direct
    # structural comparison (`m1 == m2`), while the `annotation=`
    # aux category exempts them from the treedef/jit-cache equality
    # (they still ride the aux and survive flatten/unflatten)
    m1 = Observed(kh=1.0)
    m2 = Observed(kh=1.0)
    m1._writes = 7                        # host-side observer state
    m1.bind(DuckTable())
    m2.bind(DuckTable())
    assert m1 == m2
    assert (jax.tree_util.tree_structure(m1)
            == jax.tree_util.tree_structure(m2))
    leaves, treedef = jax.tree_util.tree_flatten(m1)
    assert jax.tree_util.tree_unflatten(treedef, leaves)._writes == 7

    @jax.jit
    def probe(module):
        return module.kh * 2.0

    probe(m1)
    compile_counter.reset()
    probe(m2)
    assert compile_counter.count == 0


@partial(jaxify, dynamic=("kh",), annotation=("_writes",))
class Observed(Module):

    """Module with a host-side observer attribute (eq-ignored)."""

    _eq_ignored_attrs = frozenset({"_writes"})

    def __init__(self, kh=1.0):
        self.kh = jnp.asarray(kh)
        self._writes = 0


# ================================================================
#  Named-ScalarField dynamic leaves (annotation-exempt metadata)
# ================================================================
@partial(jaxify, dynamic=("profile",))
class FieldCarrier(Module):

    """Module carrying a named ScalarField as a dynamic leaf."""

    def __init__(self, profile):
        self.profile = profile


def test_named_scalarfield_leaf_is_treedef_stable(compile_counter):
    # the wave-1 annotation-exempt metadata equality makes named
    # ScalarField dynamic leaves legal: metadata differences change
    # neither the treedef nor the jit-cache entry
    grid = Grid((IntervalMesh(8, (0.0, 1.0), name="x"),))
    fa = grid.create_field(data=jnp.arange(8.0), name="n2")
    fb = grid.create_field(data=jnp.arange(8.0) + 1.0, name="other")
    ma, mb = FieldCarrier(fa), FieldCarrier(fb)
    assert (jax.tree_util.tree_structure(ma)
            == jax.tree_util.tree_structure(mb))

    @jax.jit
    def probe(module):
        return module.profile.data.sum()

    assert float(probe(ma)) == pytest.approx(28.0)
    compile_counter.reset()
    assert float(probe(mb)) == pytest.approx(36.0)
    assert compile_counter.count == 0


# ================================================================
#  extra_halo and state_type
# ================================================================
def test_extra_halo_storage():
    class Halowed(Module):
        def __init__(self):
            self.extra_halo = HaloSpec({"x": 2})

    assert Module().extra_halo is None
    module = Halowed()
    assert module.extra_halo["x"] == 2
    leaves, treedef = jax.tree_util.tree_flatten(module)
    clone = jax.tree_util.tree_unflatten(treedef, leaves)
    assert clone.extra_halo["x"] == 2     # rides the static aux


def test_state_type_hook_stays_out_of_the_instance_dict():
    class DummyState:
        pass

    class CoreModule(Module):
        state_type = DummyState

    assert Module.state_type is None
    m1, m2 = CoreModule(), CoreModule()
    assert m1.state_type is DummyState
    assert "state_type" not in m1.__dict__
    assert (jax.tree_util.tree_structure(m1)
            == jax.tree_util.tree_structure(m2))
