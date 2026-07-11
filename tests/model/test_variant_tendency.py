"""Tests for Model.tendency / Model.variant / fr.linearize (2.8).

A toy dynamical core with two linear terms (coriolis, stratification)
and one nonlinear term (advection) exercises the read-only composed
tendency, the fr.terms filter, the variant re-assembly (shared State
treedef), and fr.linearize.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import dtype_real, jaxify
from fridom.model import term_predicates as terms
from fridom.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.model.errors import AssemblyError
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration
from fridom.model.term_predicates import linearize
from fridom.model.terms import term
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

N = 8
DT = 1e-3
F0 = 2.0
N2 = 0.5


@jaxify
class Core(Module):

    """Two-field core: linear coriolis + stratification, nonlinear adv."""

    f0 = F0
    n2 = N2

    field_declarations = (
        FieldDeclaration("u", space=Collocated()),
        FieldDeclaration("v", space=Collocated()),
    )

    @term(name="coriolis", advances=("u", "v"), linear=True)
    def coriolis(self, state, _ctx):
        return {"u": state["v"] * self.f0,
                "v": state["u"] * (-self.f0)}

    @term(name="stratification", advances=("u",), linear=True)
    def stratification(self, state, _ctx):
        return {"u": state["u"] * self.n2}

    @term(name="advection", advances=("u",))
    def advection(self, state, _ctx):
        return {"u": state["u"] * state["u"]}


@partial(jaxify, dynamic=("value",))
class Provider(Module):

    """A pure scalar parameter provider (unused by any term)."""

    def __init__(self, value=1.0):
        self.value = jnp.asarray(value, dtype=dtype_real())

    parameter_declarations = (
        ParameterDeclaration("toy.value", attr="value", units="1"),)


def make_model(with_provider=False, **kwargs):
    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))
    modules = (Core(), Provider()) if with_provider else (Core(),)
    return Model(grid=grid, modules=modules,
                 time_stepper=AdamBashforth(DT, order=2), **kwargs)


def _ic(model):
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    model.set_fields(u=np.sin(2 * np.pi * x) + 0.5,
                     v=np.cos(2 * np.pi * x))
    return model.state


# ================================================================
#  model.tendency correctness
# ================================================================
def test_tendency_matches_the_composed_terms():
    model = make_model()
    state = _ic(model)
    td = model.tendency(state)
    u, v = state["u"].data, state["v"].data
    expected_u = v * F0 + u * N2 + u * u
    expected_v = -u * F0
    assert jnp.allclose(td["u"].data, expected_u)
    assert jnp.allclose(td["v"].data, expected_v)


def test_tendency_component_names_are_prognostic():
    model = make_model()
    state = _ic(model)
    assert model.tendency(state).component_names == ("u", "v")


def test_tendency_linear_filter_drops_advection():
    model = make_model()
    state = _ic(model)
    td = model.tendency(state, filter=terms.linear)
    u, v = state["u"].data, state["v"].data
    assert jnp.allclose(td["u"].data, v * F0 + u * N2)  # no u*u
    assert jnp.allclose(td["v"].data, -u * F0)


def test_tendency_named_filter_single_term():
    model = make_model()
    state = _ic(model)
    td = model.tendency(state, filter=terms.named("Core/coriolis"))
    u, v = state["u"].data, state["v"].data
    assert jnp.allclose(td["u"].data, v * F0)
    assert jnp.allclose(td["v"].data, -u * F0)


def test_tendency_unknown_named_key_errors():
    model = make_model()
    state = _ic(model)
    with pytest.raises(AssemblyError, match="unknown terms"):
        model.tendency(state, filter=terms.named("Core/nope"))


def test_tendency_never_advances_the_carry():
    model = make_model()
    state = _ic(model)
    before = model.clock.it
    model.tendency(state)
    assert model.clock.it == before
    assert jnp.array_equal(model.state["u"].data, state["u"].data)


def test_tendency_reuses_one_compiled_entry():
    model = make_model()
    state = _ic(model)
    model.tendency(state)
    model.tendency(state)  # same filter/constraints -> cache hit
    assert len(model._tendency_cache) == 1


def test_tendency_at_explicit_time():
    model = make_model()
    state = _ic(model)
    td0 = model.tendency(state, t=0.0)
    td1 = model.tendency(state, t=1.0)
    # no time-dependent params in the toy: identical results
    assert jnp.allclose(td0["u"].data, td1["u"].data)


# ================================================================
#  model.variant
# ================================================================
def test_variant_shares_state_treedef():
    model = make_model()
    _ic(model)
    variant = model.variant(term_filter=terms.linear)
    assert (jax.tree_util.tree_structure(model.state)
            == jax.tree_util.tree_structure(variant.state))


def test_variant_filters_terms_only():
    model = make_model()
    variant = model.variant(term_filter=terms.linear)
    keys = {e.key for e in
            variant._artifacts.schedule.kind_entries(None)}
    assert keys == {"Core/coriolis", "Core/stratification"}


def test_variant_names_default():
    model = make_model(name="toy")
    variant = model.variant(term_filter=terms.linear)
    assert variant.name == "toy/variant"


def test_variant_leaves_the_parent_rebindable():
    # regression: the child assembly binds (and freezes) the module
    # instances it is handed; variant must pass fresh clones so the
    # parent's carry modules stay unbound — any number of variants
    # (fr.linearize) can then be taken from one parent, including
    # a variant of a variant
    model = make_model()
    first = linearize(model)
    second = linearize(model)
    third = second.variant(term_filter=terms.linear)
    for derived in (first, second, third):
        keys = {e.key for e in
                derived._artifacts.schedule.kind_entries(None)}
        assert keys == {"Core/coriolis", "Core/stratification"}


def test_variant_updates_change_dt_sign():
    model = make_model()
    variant = model.variant(
        updates={"stepper.dt": -DT})
    assert float(variant._stepper.dt) == pytest.approx(-DT)


def test_variant_updates_module_parameter():
    model = make_model(with_provider=True)
    variant = model.variant(updates={"toy.value": 7.0})
    assert float(variant.parameters["toy.value"]) == pytest.approx(7.0)


def test_variant_updates_unknown_parameter_errors():
    model = make_model()
    with pytest.raises(AssemblyError, match=r"not provided|not bound"):
        model.variant(updates={"nope.param": 1.0})


def test_tendency_plain_callable_filter():
    # a legacy two-argument callable (no fr.terms metadata)
    model = make_model()
    state = _ic(model)
    td = model.tendency(state, filter=lambda key, _term: "coriolis"
                        in key)
    u, v = state["u"].data, state["v"].data
    assert jnp.allclose(td["u"].data, v * F0)
    assert jnp.allclose(td["v"].data, -u * F0)


# ================================================================
#  model.constrain (the H1 public projector matvec)
# ================================================================
def test_constrain_without_constraint_stages_is_identity():
    # the toy carries no CONSTRAINT stage: the matvec returns the
    # input's PROGNOSTIC subset unchanged (and never the carry)
    model = make_model()
    state = _ic(model)
    out = model.constrain(state)
    assert out.component_names == ("u", "v")
    assert jnp.array_equal(out["u"].data, state["u"].data)
    assert jnp.array_equal(out["v"].data, state["v"].data)


def test_constrain_reuses_one_compiled_entry():
    model = make_model()
    state = _ic(model)
    model.constrain(state)
    model.constrain(state, t=0.0)  # same executable -> cache hit
    assert ("constrain",) in model._tendency_cache
    assert len(model._tendency_cache) == 1


def test_constrain_needs_prognostic_fields():
    class AuxOnly(Module):
        field_declarations = (
            FieldDeclaration("q", space=Collocated(),
                             lifecycle=Lifecycle.AUXILIARY,
                             default=lambda x: 0.0 * x),
        )

    grid = Grid((IntervalMesh(N, (0.0, 1.0), periodic=True,
                              name="x"),))
    model = Model(grid=grid, modules=(AuxOnly(),),
                  time_stepper=AdamBashforth(DT, order=2))
    with pytest.raises(NotImplementedError, match="PROGNOSTIC"):
        model.constrain(model.state)


# ================================================================
#  fr.linearize
# ================================================================
def test_linearize_keeps_only_linear_terms():
    model = make_model(name="toy")
    lin = linearize(model)
    keys = {e.key for e in
            lin._artifacts.schedule.kind_entries(None)}
    assert keys == {"Core/coriolis", "Core/stratification"}
    assert lin.name == "toy/linear"


def test_linearize_tendency_equals_filtered_tendency():
    model = make_model()
    state = _ic(model)
    lin = linearize(model)
    lin.set_fields(u=state["u"].data, v=state["v"].data)
    from_variant = lin.tendency(lin.state)
    from_filter = model.tendency(state, filter=terms.linear)
    assert jnp.allclose(from_variant["u"].data, from_filter["u"].data)
    assert jnp.allclose(from_variant["v"].data, from_filter["v"].data)
