"""Tests for the wave-5 schedule seams (treatment partition, etc.).

Covers ``TendencySums.__getitem__`` by ``Treatment`` (the ordered
static representation — the enum members are not sortable), the
partition (EXPLICIT sums vs the merged IMPLICIT operator set),
``merged_with`` collapsing a mergeable family to ONE bound operator
with the kappa summed exactly, ``advance_stages`` running the ADVANCE
group as a sequential Gauss-Seidel walk, and the dry-run params fix
(a bound ``ctx.params`` read passes; an unbound name surfaces
attributed through the ``TermEvaluationError`` chain).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.grid import Grid
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.model.assembly import Params
from fridom.framework2.model.composer import TendencyComposer
from fridom.framework2.model.context import StepContext
from fridom.framework2.model.declarations import Lifecycle
from fridom.framework2.model.errors import TermEvaluationError
from fridom.framework2.model.implicit import VerticalDiffusion
from fridom.framework2.model.schedule import (
    BoundImplicitOperator,
    TendencySums,
)
from fridom.framework2.model.stages import Stage, StageKind
from fridom.framework2.model.terms import (
    TERM_ATTRIBUTE,
    TendencyTerm,
    Treatment,
    term,
)


# ================================================================
#  Fakes
# ================================================================
class Record(NamedTuple):
    name: str
    space: object
    lifecycle: Lifecycle
    owner: int


class FakeFieldTable:
    def __init__(self, grid, records):
        self.grid = grid
        self._records = tuple(records)

    def __iter__(self):
        return iter(self._records)


class ExplicitCore:
    @term(name="decay")
    def du(self, state, _ctx):
        return {"u": state["u"] * (-0.7)}


class Mixer:

    """Owner of the implicit vertical-diffusion terms."""


class Advancer:

    """An ADVANCE stage writing a prognostic subset (identity add)."""

    def bump(self, state, _ctx):
        return {"u": state["u"] + 1.0}


class FakeStepper:
    supported_treatments = frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})


def term_of(method):
    return getattr(method, TERM_ATTRIBUTE)


def kappa_const(value):
    def kappa(_module, _state, _ctx, _name):
        return value
    return kappa


@pytest.fixture
def mz():
    return IntervalMesh(8, (0.0, 1.0), name="z")


@pytest.fixture
def grid(mz):
    return Grid((mz,))


@pytest.fixture
def field_table(grid, mz):
    return FakeFieldTable(grid, (
        Record("u", mz.center, Lifecycle.PROGNOSTIC, 0),
        Record("b", mz.center, Lifecycle.PROGNOSTIC, 1),
    ))


def make_state(field_table, **data):
    fields = {}
    for record in field_table:
        value = data.get(record.name, np.zeros(record.space.shape))
        fields[record.name] = field_table.grid.create_field(
            record.space, data=jnp.asarray(value), name=record.name)
    return VectorField(fields)


def make_ctx(sums=None):
    return StepContext(params={}, clock=jnp.asarray(0.0),
                       dt=jnp.asarray(1.0), stage_dt=jnp.asarray(1.0),
                       tendency_sums=sums)


# ================================================================
#  TendencySums.__getitem__ by Treatment
# ================================================================
def test_getitem_explicit_returns_the_explicit_sum(field_table):
    sums = TendencySums(explicit=make_state(field_table))
    assert sums[Treatment.EXPLICIT] is sums.explicit


def test_getitem_implicit_absent_raises_keyerror(field_table):
    sums = TendencySums(explicit=make_state(field_table))
    with pytest.raises(KeyError, match="no IMPLICIT"):
        _ = sums[Treatment.IMPLICIT]


def test_getitem_implicit_present_returns_the_sum(field_table):
    explicit = make_state(field_table)
    implicit = make_state(field_table)
    sums = TendencySums(explicit=explicit, implicit=implicit)
    assert sums[Treatment.IMPLICIT] is implicit
    assert sums[Treatment.EXPLICIT] is explicit


def test_getitem_uses_the_ordered_name_representation(field_table):
    # the enum members are not sortable; the partition keys on the
    # member NAME (an ordered static representation, W2)
    sums = TendencySums(explicit=make_state(field_table))
    with pytest.raises(TypeError):
        sorted([Treatment.EXPLICIT, Treatment.IMPLICIT])
    assert sums[Treatment.EXPLICIT] is sums.explicit


def test_tendency_sums_round_trip_with_implicit(field_table):
    sums = TendencySums(explicit=make_state(field_table),
                        implicit=make_state(field_table))
    leaves, treedef = jax.tree_util.tree_flatten(sums)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.explicit is not None
    assert rebuilt.implicit is not None


# ================================================================
#  The partition — EXPLICIT sums vs the IMPLICIT operator set
# ================================================================
def implicit_composer(field_table, stepper, *, ops):
    terms = [(0, term_of(ExplicitCore.du))]
    for index, op in enumerate(ops):
        terms.append((1, TendencyTerm(
            name=f"mix{index}", treatment=Treatment.IMPLICIT,
            implicit=op)))
    return TendencyComposer(
        field_table=field_table,
        modules=(ExplicitCore(), Mixer()),
        terms=tuple(terms), stages=(), time_stepper=stepper,
        binding_table=None)


def test_tendency_holds_only_the_explicit_partition(field_table):
    op = VerticalDiffusion("z", ("b",), kappa_const(2.0))
    schedule = implicit_composer(
        field_table, FakeStepper(), ops=(op,)).schedule
    bound = schedule.bind((ExplicitCore(), Mixer()))
    state = make_state(field_table, u=np.ones(8), b=np.ones(8))
    sums = bound.tendency(state, make_ctx())
    # u carries its explicit tendency; b (implicit only) stays zero in
    # the explicit sum (the solve carries it, not the S2 accumulation)
    assert np.allclose(sums.explicit["u"].data, -0.7)
    assert np.allclose(sums.explicit["b"].data, 0.0)


def test_implicit_returns_bound_operators(field_table):
    op = VerticalDiffusion("z", ("b",), kappa_const(2.0))
    schedule = implicit_composer(
        field_table, FakeStepper(), ops=(op,)).schedule
    bound = schedule.bind((ExplicitCore(), Mixer()))
    operators = bound.implicit
    assert len(operators) == 1
    assert isinstance(operators[0], BoundImplicitOperator)
    assert operators[0].fields == ("b",)


# ================================================================
#  merged_with — ONE operator per mergeable family, kappa summed
# ================================================================
def test_mergeable_family_collapses_to_one_bound_operator(
        field_table, grid, mz):
    # two vertical-diffusion closures on the SAME field/axis: one
    # merged operator, kappa summed exactly (2 + 3.5 = 5.5)
    op1 = VerticalDiffusion("z", ("b",), kappa_const(2.0))
    op2 = VerticalDiffusion("z", ("b",), kappa_const(3.5))
    composer = implicit_composer(
        field_table, FakeStepper(), ops=(op1, op2))
    # one merge group -> one merged (op, slot) pair
    assert len(composer.schedule.implicit_merged) == 1
    bound = composer.schedule.bind((ExplicitCore(), Mixer()))
    operators = bound.implicit
    assert len(operators) == 1

    # the merged solve uses the summed coefficient (5.5)
    rhs0 = np.linspace(0.0, 1.0, 8)
    rhs = {"b": grid.create_field(
        mz.center, data=jnp.asarray(rhs0), name="b")}
    merged = operators[0].solve(rhs, jnp.asarray(0.1), make_ctx())["b"]
    reference = VerticalDiffusion(
        "z", ("b",), kappa_const(5.5)).solve(
        None, rhs, jnp.asarray(0.1), None)["b"]
    assert np.allclose(np.asarray(merged.data),
                       np.asarray(reference.data), atol=1e-13)


# ================================================================
#  advance_stages — the sequential ADVANCE (Gauss-Seidel) walk
# ================================================================
def test_advance_stages_runs_the_advance_group(field_table):
    stepper = FakeStepper()
    composer = TendencyComposer(
        field_table=field_table,
        modules=(ExplicitCore(), Advancer()),
        terms=((0, term_of(ExplicitCore.du)),),
        stages=((1, Stage(kind=StageKind.ADVANCE, fn=Advancer.bump,
                          name="bump", advances=("u",))),),
        time_stepper=stepper, binding_table=None)
    bound = composer.schedule.bind((ExplicitCore(), Advancer()))
    state = make_state(field_table, u=np.full(8, 2.0))
    out = bound.advance_stages(state, make_ctx())
    assert np.allclose(out["u"].data, 3.0)  # 2 + 1 (the ADVANCE add)


# ================================================================
#  The dry-run params fix
# ================================================================
class ReadsDt:
    @term(name="reads_dt")
    def du(self, state, ctx):
        return {"u": state["u"] * float(ctx.params["stepper.dt"])}


class ReadsMissing:
    @term(name="reads_missing")
    def du(self, state, ctx):
        return {"u": state["u"] * float(ctx.params["nope.missing"])}


def single_prognostic_table(grid, mz):
    return FakeFieldTable(grid, (
        Record("u", mz.center, Lifecycle.PROGNOSTIC, 0),))


def test_dry_run_resolves_a_bound_param(grid, mz):
    table = single_prognostic_table(grid, mz)
    composer = TendencyComposer(
        field_table=table, modules=(ReadsDt(),),
        terms=((0, term_of(ReadsDt.du)),), stages=(),
        time_stepper=FakeStepper(), binding_table=None)
    # a term reading ctx.params["stepper.dt"] no longer dies on {}
    composer.dry_run(params=Params({"stepper.dt": jnp.asarray(60.0)}))


def test_dry_run_default_params_is_empty(grid, mz):
    # omitting params keeps the pre-fix behaviour (empty mapping): a
    # param read would raise, but a param-free term validates fine
    table = single_prognostic_table(grid, mz)
    composer = TendencyComposer(
        field_table=table, modules=(ExplicitCore(),),
        terms=((0, term_of(ExplicitCore.du)),), stages=(),
        time_stepper=FakeStepper(), binding_table=None)
    composer.dry_run()  # no params, no param reads -> passes


def test_dry_run_unbound_param_surfaces_attributed(grid, mz):
    table = single_prognostic_table(grid, mz)
    composer = TendencyComposer(
        field_table=table, modules=(ReadsMissing(),),
        terms=((0, term_of(ReadsMissing.du)),), stages=(),
        time_stepper=FakeStepper(), binding_table=None)
    with pytest.raises(TermEvaluationError) as excinfo:
        composer.dry_run(
            params=Params({"stepper.dt": jnp.asarray(60.0)}))
    # attributed to the term, and the missing name surfaces in the
    # chain (NOTE: the chained cause is a KeyError rather than the
    # hinted MissingParameterError because StepContext coerces params
    # to a plain dict in context.py — outside this wave's scope)
    assert "reads_missing" in str(excinfo.value)
    assert "nope.missing" in str(excinfo.value)
