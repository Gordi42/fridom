"""Composer term-envelope tests (prefix shard of test_composer).

The C1 mechanism: a ``TendencyEnvelope`` module in the tuple makes
the composer wrap every predicate-matched term hook in the
``ctx.params["ramping.envelope"]`` multiply — delivered through the
per-substage context, never host-captured. Covers the wrap
semantics (matched scaled, unmatched untouched), the ``enveloped``
schedule flag, and the four taught refusals (two envelope modules,
an IMPLICIT match, a linear match, an empty match — error without a
``term_filter``, warning under one).

Self-contained per the oversized-module shard convention: the small
composer fakes are duplicated from ``test_composer``.
"""
from typing import NamedTuple

import jax.numpy as jnp
import pytest

from fridom.model import term_predicates as terms
from fridom.model.composer import TendencyComposer
from fridom.model.context import StepContext
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError
from fridom.model.modules.ramping import TendencyEnvelope
from fridom.model.params import RAMPING_ENVELOPE
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

RHO = 0.5


# ================================================================
#  Fakes (duck-typed composer inputs, as in test_composer)
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


class Rotation:

    """Fake linear-term module (slot 0 by convention)."""

    def du(self, state, _ctx):
        return {"u": state["b"] * 0.5}


class Advection:

    """Fake nonlinear-term module (slot 1 by convention)."""

    def db(self, state, _ctx):
        return {"b": state["u"] * 0.25 + 1.0}


class FakeImplicitOp:
    def __init__(self, fields):
        self.fields = tuple(fields)

    def apply(self, _module, state, _ctx):
        return {name: state[name] * 0.1 for name in self.fields}

    def solve(self, _module, rhs, _dt_gamma, _ctx):
        return rhs

    def merge_key(self):
        return None

    def merged_with(self, _other):
        return self


# ================================================================
#  Fixtures / builders
# ================================================================
@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")


@pytest.fixture
def grid(mx):
    return Grid((mx,))


@pytest.fixture
def cc(mx):
    return mx.center


@pytest.fixture
def field_table(grid, cc):
    return FakeFieldTable(grid, (
        Record("u", cc, Lifecycle.PROGNOSTIC, 0),
        Record("b", cc, Lifecycle.PROGNOSTIC, 1),
    ))


def default_terms():
    return (
        (0, TendencyTerm(name="du", fn=Rotation.du, linear=True)),
        (1, TendencyTerm(name="db", fn=Advection.db)),
    )


def make_composer(field_table, *, modules, terms=None,
                  term_filter=None):
    if terms is None:
        terms = default_terms()
    return TendencyComposer(
        field_table=field_table, modules=modules, terms=terms,
        stages=(), time_stepper=None, binding_table=None,
        term_filter=term_filter)


def make_ctx(rho=RHO):
    return StepContext(
        params={RAMPING_ENVELOPE: jnp.asarray(rho)},
        clock=jnp.asarray(0.0), dt=jnp.asarray(1.0),
        stage_dt=jnp.asarray(1.0))


def seeded_state(field_table, cc):
    grid = field_table.grid
    return VectorField({
        "u": grid.create_field(cc, data=jnp.linspace(0.0, 1.0, 8),
                               name="u"),
        "b": grid.create_field(cc, data=jnp.linspace(1.0, 2.0, 8),
                               name="b"),
    })


NONLINEAR = ~terms.linear & terms.explicit


# ================================================================
#  Wrap semantics: matched scaled by rho, unmatched untouched
# ================================================================
def test_matched_term_is_scaled_by_the_ctx_rho(field_table, cc):
    envelope = TendencyEnvelope(terms=NONLINEAR)
    composer = make_composer(
        field_table, modules=(Rotation(), Advection(), envelope))
    body = composer.compose()
    state = seeded_state(field_table, cc)
    modules = (Rotation(), Advection(), envelope)
    _, sums = body(state, modules, make_ctx())
    # the linear rotation term is untouched by the envelope
    assert jnp.allclose(sums.explicit["u"].data,
                        state["b"].data * 0.5)
    # the nonlinear advection contribution is scaled by rho
    assert jnp.allclose(sums.explicit["b"].data,
                        RHO * (state["u"].data * 0.25 + 1.0))


def test_rho_is_read_live_from_the_context(field_table, cc):
    # the same composed body sees a different rho per ctx: nothing
    # was host-captured at composition (the D2 rule)
    envelope = TendencyEnvelope(terms=NONLINEAR)
    modules = (Rotation(), Advection(), envelope)
    body = make_composer(field_table, modules=modules).compose()
    state = seeded_state(field_table, cc)
    _, at_zero = body(state, modules, make_ctx(rho=0.0))
    _, at_one = body(state, modules, make_ctx(rho=1.0))
    assert jnp.allclose(at_zero.explicit["b"].data, 0.0)
    assert jnp.allclose(at_one.explicit["b"].data,
                        state["u"].data * 0.25 + 1.0)


def test_schedule_marks_only_matched_entries_enveloped(field_table):
    envelope = TendencyEnvelope(terms=NONLINEAR)
    composer = make_composer(
        field_table, modules=(Rotation(), Advection(), envelope))
    flags = {entry.key: entry.enveloped
             for entry in composer.schedule.kind_entries(None)}
    assert flags == {"Rotation/du": False, "Advection/db": True}


def test_no_envelope_module_leaves_entries_plain(field_table):
    composer = make_composer(
        field_table, modules=(Rotation(), Advection()))
    assert all(not entry.enveloped
               for entry in composer.schedule.kind_entries(None))


def test_dry_run_passes_with_the_envelope_param(field_table):
    envelope = TendencyEnvelope(terms=NONLINEAR)
    composer = make_composer(
        field_table, modules=(Rotation(), Advection(), envelope))
    composer.dry_run(params={RAMPING_ENVELOPE: jnp.asarray(0.0)})


def test_derived_explicit_hook_is_enveloped(field_table, cc):
    # an EXPLICIT term with fn=None derives its hook from
    # implicit.apply; the envelope wraps that derived hook
    op = FakeImplicitOp(fields=("b",))
    term_rows = (
        (0, TendencyTerm(name="du", fn=Rotation.du, linear=True)),
        (1, TendencyTerm(name="mix", implicit=op)),
    )
    envelope = TendencyEnvelope(terms=NONLINEAR)
    modules = (Rotation(), Advection(), envelope)
    composer = make_composer(field_table, modules=modules,
                             terms=term_rows)
    body = composer.compose()
    state = seeded_state(field_table, cc)
    _, sums = body(state, modules, make_ctx())
    assert jnp.allclose(sums.explicit["b"].data,
                        RHO * state["b"].data * 0.1)


# ================================================================
#  Taught refusals
# ================================================================
def test_two_envelope_modules_refused(field_table):
    with pytest.raises(AssemblyError, match="one envelope module"):
        make_composer(
            field_table,
            modules=(Rotation(), Advection(),
                     TendencyEnvelope(terms=NONLINEAR),
                     TendencyEnvelope(terms=terms.explicit)))


def test_implicit_match_refused(field_table):
    op = FakeImplicitOp(fields=("b",))
    term_rows = (
        (0, TendencyTerm(name="du", fn=Rotation.du)),
        (1, TendencyTerm(name="mix", treatment=Treatment.IMPLICIT,
                         implicit=op)),
    )
    with pytest.raises(AssemblyError,
                       match=r"IMPLICIT term Advection/mix"):
        make_composer(
            field_table,
            modules=(Rotation(), Advection(),
                     TendencyEnvelope(terms=terms.implicit)),
            terms=term_rows)


def test_linear_match_refused(field_table):
    with pytest.raises(AssemblyError,
                       match=r"linear=True term Rotation/du"):
        make_composer(
            field_table,
            modules=(Rotation(), Advection(),
                     TendencyEnvelope(terms=terms.explicit)))


def test_empty_match_without_filter_is_an_error(field_table):
    term_rows = (
        (0, TendencyTerm(name="du", fn=Rotation.du, linear=True)),
        (1, TendencyTerm(name="db", fn=Advection.db, linear=True)),
    )
    with pytest.raises(AssemblyError, match="matches no collected"):
        make_composer(
            field_table,
            modules=(Rotation(), Advection(),
                     TendencyEnvelope(terms=NONLINEAR)),
            terms=term_rows)


def test_empty_match_under_a_filter_downgrades_to_a_warning(
        field_table):
    # the load-bearing case: a linear-only filtered leg (OB's
    # backward_filter=fr.terms.linear) keeps an inert envelope
    envelope = TendencyEnvelope(terms=NONLINEAR)
    with pytest.warns(UserWarning, match="matches no collected"):
        composer = make_composer(
            field_table,
            modules=(Rotation(), Advection(), envelope),
            term_filter=terms.linear)
    kept = composer.schedule.kind_entries(None)
    assert [entry.key for entry in kept] == ["Rotation/du"]
    assert not kept[0].enveloped
