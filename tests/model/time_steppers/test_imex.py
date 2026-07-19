"""Tests for the IMEX multistep driver (time_steppers/imex.py).

Covers the exact coefficient-level tables (verbatim), the CNAB2/SBDF2
factories, the reference vertical-diffusion consumer (exact 1D modal
decay within scheme order, and the stiff-kappa column stable where an
explicit scheme would blow up), the empty-implicit degeneration
(CNAB2 with no IMPLICIT terms == textbook AB2 driving), the
IMPLICIT-under-explicit assembly error (never a silent demotion), the
warm-up crossing a single jit/scan compile, and the restart-bitwise
round trip of every IMEXState leaf.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fridom.framework.utils import jaxify
from fridom.model.clock import Clock
from fridom.model.composer import TendencyComposer
from fridom.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.model.errors import AssemblyError
from fridom.model.implicit import VerticalDiffusion
from fridom.model.model import Model
from fridom.model.module import Module
from fridom.model.terms import (
    TERM_ATTRIBUTE,
    TendencyTerm,
    Treatment,
    term,
)
from fridom.model.time_steppers.adam_bashforth import (
    AdamBashforth,
)
from fridom.model.time_steppers.imex import (
    CNAB2,
    SBDF2,
    IMEXMultistep,
    IMEXState,
    _scaled,
    _zero_vector,
)
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.space_patterns import Collocated

KAPPA = 0.05
LAM = 0.7


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


class Decay:

    """Explicit du/dt = -lam*u."""

    @term(name="decay")
    def du(self, state, _ctx):
        return {"u": state["u"] * (-LAM)}


class Mixer:

    """Holds the implicit vertical-diffusion term (no explicit fn)."""


def term_of(method):
    return getattr(method, TERM_ATTRIBUTE)


def const_kappa(_module, _state, _ctx, _name):
    return KAPPA


# ================================================================
#  1D column fixtures (the vertical-diffusion consumer)
# ================================================================
COLUMN = 16


@pytest.fixture
def mz():
    return IntervalMesh(COLUMN, (0.0, 1.0), periodic=False, name="z")


@pytest.fixture
def grid(mz):
    return Grid((mz,))


@pytest.fixture
def column_table(grid, mz):
    return FakeFieldTable(grid, (
        Record("b", mz.center, Lifecycle.PROGNOSTIC, 0),
    ))


def column_state(table, data):
    (record,) = tuple(table)
    return VectorField({
        "b": table.grid.create_field(
            record.space, data=jnp.asarray(data), name="b")})


def diffusion_schedule(table, stepper, *, fields=("b",)):
    op = VerticalDiffusion("z", fields, const_kappa)
    modules = (Mixer(),)
    schedule = TendencyComposer(
        field_table=table, modules=modules,
        terms=((0, TendencyTerm(name="mix",
                                treatment=Treatment.IMPLICIT,
                                implicit=op)),),
        stages=(), time_stepper=stepper,
        binding_table=None).schedule
    return schedule, modules


def numpy_operator(grid, mz):
    """Build the dense (N,N) kernel-matching operator (Neumann rows)."""
    z = np.asarray(
        grid.evaluation_nodes(mz.center, "z").data).reshape(-1)
    dz = z[1] - z[0]
    main = np.full(COLUMN, -2.0)
    main[0] = -1.0
    main[-1] = -1.0
    d2 = (np.diag(main) + np.diag(np.ones(COLUMN - 1), 1)
          + np.diag(np.ones(COLUMN - 1), -1)) / dz ** 2
    return KAPPA * d2


def run(stepper, schedule, modules, state, steps):
    sst = stepper.init(state)
    clock = Clock()
    for _ in range(steps):
        sst, state, clock = stepper.step(
            sst, state, schedule.bind(modules), clock)
    return sst, state, clock


# ================================================================
#  The exact coefficient-level tables (verbatim)
# ================================================================
def test_cnab2_levels_verbatim():
    assert CNAB2(1.0).levels == (
        ((1.0, 0.0), (1.0,), 0.0, 1.0),
        ((3 / 2, -1 / 2), (1.0,), 1 / 2, 1 / 2),
    )


def test_sbdf2_levels_verbatim():
    assert SBDF2(1.0).levels == (
        ((1.0, 0.0), (1.0, 0.0), 0.0, 1.0),
        ((4 / 3, -2 / 3), (4 / 3, -1 / 3), 0.0, 2 / 3),
    )


def test_factories_return_configured_drivers():
    assert isinstance(CNAB2(1.0), IMEXMultistep)
    assert CNAB2(1.0).scheme == "cnab2"
    assert SBDF2(1.0).scheme == "sbdf2"


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError, match="unknown IMEX scheme"):
        IMEXMultistep(1.0, scheme="cnab3")


def test_supported_treatments():
    assert CNAB2(1.0).supported_treatments == frozenset(
        {Treatment.EXPLICIT, Treatment.IMPLICIT})


def test_fingerprint_excludes_dt_includes_scheme():
    assert (CNAB2(1.0).fingerprint_token()
            == CNAB2(-9.0).fingerprint_token())
    assert (CNAB2(1.0).fingerprint_token()
            != SBDF2(1.0).fingerprint_token())


# ================================================================
#  IMEXState — restart-bitwise round trip
# ================================================================
@pytest.mark.parametrize("factory", [CNAB2, SBDF2])
def test_imex_state_round_trip_bitwise(column_table, factory):
    stepper = factory(0.02)
    schedule, modules = diffusion_schedule(column_table, stepper)
    b0 = np.cos((np.arange(COLUMN) + 0.5) * 2 * np.pi / COLUMN)
    sst, _, _ = run(stepper, schedule, modules,
                    column_state(column_table, b0), 3)
    leaves, treedef = jax.tree_util.tree_flatten(sst)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is IMEXState
    for original, restored in zip(
            jax.tree_util.tree_leaves(sst),
            jax.tree_util.tree_leaves(rebuilt), strict=True):
        assert np.array_equal(np.asarray(original),
                              np.asarray(restored))


def test_cnab2_has_no_state_ring(column_table):
    # x_history is () for CNAB2, depth-1 for SBDF2 (a treedef delta)
    template = column_state(column_table, np.zeros(COLUMN))
    assert CNAB2(0.1).init(template).x_history == ()
    assert len(SBDF2(0.1).init(template).x_history) == 1


@pytest.mark.parametrize("factory", [CNAB2, SBDF2])
def test_f_ring_carries_past_entries_only(column_table, factory):
    # the F ring stores explicit_depth-1 PAST contributions — the
    # newest f_n is computed fresh each step (the AB memory cut)
    template = column_state(column_table, np.zeros(COLUMN))
    assert len(factory(0.1).init(template).f_history) == 1


def test_imex_state_is_frozen(column_table):
    sst = CNAB2(0.1).init(column_state(column_table,
                                       np.zeros(COLUMN)))
    with pytest.raises(AttributeError, match="frozen"):
        sst.warmup = jnp.asarray(1, dtype=jnp.int32)


# ================================================================
#  The reference vertical-diffusion consumer — modal decay
# ================================================================
@pytest.mark.parametrize("factory", [CNAB2, SBDF2])
def test_modal_decay_matches_the_exact_solution(
        column_table, grid, mz, factory):
    matrix = numpy_operator(grid, mz)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    b0 = np.cos((np.arange(COLUMN) + 0.5) * 3 * np.pi / COLUMN)
    dt, steps = 0.02, 50
    stepper = factory(dt)
    schedule, modules = diffusion_schedule(column_table, stepper)
    _, state, _ = run(stepper, schedule, modules,
                      column_state(column_table, b0), steps)
    exact = eigenvectors @ (
        np.exp(eigenvalues * dt * steps) * (eigenvectors.T @ b0))
    assert np.allclose(np.asarray(state["b"].data), exact, atol=2e-4)


def test_cnab2_is_second_order_in_time(column_table, grid, mz):
    matrix = numpy_operator(grid, mz)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    b0 = np.cos((np.arange(COLUMN) + 0.5) * 3 * np.pi / COLUMN)
    exact = eigenvectors @ (
        np.exp(eigenvalues * 1.0) * (eigenvectors.T @ b0))

    def error(dt):
        steps = round(1.0 / dt)
        stepper = CNAB2(dt)
        schedule, modules = diffusion_schedule(column_table, stepper)
        _, state, _ = run(stepper, schedule, modules,
                          column_state(column_table, b0), steps)
        return np.max(np.abs(np.asarray(state["b"].data) - exact))

    order = np.log2(error(0.04) / error(0.02))
    assert order == pytest.approx(2.0, abs=0.4)


def test_stiff_kappa_column_is_stable(column_table):
    # kappa*dt huge: CNAB2 stays bounded and decays where an explicit
    # scheme would blow up
    b0 = np.cos((np.arange(COLUMN) + 0.5) * 3 * np.pi / COLUMN)
    stepper = CNAB2(0.5)  # kappa*dt/dz^2 ~ O(N^2) -> deeply stiff
    schedule, modules = diffusion_schedule(column_table, stepper)
    _, state, _ = run(stepper, schedule, modules,
                      column_state(column_table, b0), 60)
    data = np.asarray(state["b"].data)
    assert np.all(np.isfinite(data))
    assert np.max(np.abs(data)) < np.max(np.abs(b0))


# ================================================================
#  Empty-implicit degeneration — CNAB2 == textbook AB2 driving
# ================================================================
def reference_ab2(u0, lam, dt, steps):
    """FB-Euler start, then textbook AB2 (the CNAB2 explicit member)."""
    u = np.asarray(u0, dtype=np.float64).copy()
    f_prev = np.zeros_like(u)
    for n in range(steps):
        f = -lam * u
        # level 0 FB Euler, then textbook AB2 (level 1+)
        u = (u + dt * f if n == 0
             else u + dt * (1.5 * f - 0.5 * f_prev))
        f_prev = f
    return u


def test_cnab2_empty_implicit_is_textbook_ab2(mz, grid):
    # a prognostic driven by a purely EXPLICIT term under CNAB2: the
    # solve set is empty and the scheme degenerates to AB2 driving
    table = FakeFieldTable(grid, (
        Record("u", mz.center, Lifecycle.PROGNOSTIC, 0),))
    stepper = CNAB2(0.05)
    modules = (Decay(),)
    schedule = TendencyComposer(
        field_table=table, modules=modules,
        terms=((0, term_of(Decay.du)),), stages=(),
        time_stepper=stepper, binding_table=None).schedule
    u0 = np.linspace(0.5, 1.5, COLUMN)
    state = column_state_named(table, "u", u0)
    _, state, _ = run(stepper, schedule, modules, state, 12)
    expected = reference_ab2(u0, LAM, 0.05, 12)
    assert np.allclose(np.asarray(state["u"].data), expected,
                       rtol=1e-11, atol=0.0)


def column_state_named(table, name, data):
    (record,) = tuple(table)
    return VectorField({
        name: table.grid.create_field(
            record.space, data=jnp.asarray(data), name=name)})


# ================================================================
#  IMPLICIT under an explicit-only stepper — assembly error
# ================================================================
def test_implicit_term_under_adam_bashforth_errors(column_table):
    op = VerticalDiffusion("z", ("b",), const_kappa)
    with pytest.raises(AssemblyError, match="supported"):
        TendencyComposer(
            field_table=column_table, modules=(Mixer(),),
            terms=((0, TendencyTerm(name="mix",
                                    treatment=Treatment.IMPLICIT,
                                    implicit=op)),),
            stages=(), time_stepper=AdamBashforth(0.1),
            binding_table=None)


# ================================================================
#  Warm-up crossing a single jit/scan compile
# ================================================================
@pytest.mark.parametrize("factory", [CNAB2, SBDF2])
def test_warmup_scan_compiles_once(column_table, compile_counter,
                                   factory):
    steps = 5
    stepper = factory(0.02)
    schedule, modules = diffusion_schedule(column_table, stepper)
    b0 = np.cos((np.arange(COLUMN) + 0.5) * 2 * np.pi / COLUMN)

    def go(stepper, carry):
        def body(carry, _):
            state, sst, clock = carry
            sst, state, clock = stepper.step(
                sst, state, schedule.bind(modules), clock)
            return (state, sst, clock), None
        return jax.lax.scan(body, carry, None, length=steps)[0]

    go_jit = jax.jit(go)
    canonical = jax.jit(lambda tree: tree)
    state = column_state(column_table, b0)
    carry = canonical((state, stepper.init(state), Clock()))
    stepper_in = canonical(stepper)
    compile_counter.reset()
    final_state, final_sst, final_clock = go_jit(stepper_in, carry)
    compiles = compile_counter.count
    assert compiles >= 1

    # a warmed carry re-enters the same trace (bitwise mid-warm-up
    # restart), and a dt sweep shares the compile
    go_jit(stepper_in, (final_state, final_sst, final_clock))
    assert compile_counter.count == compiles
    other = factory(0.01)
    state2 = column_state(column_table, b0)
    go_jit(canonical(other),
           canonical((state2, other.init(state2), Clock())))
    assert compile_counter.count == compiles

    assert int(final_sst.warmup) == 1  # saturated at levels - 1
    assert int(final_clock.it) == steps
    _, eager, _ = run(stepper, schedule, modules,
                      column_state(column_table, b0), steps)
    assert np.allclose(np.asarray(final_state["b"].data),
                       np.asarray(eager["b"].data), rtol=1e-11)


# ================================================================
#  End-to-end: an IMEX toy model through fr.Model + advance
# ================================================================
@partial(jaxify, dynamic=())
class ToyModule(Module):

    """Explicit decay on u + implicit vertical diffusion on b."""

    field_declarations = (
        FieldDeclaration("u", space=Collocated(), long_name="u"),
        FieldDeclaration("b", space=Collocated(), long_name="b"),
    )

    @term(name="decay", advances=("u",))
    def decay(self, state, _ctx):
        return {"u": state["u"] * (-0.5)}

    @term(name="mix", treatment=Treatment.IMPLICIT,
          implicit=VerticalDiffusion("z", ("b",), const_kappa))
    def mix(self, _state, _ctx):
        return {}  # IMPLICIT: the solve/apply carry it, not S2


def test_end_to_end_imex_model_advances(compile_counter):
    grid = Grid((IntervalMesh(COLUMN, (0.0, 1.0), periodic=False,
                              name="z"),))
    model = Model(grid=grid, modules=(ToyModule(),),
                  time_stepper=CNAB2(0.02))
    b0 = np.cos((np.arange(COLUMN) + 0.5) * 3 * np.pi / COLUMN)
    model.set_fields(u=jnp.full(COLUMN, 2.0), b=jnp.asarray(b0))
    treedef = jax.tree_util.tree_structure(model.carry)

    compile_counter.reset()
    model.advance(40)
    compiles = compile_counter.count
    assert compiles >= 1

    # treedef stable across the advance, and a second advance shares
    # the compiled chunk (single compile)
    assert jax.tree_util.tree_structure(model.carry) == treedef
    model.advance(40)
    assert compile_counter.count == compiles

    u = np.asarray(model.state["u"].data)
    b = np.asarray(model.state["b"].data)
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(b))
    # the explicit term decays u (exp(-0.5 * 80 * 0.02)) ...
    assert np.max(np.abs(u)) < 2.0
    # ... and the implicit diffusion decays b toward its mean
    assert np.max(np.abs(b)) < np.max(np.abs(b0))


# ================================================================
#  Deferred surface
# ================================================================
def test_time_discretization_effect_deferred():
    with pytest.raises(NotImplementedError, match="deferred"):
        CNAB2(0.1).time_discretization_effect(np.array([1.0]))


# ================================================================
#  Storage-frame _scaled / _zero_vector (indivisible-shard Phase 3)
# ================================================================
def _corrupt_ghosts(field):
    """Return the field with NaN in every ghost slot (interior intact)."""
    probe = field.with_data(jnp.ones_like(field.data)).storage
    stored = field.with_data(field.data).storage
    return field.with_storage(jnp.where(probe == 0, jnp.nan, stored))


def test_scaled_storage_frame_matches_true_frame(grid, mz):
    # storage-frame _scaled == old unpad -> scale -> pad on the TRUE
    # DOFs, even with garbage (NaN) ghost lanes.
    field = _corrupt_ghosts(grid.create_field(
        mz.center, data=jnp.linspace(0.5, 1.5, COLUMN), name="b"))
    assert field.storage.shape != field.data.shape  # ghosts exist
    vec = VectorField({"b": field})
    weight = jnp.asarray(0.75 * 0.02, dtype=jnp.float64)
    new = _scaled(vec, weight)
    old = vec.map(lambda f: f.with_data(weight * f.data))
    new_data = np.asarray(new["b"].data)
    assert np.all(np.isfinite(new_data))  # no ghost NaN leaked
    assert np.array_equal(new_data, np.asarray(old["b"].data))


def test_zero_vector_storage_frame_matches_true_frame(grid, mz):
    # storage-frame _zero_vector is zero on EVERY slot (both spellings
    # zero-fill the ghosts), and identical to the old spelling.
    field = _corrupt_ghosts(grid.create_field(
        mz.center, data=jnp.linspace(0.5, 1.5, COLUMN), name="b"))
    vec = VectorField({"b": field})
    new = _zero_vector(vec)
    old = vec.map(lambda f: f.with_data(jnp.zeros_like(f.data)))
    new_data = np.asarray(new["b"].data)
    assert np.array_equal(new_data, np.zeros(COLUMN))
    assert np.array_equal(new_data, np.asarray(old["b"].data))
    # zeroed on the whole storage frame too (ghost NaNs are gone)
    assert np.all(np.asarray(new["b"].storage) == 0.0)
