"""
The Model composition root and the run loop.

Description
-----------
Wave 4.2: ``Model`` (the host-side composition root and driver),
``ModelState`` (the full dynamic carry), ``step_chunk`` (THE one
framework-level jitted entry), ``ParameterView``,
``DiagnosticsNamespace`` and ``BoundDiagnostic``. Owning class spec:
``design/specs/model/classes/model.md``; run-loop rules:
``design/specs/model/02_rules.md`` (Clock precision, the
three-operation matrix, the panic-flag ledger).

The Model is a **host object, never a pytree**: it holds the frozen
assembly artifacts plus the single mutable slot ``_carry``; every
lifecycle method is a mutating spelling over a pure carry
transformer (a new carry is built functionally and the reference is
swapped). No run-loop state lives outside the carry (CS-11).

Carry-sharding discipline (wave-4.1 finding): fields born via
``grid.create_field`` and jit outputs may carry differently-spelled
(same-layout) shardings and weakly/strongly typed scalars — a
different jit-cache key. Every carry committed by the model
therefore passes once through the module-level jitted
:func:`_canonicalize` identity, so repeated ``advance()`` calls hit
ONE compiled-chunk cache entry (proven by the compile-counter
tests).
"""
# Wave 4.2: Model, ModelState, PanicState, step_chunk,
#    ParameterView, DiagnosticsNamespace, BoundDiagnostic
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real, jaxify
from fridom.model.assembly import _collect_terms, assemble
from fridom.model.clock import Clock
from fridom.model.context import StepContext
from fridom.model.declarations import Lifecycle
from fridom.model.errors import AssemblyError, MissingParameterError
from fridom.model.io.snapshots import (
    FORMAT_VERSION,
    SnapshotManifest,
    Snapshots,
    check_dt,
    check_fingerprint,
    read_leaves,
    read_manifest,
    write_snapshot,
)
from fridom.model.io.streams import (
    IOCollisionError,
    SnapshotMismatchError,
)
from fridom.model.results import (
    AdvanceResult,
    PanicError,
    RunResult,
    RunTargetError,
)
from fridom.model.schedule import (
    BoundSchedule,
    apply_add,
    evaluate_entry,
    zero_like,
)
from fridom.model.stages import StageKind
from fridom.model.terms import Treatment
from fridom.model.time_dependent import resolve_at
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.vector_field import (
    VectorField,
    rehome_component,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from fridom.model.assembly import (
        AssemblyArtifacts,
        AssemblyRecord,
        Fingerprint,
        ParameterBinding,
    )
    from fridom.model.field_table import (
        FieldRecord,
        FieldTable,
    )
    from fridom.model.parameters import ParameterDeclaration
    from fridom.model.report import AssemblyReport
    from fridom.model.time_steppers.base import (
        StepperState,
        TimeStepper,
    )
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import TensorProductSpace

_log = logging.getLogger(__name__)

M = TypeVar("M")

# the fixed carry field set (write-once frozen discipline; also the
# dynamic pytree leaf order)
_CARRY_FIELDS: Final[tuple[str, ...]] = (
    "state", "modules", "stepper_state", "clock", "panic")

# the default host-sync granularity of advance() (04 section 6.3:
# max_chunk auto ~256; run()'s trigger-driven subdivision is wave 5)
_DEFAULT_CHUNK: Final[int] = 256

# the sign-agnostic run-target snap tolerance: steps = ceil(k - eps),
# so an exact-multiple target does not overshoot by a whole step
# (04 section 6.3, amended V-S1).
_TARGET_EPS: Final[float] = 1e-9


def _run_target_seconds(value: object, *, name: str) -> float:
    """Convert a run-target spelling to float seconds."""
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RunTargetError(
            f"{name}= takes float seconds or np.timedelta64; got "
            f"{value!r}") from exc


def _dtype_int() -> jnp.dtype:
    """Return the default integer dtype (the clock's ``it`` twin)."""
    return jnp.result_type(int)


def _validate_filter_names(
    term_filter: Callable | None, term_keys: tuple[str, ...],
) -> None:
    """Reject a filter that names an unknown term (build error, 2.8)."""
    if term_filter is None:
        return
    referenced = getattr(term_filter, "referenced_names", None)
    if referenced is None:
        return
    unknown = frozenset(referenced()) - set(term_keys)
    if unknown:
        raise AssemblyError(
            f"the term filter names unknown terms "
            f"{tuple(sorted(unknown))}; declared terms: "
            f"{tuple(sorted(term_keys))}")


# ================================================================
#  PanicState — the S5 ledger entry
# ================================================================
class PanicState(NamedTuple):

    """
    The carried S5 NaN-seam record: sticky flag + detection it.

    Description
    -----------
    A NamedTuple (hence automatically a pytree): ``flag`` is the
    sticky panic bit (once true, stays true for the run), ``it``
    records the clock iteration of the chunk boundary at which the
    non-finite state was DETECTED. The chunk body reduces into it
    once per chunk — the 2026-07-12 GPU benchmark priced the
    per-step reduction at a measurable share of the step, so the
    sanctioned chunk-boundary-only cadence (04_run_loop_io §6.3) is
    now the behavior; the exact first-bad step remains recoverable
    via ``advance(debug_nan=True)`` + ``replay_nan`` (chunk(1)
    re-stepping). The host reads the pair at chunk boundaries only
    (the one sanctioned host synchronization, CS-4).

    Parameters
    ----------
    flag : jax.Array
        Sticky boolean: non-finite values detected so far.
    it : jax.Array
        Detection iteration — the chunk-end iteration of the first
        chunk whose final state was non-finite (equals the exact
        first-bad step for chunk length 1); meaningful only while
        ``flag`` is set.
    """

    flag: jax.Array
    it: jax.Array


def _fresh_panic() -> PanicState:
    """Return the cleared panic ledger ``(False, 0)``."""
    return PanicState(
        flag=jnp.zeros((), dtype=bool),
        it=jnp.asarray(0, dtype=_dtype_int()))


# ================================================================
#  ModelState — the full dynamic carry
# ================================================================
@partial(jaxify, dynamic=_CARRY_FIELDS)
class ModelState:

    """
    The full dynamic carry (variable spelled ``model_state``).

    Description
    -----------
    All five slots are dynamic pytree subtrees — the aux data is
    empty, so every carry of one assembly shares one treedef. No
    coupling/run-loop state lives outside it (CS-11): restart
    restores every leaf bitwise. Frozen: advance builds fresh
    carries functionally (:meth:`replace`).

    Parameters
    ----------
    state : VectorField | None
        All declared components, all lifecycles (``None`` iff the
        composition declares no fields — CS-13).
    modules : tuple
        The live module pytrees (parameter leaves, own AUX).
    stepper_state : StepperState
        Multistep buffers + warm-up counter.
    clock : Clock
        The traced clock (global-width leaves).
    panic : PanicState
        The S5 ledger: (sticky flag, first-failure it).
    """

    state: VectorField | None
    modules: tuple
    stepper_state: StepperState
    clock: Clock
    panic: PanicState

    def __init__(
        self,
        state: VectorField | None,
        modules: tuple,
        stepper_state: StepperState,
        clock: Clock,
        panic: PanicState,
    ) -> None:
        """Freeze the carry; see the class docstring."""
        self.state = state
        self.modules = tuple(modules)
        self.stepper_state = stepper_state
        self.clock = clock
        self.panic = panic

    def __setattr__(self, name: str, value: object) -> None:
        """Set a carry field exactly once (frozen thereafter)."""
        # write-once: __init__ and pytree unflattening set fresh
        # attributes; everything else raises
        if name in _CARRY_FIELDS and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"ModelState is frozen: cannot set {name!r} (build a "
            "new carry via replace)")

    def __delattr__(self, name: str) -> None:
        """Raise: ModelState is frozen."""
        raise AttributeError(
            f"ModelState is frozen: cannot delete {name!r}")

    def replace(self, **changes: object) -> ModelState:
        """
        Return a new carry with the named slots replaced.

        Parameters
        ----------
        **changes : object
            New values for carry slots (``state``, ``modules``,
            ``stepper_state``, ``clock``, ``panic``).

        Returns
        -------
        ModelState
            The new carry; ``self`` is unchanged.
        """
        unknown = tuple(name for name in changes
                        if name not in _CARRY_FIELDS)
        if unknown:
            raise TypeError(
                f"unknown carry slots {unknown}; slots are "
                f"{_CARRY_FIELDS}")
        values = {name: changes.get(name, getattr(self, name))
                  for name in _CARRY_FIELDS}
        return ModelState(**values)

    def __repr__(self) -> str:
        """Compact host-side summary."""
        names = (self.state.component_names
                 if self.state is not None else ())
        return (f"ModelState(state over {names!r}, "
                f"{len(self.modules)} modules, {self.clock!r})")


# ================================================================
#  Carry canonicalization (the wave-4.1 sharding discipline)
# ================================================================
#: leaves at/above this size are the carry's big cubes — the only
#: ones donated into :func:`_canonicalize` (the memory win). Smaller
#: leaves are copied first so a caller-held handle (a module's
#: parameter scalar, fed straight into the construction carry) is
#: never invalidated by the donation.
_DONATE_MIN_BYTES: Final[int] = 1 << 20  # 1 MiB


def _strengthen(leaf: object) -> jax.Array:
    """Coerce one leaf to a strongly-typed array of its own dtype."""
    array = jnp.asarray(leaf)
    # astype with an explicit dtype strips the weak type; a no-op
    # for already-strong leaves
    return array.astype(array.dtype)


@partial(jax.jit, donate_argnums=0)
def _canonicalize_donating(tree: Any) -> Any:
    """Strengthen every leaf under a jit that DONATES its argument."""
    return jax.tree_util.tree_map(_strengthen, tree)


def _prepare_for_donation(tree: Any) -> Any:
    """
    Copy the leaves that cannot be donated as-is; pass the rest.

    Description
    -----------
    :func:`_canonicalize` donates its argument, so two leaf classes
    must be handed a private buffer first:

    - **Aliased buffers.** ``AdamBashforth.init`` builds its history
      ring as ``(zeros,) * (order - 1)`` — the SAME device buffers in
      every ring slot — so a freshly warmed carry (construction and
      every re-warm ``_commit``) holds each ring buffer more than
      once. jax 0.10.2 raises ``INVALID_ARGUMENT: Attempt to donate
      the same buffer twice`` on such a tree; copying the second and
      later occurrences gives donation a distinct buffer per slot.
    - **Small leaves.** A module's parameter leaves (0-d strong
      arrays) are fed straight into the construction carry (assembly
      does not clone modules), and jax donates 0-d scalars too — a
      caller that keeps a module handle would otherwise see its buffer
      deleted. Copying every sub-:data:`_DONATE_MIN_BYTES` leaf keeps
      such handles valid at negligible cost; the win rides on the big
      cubes, which are unique and pass through to be donated.

    Non-array leaves (clock floats, ints, ``None``) pass through
    untouched — the jitted :func:`_strengthen` coerces them and they
    are never donated.
    """
    seen: set[int] = set()

    def prepare(leaf: object) -> object:
        if not isinstance(leaf, jax.Array):
            return leaf
        key = id(leaf)
        if key in seen or leaf.nbytes < _DONATE_MIN_BYTES:
            # aliased (would donate twice) or small (a caller may still
            # hold it): hand donation a fresh, privately owned buffer
            return jnp.copy(leaf)
        seen.add(key)
        return leaf

    return jax.tree_util.tree_map(prepare, tree)


def _canonicalize(tree: Any) -> Any:
    """
    Normalize a carry's leaves to the canonical jit-output form.

    Description
    -----------
    Host-born leaves (``grid.create_field`` products, ``asarray``
    scalars) may carry same-layout-but-differently-spelled shardings
    and weakly-typed avals; jit outputs are the canonical form. One
    pass through the jitted identity commits every leaf, so every
    carry the model holds keys the SAME compiled-chunk cache entry.

    Donation contract
    -----------------
    The argument is DONATED: its big device buffers are consumed to
    build the output, so the whole-carry copy no longer transiently
    doubles the ~carry-sized footprint. Callers must drop every
    reference to the input carry and its leaves after the call — the
    model does (it rebinds ``self._carry`` to the return).
    :func:`_prepare_for_donation` first copies the leaves that cannot
    be donated safely (aliased ring buffers, small parameter scalars a
    caller might still hold), so only large model-internal buffers are
    actually donated.
    """
    return _canonicalize_donating(_prepare_for_donation(tree))


def _copy_leaves(tree: Any) -> Any:
    """Deep-copy a pytree's array leaves (the debug_nan copy)."""
    return jax.tree_util.tree_map(jnp.copy, tree)


# ================================================================
#  Carry defragmentation (the one-time pre-chunk pool re-coalesce)
# ================================================================
#: leaves at/above this size are repacked by the defragmenter (the
#: same big-cube threshold the donation uses)
_DEFRAG_MIN_BYTES: Final[int] = 1 << 20  # 1 MiB


def _defrag_enabled() -> bool:
    """Whether the defrag kill switch is unset (read at call time)."""
    # mirrors the env-knob style of src/fridom/_compile_cache.py
    return os.environ.get("FRIDOM_DISABLE_DEFRAG") != "1"


def _template_builder(
    template: VectorField,
) -> Callable[[], VectorField]:
    """
    Turn one built template into a trace-stable zero-builder.

    Description
    -----------
    The tendency template is only ever CONSUMED by ``stepper.init``
    (at construction and on every re-warm), so caching its device
    buffers would keep one zero PROGNOSTIC vector — full padded
    fields — alive for the model's lifetime. Instead the template's
    STRUCTURE is captured host-side (treedef + per-leaf
    shape/dtype/sharding; no device buffers) and the zeros are
    rebuilt through one jitted zero-arg builder pinned to the
    negotiated layout: repeated re-warms
    (``update_parameters``/``reset``/``set_aux(rewarm=True)``) hit
    the one cached trace — never the grid's pad-and-sync store path
    (a per-call retrace under multi-device) — and the buffers are
    freed as soon as ``init`` consumes them.

    Parameters
    ----------
    template : VectorField
        A freshly built zero PROGNOSTIC vector (born on the grid's
        negotiated layout); only its structure is retained.

    Returns
    -------
    Callable[[], VectorField]
        The cached zero-template builder.
    """
    leaves, treedef = jax.tree_util.tree_flatten(template)
    specs = tuple(
        (tuple(leaf.shape), leaf.dtype) for leaf in leaves)
    shardings = tuple(leaf.sharding for leaf in leaves)
    # the jitted zero-maker is shared across models (keyed by
    # structure) so identical re-assemblies add ZERO compiles —
    # the _CHUNK_EXECUTABLES discipline
    key = (specs, shardings)
    zeros = _ZERO_MAKERS.get(key)
    if zeros is None:

        @partial(jax.jit, out_shardings=shardings)
        def zeros() -> tuple[jax.Array, ...]:
            return tuple(
                jnp.zeros(shape, dtype) for shape, dtype in specs)

        _ZERO_MAKERS[key] = zeros

    def build() -> VectorField:
        return jax.tree_util.tree_unflatten(treedef, zeros())

    return build


#: shared jitted zero-template makers, keyed by (specs, shardings)
_ZERO_MAKERS: Final[dict[tuple, Callable]] = {}


def _is_field(node: object) -> bool:
    """Whether ``node`` is a field (the claim-reset tree boundary)."""
    return isinstance(node, ScalarField)


def _reset_ghost_claims(tree: M) -> M:
    """
    Zero every field's ghost-validity claim in a carry (sub)tree.

    Description
    -----------
    ``halo_valid`` is treedef-participating static aux, and the
    chunk scan demands ONE carry treedef across iterations
    (``lax.scan``). Field arithmetic propagates ghost validity
    through the step (storage-frame combine, task 1.8 stage B), so
    a step output's claims are whatever the trace produced — e.g.
    the AB tendency ring would enter with the zero claims of its
    ``init()`` template and leave with the summed tendencies'
    nonzero claims. Resetting the claims at the carry boundary is
    always sound (claiming fewer valid layers than the storage
    holds) and free at runtime (static metadata, no array op);
    within-step validity reuse — where the elided halo fills live —
    is untouched.
    """
    def reset(leaf: object) -> object:
        if isinstance(leaf, ScalarField):
            zero = HaloSpec.zero(tuple(leaf.function_space.names))
            if leaf.halo_valid == zero:
                return leaf
            return type(leaf)(
                leaf.grid, leaf.function_space,
                leaf._data,  # noqa: SLF001 — plumbing-constructor seam
                leaf.metadata)
        return leaf

    return jax.tree_util.tree_map(reset, tree, is_leaf=_is_field)


def _seal_carry_ghosts(tree: M) -> M:
    """
    Sync every field of a carry (sub)tree to full ghost validity.

    Description
    -----------
    The production-side twin of :func:`_reset_ghost_claims`: instead
    of zeroing the claims at the carry boundary (which forces every
    ghost consumer of the NEXT step to re-fill at consumption, where
    the fill is absorbed into the consuming kernels and re-derived at
    every offset they read), the state vector is synced ONCE at the
    boundary — where its buffers materialize anyway (the scan carry
    is a materialization point, so the fill fuses into the kernel
    that writes the carry instead of into its many consumers).
    Fields already claiming full validity (untouched AUXILIARY
    fields riding the carry) pass through — the seal is a no-op for
    them, so only fields the step actually rebuilt pay a fill.
    """
    def seal(leaf: object) -> object:
        if isinstance(leaf, ScalarField):
            full = leaf.grid.decomposition.halo.over(
                tuple(leaf.function_space.names))
            if leaf.halo_valid == full:
                return leaf
            return leaf.grid.sync(leaf, materialize=True)
        return leaf

    return jax.tree_util.tree_map(seal, tree, is_leaf=_is_field)


def _scrub_ghost_storage(tree: M) -> M:
    """
    Re-store every field at the zero-ghost true-shape spelling.

    Description
    -----------
    The chunk-commit twin of :func:`_reset_ghost_claims`: storage-
    frame arithmetic leaves computed values in the (invalid) ghost
    slots, but persistence gathers true-shape leaves and re-pads on
    load (zero ghosts) — restart restores every leaf bitwise only
    if the committed carry itself carries the zero-ghost spelling.
    One unpad/re-pad per field per CHUNK (not per step), traced
    into the chunk body: amortized noise, interior bits untouched.
    """
    def scrub(leaf: object) -> object:
        if isinstance(leaf, ScalarField):
            return leaf.with_data(leaf.data)
        return leaf

    return jax.tree_util.tree_map(scrub, tree, is_leaf=_is_field)


# ================================================================
#  The live per-step schedule view
# ================================================================
class _LiveBoundSchedule(BoundSchedule):

    """
    A ``BoundSchedule`` whose ctx params read the LIVE stepper.

    Description
    -----------
    The landed ``BoundSchedule.context`` evaluates parameters
    through ``schedule.time_stepper`` — the assembly-time stepper
    object, whose ``dt`` leaf would enter the trace as a baked
    constant. The chunk body receives the stepper as a loop-invariant
    traced input instead, so this per-step view overrides P0 to
    resolve ``fr.params.TIME_STEP`` (and any other stepper-provided
    leaf) from the live traced stepper. Ephemeral, built per step
    inside the trace — exactly like its base.

    Parameters
    ----------
    schedule : Schedule
        The static schedule.
    modules : tuple
        The carry's live module tuple.
    stepper : TimeStepper
        The loop-invariant traced stepper input.
    """

    def __init__(
        self,
        schedule: Any,
        modules: tuple,
        stepper: TimeStepper,
    ) -> None:
        """Close the view over the live stepper; see class doc."""
        super().__init__(schedule, modules)
        self._live_stepper = stepper

    def context(
        self,
        clock: Any,
        *,
        dt: Any,
        stage_dt: Any,
        sums: Any = None,
    ) -> StepContext:
        """Build the P0 context with live-stepper parameter reads."""
        table = self.schedule.binding_table
        if table is None:
            params: Mapping[str, Any] = {}
        else:
            stage_time = getattr(clock, "time", clock)
            params = table.eval_params(
                self._modules, self._live_stepper, stage_time)
        return StepContext(params=params, clock=clock, dt=dt,
                           stage_dt=stage_dt, tendency_sums=sums)


# ================================================================
#  step_chunk — THE framework-level jitted entry
# ================================================================
def _all_finite(state: VectorField) -> jax.Array:
    """0-d bool: every component's true-shape data is finite (S5)."""
    checks = [jnp.isfinite(field.data).all() for field in state]
    return jnp.stack(checks).all()


def _chunk_body(
    record: AssemblyRecord,
    n: int,
    model_state: ModelState,
    stepper: TimeStepper,
    force_unroll: int | None = None,
    *,
    remat: bool = False,
) -> ModelState:
    """
    Trace one chunk: ``lax.scan`` of the composed step body.

    Description
    -----------
    Per step: ``schedule.bind`` over the carry's current modules
    (via the live view), ``stepper.step``, then the S6 DIAGNOSTIC
    stages. The S5 ``isfinite`` reduction into the sticky ``panic``
    pair runs ONCE per chunk, on the scan's final state (the
    chunk-boundary-only cadence sanctioned in 04_run_loop_io §6.3;
    priced per-step at a measurable share of the GPU step,
    2026-07-12 benchmark). Non-finite values propagate through the
    step algebra, so a mid-chunk blow-up is still caught at the
    boundary; the exact first-bad step is ``debug_nan``/
    ``replay_nan`` territory. The epilogue is the CHUNK BODY's,
    never the stepper's. NOTE (wave-5 seam): the S6 ctx carries no
    tendency sums yet — the treatment partition threads them at
    2.5.

    ``remat`` (keyword-only, default off) wraps the scan body in
    ``jax.checkpoint`` and forces ``unroll=1``: under reverse-mode
    autodiff the per-step forward pass is then re-run during
    backprop instead of held on the tape, trading recompute for
    O(n_steps) tape memory (the ``Model.propagator(remat=True)``
    knob). It is a no-op forward, so primal values are unchanged;
    the compiled ``advance`` path never sets it.
    """
    schedule = record.schedule
    run_diagnostics = bool(
        schedule.kind_entries(StageKind.DIAGNOSTIC))

    def one_step(
        carry: ModelState, _: None,
    ) -> tuple[ModelState, None]:
        """One composed step plus the S6 epilogue."""
        bound = _LiveBoundSchedule(schedule, carry.modules, stepper)
        stepper_state, state, clock = stepper.step(
            carry.stepper_state, carry.state, bound, carry.clock)
        # ---- S6: DIAGNOSTIC stages ------------------------------
        if run_diagnostics:
            ctx = bound.context(clock, dt=stepper.dt,
                                stage_dt=stepper.dt)
            state = bound.diagnostics(state, ctx)
        # ghost-claim discipline: the scan carry keeps ONE treedef
        # (halo_valid is static aux). The state vector is SEALED —
        # synced at the carry boundary, where its buffers materialize
        # anyway — so next step's consumers read valid ghosts instead
        # of re-filling at consumption; everything else (AB ring,
        # module state) gets the plain claim reset as before.
        return ModelState(
            _seal_carry_ghosts(state),
            _reset_ghost_claims(carry.modules),
            _reset_ghost_claims(stepper_state),
            clock, carry.panic), None

    # unroll by the stepper's carry period (e.g. the AB tendency
    # ring): the structural ring shift becomes dataflow renaming
    # instead of per-step buffer copies (TimeStepper.scan_unroll).
    # force_unroll is the async two-tier transient override: it
    # replaces the stepper's preference with a CHEAP unroll for the
    # length-C tier served while the full executable compiles.
    base_unroll = (int(stepper.scan_unroll) if force_unroll is None
                   else int(force_unroll))
    # remat checkpoints each step (adjoint tape -> recompute) and
    # forces unroll=1, so the checkpointed body IS one step.
    if remat:
        one_step = jax.checkpoint(one_step)
        base_unroll = 1
    unroll = max(1, min(base_unroll, n))
    # the scan init must carry the body's fixed-point treedef: state
    # sealed to full validity, all other claims zeroed
    init = ModelState(
        _seal_carry_ghosts(model_state.state),
        _reset_ghost_claims(model_state.modules),
        _reset_ghost_claims(model_state.stepper_state),
        model_state.clock, model_state.panic)
    out, _ = jax.lax.scan(one_step, init, xs=None, length=n,
                          unroll=unroll)
    # commit at the zero-ghost spelling (persistence contract)
    out = _scrub_ghost_storage(out)
    # ---- S5: one isfinite reduction into the sticky pair --------
    finite = _all_finite(out.state)
    newly_bad = jnp.logical_and(~out.panic.flag, ~finite)
    panic = PanicState(
        flag=jnp.logical_or(out.panic.flag, ~finite),
        it=jnp.where(newly_bad, out.clock.it, out.panic.it))
    return out.replace(panic=panic)


#: AOT-compiled chunk executables, keyed by (record, n, structure)
_CHUNK_EXECUTABLES: Final[dict[tuple, Any]] = {}

#: per-key compile accounting: key -> (seconds, memory analysis)
_CHUNK_COMPILE_LOG: Final[dict[tuple, tuple[float, object]]] = {}

#: guards writes to the two shared chunk dicts. Only meaningful
#: because the async two-tier holder runs a daemon compile thread
#: that swaps its executable in at a later chunk boundary — the lock
#: keeps that read-modify-write invariant honest.
_CHUNK_LOCK: Final[threading.Lock] = threading.Lock()


def _memory_analysis(compiled: Any) -> object:
    """One compiled executable's memory analysis (None if absent)."""
    try:
        return compiled.memory_analysis()
    except Exception:  # noqa: BLE001 — backend-optional report
        return None  # pragma: no cover


def _leaf_signature(leaf: object) -> tuple:
    """One leaf's contribution to the compiled-chunk cache key."""
    if isinstance(leaf, jax.Array):
        return (tuple(leaf.shape), str(leaf.dtype),
                bool(leaf.weak_type), str(leaf.sharding))
    return ("host", type(leaf).__name__)


def _leaf_sharding(leaf: object) -> jax.sharding.Sharding | None:
    """One carry leaf's sharding (None for a non-array leaf)."""
    return leaf.sharding if isinstance(leaf, jax.Array) else None


def _lower_chunk(
    record: AssemblyRecord,
    carry: ModelState,
    stepper: TimeStepper,
    n: int,
    force_unroll: int | None = None,
) -> Any:
    """
    Trace + lower one chunk to a ``Lowered`` (no XLA compile yet).

    Description
    -----------
    The carry-live half of :func:`_compile_chunk`: it captures the HLO
    module and donation/sharding intent from the carry's avals, holding
    NO reference to the carry's live buffers. The returned ``Lowered``
    can therefore be ``.compile()``-d on a background thread AFTER the
    main thread has already donated (consumed) the carry — the ordering
    the async two-tier scheme relies on.

    The chunk is pinned as a sharding **fixed-point**: its output carry
    is pinned (``out_shardings``) to the INPUT carry's per-leaf
    shardings. Without the pin a scan-internal reshard — e.g. the
    replicated tendency XLA produces for a spatially-uniform term,
    shifted into the AdamBashforth history ring — leaves the ring
    ``P()`` while the setup ring is ``P('devices')``, so the next
    ``advance``/``set_fields`` re-keys the cache and recompiles under
    multi-device (the wave-4.1 discipline holds the state field but
    not the stepper carry). The pin keeps every committed carry on the
    negotiated layout, so repeated advances key ONE entry on any
    device count.

    ``force_unroll`` (the async two-tier transient override) replaces
    the stepper's unroll for the cheap tier; both tiers share the same
    ``out_shardings`` pin, so swapping the full executable in for the
    cheap one at a chunk boundary is reshard-free.
    """
    out_shardings = jax.tree_util.tree_map(_leaf_sharding, carry)
    jitted = jax.jit(
        _chunk_body, static_argnums=(0, 1, 4), donate_argnums=(2,),
        out_shardings=out_shardings)
    return jitted.lower(record, n, carry, stepper, force_unroll)


def _compile_chunk(
    record: AssemblyRecord,
    carry: ModelState,
    stepper: TimeStepper,
    n: int,
    force_unroll: int | None = None,
) -> Any:
    """
    Lower + compile one chunk, pinned to the carry's shardings.

    Description
    -----------
    The record and chunk length are static; the carry is donated; the
    stepper is a loop-invariant NON-donated input. The jit object is
    ephemeral (one per cache miss) — the shared cache is
    :data:`_CHUNK_EXECUTABLES`, keyed below, not jax's internal one.
    Lowering (:func:`_lower_chunk`) and XLA ``.compile()`` are split so
    the async two-tier scheme can lower on the (carry-live) main thread
    and run only the slow compile off-thread; the sharding fixed-point
    rationale lives on :func:`_lower_chunk`. ``force_unroll`` overrides
    the stepper's unroll for the cheap transient tier.
    """
    return _lower_chunk(
        record, carry, stepper, n, force_unroll).compile()


def _chunk_key(
    record: AssemblyRecord,
    carry: ModelState,
    stepper: TimeStepper,
    n: int,
) -> tuple:
    """Build the compiled-chunk cache key (record + structure)."""
    leaves, treedef = jax.tree_util.tree_flatten((carry, stepper))
    return (record, n, treedef,
            tuple(_leaf_signature(leaf) for leaf in leaves))


class _TwoTier:

    """
    Serve a cheap chunk executable until the full one compiles.

    Description
    -----------
    The async two-tier holder: on a chunk cache miss it serves the
    CHEAP tier — a low-unroll (``force_unroll=1``) length-C executable
    — on the calling thread while a daemon thread XLA-compiles the
    full-unroll length-C executable from a pre-built ``Lowered``. Once
    ready, :meth:`settled` hands back the full executable so the caller
    can swap it into the cache; steady state is then bitwise the
    single-tier baseline with no holder indirection. A compile
    exception in the daemon is captured and re-raised by :meth:`settled`
    on the next call.

    Parameters
    ----------
    cheap : Any
        The already-compiled cheap-tier executable (served now).
    lowered_full : Any
        The full-unroll ``Lowered`` whose ``.compile()`` runs on the
        daemon thread (the carry is already consumed at this point).
    """

    def __init__(self, cheap: Any, lowered_full: Any) -> None:
        """Start the background full-compile; see the class doc."""
        self._cheap = cheap
        self._full: Any = None
        self._full_compile_s: float | None = None
        self._full_memory: object = None
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._compile_full, args=(lowered_full,),
            name="fridom-chunk-compile", daemon=True)
        self._thread.start()

    def _compile_full(self, lowered_full: Any) -> None:
        """Daemon body: XLA-compile the full executable off-thread."""
        started = time.perf_counter()
        try:
            compiled = lowered_full.compile()
        except BaseException as exc:  # noqa: BLE001 — re-raised on call
            with self._lock:
                self._error = exc
            return
        seconds = time.perf_counter() - started
        memory = _memory_analysis(compiled)
        with self._lock:
            self._full_compile_s = seconds
            self._full_memory = memory
            self._full = compiled

    def settled(self) -> Any:
        """
        Return the full executable if ready, else ``None``.

        Description
        -----------
        The fast path is a plain attribute load under the GIL (no
        lock): steady state after the swap never reaches here. A
        background compile exception is re-raised on the next call so
        the failure surfaces to the run loop rather than being lost on
        the daemon thread.

        Returns
        -------
        Any
            The compiled full executable, or ``None`` while pending.
        """
        full = self._full  # atomic ref load (GIL); no lock on hot path
        if full is not None:
            return full
        if self._error is not None:
            with self._lock:
                raise self._error
        return None

    def serve_cheap(
        self, carry: ModelState, stepper: TimeStepper,
    ) -> ModelState:
        """Run the cheap length-C executable on the donated carry."""
        return self._cheap(carry, stepper)

    def compile_log(self) -> tuple[float, object]:
        """Return the full tier's (seconds, memory), settled (tests)."""
        return (self._full_compile_s, self._full_memory)

    def wait(self) -> None:
        """Block until the background full-compile finishes (tests)."""
        self._thread.join()


def step_chunk(
    record: AssemblyRecord,
    carry: ModelState,
    stepper: TimeStepper,
    n: int,
    *,
    async_compile: bool = False,
) -> ModelState:
    """
    Run one compiled chunk of ``n`` steps (donating the carry).

    Description
    -----------
    The framework-level entry the run loop calls: AOT
    ``lower().compile()`` on first use of a (record, length,
    structure) key — replacing the first-step compile-timing hack
    and recording compile seconds + the executable's memory
    analysis — then the cached executable thereafter. Compiled
    lengths are {C, 1} only (``Model.advance`` plans greedy
    C-chunks plus length-1 tails). The carry's buffers are DONATED:
    the caller must drop its reference; the stepper is passed
    non-donated.

    When ``async_compile`` is set and a miss's natural unroll is > 1
    (so ``n > 1`` too), the miss serves a cheap ``force_unroll=1``
    length-C executable via a :class:`_TwoTier` holder while the full
    executable compiles on a daemon thread, then swaps the cache entry
    to the full executable at a later chunk boundary — so steady state
    is the single-tier baseline with no holder indirection and the
    compile log records the full tier's numbers. Compiled lengths stay
    {C, 1}; the length-1 tails always take the synchronous path.

    Parameters
    ----------
    record : AssemblyRecord
        The hashable static assembly bundle (jit-cache key).
    carry : ModelState
        The donated carry.
    stepper : TimeStepper
        The loop-invariant, non-donated stepper pytree.
    n : int
        The static chunk length.
    async_compile : bool, optional
        Serve a cheap unroll-1 tier while the full-unroll executable
        compiles off-thread (default: False).

    Returns
    -------
    ModelState
        The advanced carry.
    """
    key = _chunk_key(record, carry, stepper, n)
    entry = _CHUNK_EXECUTABLES.get(key)
    if entry is not None:
        if type(entry) is _TwoTier:
            full = entry.settled()   # raises any background error
            if full is None:
                return entry.serve_cheap(carry, stepper)
            with _CHUNK_LOCK:        # swap out the holder indirection
                _CHUNK_EXECUTABLES[key] = full
                _CHUNK_COMPILE_LOG[key] = entry.compile_log()
            return full(carry, stepper)
        return entry(carry, stepper)
    natural = max(1, min(int(stepper.scan_unroll), n))
    if not async_compile or natural <= 1 or n <= 1:
        # legacy synchronous single-tier compile
        started = time.perf_counter()
        compiled = _compile_chunk(record, carry, stepper, n)
        seconds = time.perf_counter() - started
        memory = _memory_analysis(compiled)
        with _CHUNK_LOCK:
            _CHUNK_EXECUTABLES[key] = compiled
            _CHUNK_COMPILE_LOG[key] = (seconds, memory)
        return compiled(carry, stepper)
    # async two-tier: lower the FULL chunk on THIS thread while the
    # carry is still live (a Lowered holds HLO, not buffers), then
    # compile the cheap unroll-1 tier and serve it while the full one
    # compiles off-thread. The log entry is replaced with the full
    # tier's numbers at swap time.
    lowered_full = _lower_chunk(record, carry, stepper, n)
    started = time.perf_counter()
    cheap = _compile_chunk(record, carry, stepper, n, force_unroll=1)
    seconds = time.perf_counter() - started
    memory = _memory_analysis(cheap)
    holder = _TwoTier(cheap, lowered_full)
    with _CHUNK_LOCK:
        _CHUNK_EXECUTABLES[key] = holder
        _CHUNK_COMPILE_LOG[key] = (seconds, memory)
    return holder.serve_cheap(carry, stepper)


def chunk_cache_size() -> int:
    """Return the number of compiled chunk executables (tests)."""
    return len(_CHUNK_EXECUTABLES)


# ================================================================
#  ParameterView — the host read surface (D2.4 tier 1)
# ================================================================
class ParameterView(Mapping):

    """
    Read-only host mapping over the binding table, live reads.

    Description
    -----------
    Values are read live from the carry's modules (and the model's
    stepper) on every access; Ramp-valued slots return the
    `TimeDependent` object itself, never a silently-evaluated value
    — evaluate via :meth:`at_time`. Unknown names raise the hinted
    ``MissingParameterError``.

    Parameters
    ----------
    model : Model
        The owning model (live carry reads go through it).
    """

    def __init__(self, model: Model) -> None:
        """Bind the view to the model; see the class docstring."""
        self._model = model

    def _live(self) -> Mapping[str, object]:
        """Build the host view over the CURRENT live providers."""
        model = self._model
        return model._binding_table.host_view(  # noqa: SLF001 — owning model
            model._carry.modules, model._stepper)  # noqa: SLF001

    def __getitem__(self, name: str) -> object:
        """Live leaf read of one bound name (Ramps returned raw)."""
        return self._live()[name]

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is bound."""
        return name in self._live()

    def __iter__(self) -> Iterator[str]:
        """Iterate the bound names, binding order."""
        return iter(self._live())

    def __len__(self) -> int:
        """Return the number of bound names."""
        return len(self._live())

    def at_time(self, t: float) -> dict[str, object]:
        """
        Evaluate every TimeDependent slot explicitly at ``t``.

        Description
        -----------
        The D2.4 rule: no implicit clock evaluation host-side —
        Ramp-valued reads must spell the time.

        Parameters
        ----------
        t : float
            The (signed) model time to evaluate at, in seconds.

        Returns
        -------
        dict[str, object]
            All bound parameters, TimeDependent values resolved.
        """
        live = self._live()
        return {name: resolve_at(live[name], t) for name in live}

    def info(self, name: str) -> ParameterDeclaration:
        """
        Return the declaration (units/doc/provider) of one name.

        Parameters
        ----------
        name : str
            The dotted parameter name.

        Returns
        -------
        ParameterDeclaration
            The provider's declaration row.
        """
        table = self._model._binding_table  # noqa: SLF001 — owning model
        return table[name].declaration

    def __repr__(self) -> str:
        """Summary repr listing the bound names."""
        return f"ParameterView({', '.join(self)})"


# ================================================================
#  DiagnosticsNamespace and BoundDiagnostic
# ================================================================
class BoundDiagnostic:

    """
    A package diagnostic bound to a model (lazy parameter reads).

    Description
    -----------
    Wraps the package-level pure function ``fn(state, params)`` with
    the model's live :class:`ParameterView` — LAZILY: a hinted
    ``MissingParameterError`` fires only when an absent-provider
    diagnostic is actually called (the view raises on the read).
    [Spec concretization: the pure-function signature ``(state,
    params) -> Field | scalar`` — confirm at first package use.]

    Parameters
    ----------
    name : str
        The diagnostic's attribute name.
    fn : Callable
        The pure package-level function ``(state, params)``.
    model : Model
        The owning model.
    """

    def __init__(
        self, name: str, fn: Callable, model: Model,
    ) -> None:
        """Bind the diagnostic; see the class docstring."""
        self._name = name
        self._fn = fn
        self._model = model

    def __call__(
        self, state: VectorField | None = None,
    ) -> object:
        """
        Evaluate on ``state`` (default: the current carry state).

        Parameters
        ----------
        state : VectorField | None, optional
            The state to evaluate on; ``None`` reads the carry
            (default: None).

        Returns
        -------
        object
            The diagnostic value (Field or scalar).
        """
        if state is None:
            state = self._model._carry.state  # noqa: SLF001 — owning model
        return self._fn(state, self._model.parameters)

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return f"<bound diagnostic {self._name!r}>"


class DiagnosticsNamespace:

    """
    ``model.diagnostics`` — bound package diagnostic expressions.

    Description
    -----------
    Diagnostics are contributed by modules through a duck-typed
    ``diagnostics`` mapping attribute (``{name: fn}`` with pure
    ``fn(state, params)`` — the D1.3 commitment-4 channel; the exact
    Module hook spelling is owed to module.md). Attribute access
    returns a :class:`BoundDiagnostic`; parameter resolution is lazy
    (call time), never at binding.

    Parameters
    ----------
    model : Model
        The owning model.
    """

    def __init__(self, model: Model) -> None:
        """Bind the namespace to the model."""
        self._model = model

    def _entries(self) -> dict[str, Callable]:
        """Collect the module-contributed diagnostics (live)."""
        entries: dict[str, Callable] = {}
        owners: dict[str, str] = {}
        modules = self._model._carry.modules  # noqa: SLF001 — owning model
        for module in modules:
            mapping = getattr(module, "diagnostics", None)
            if not isinstance(mapping, Mapping):
                continue
            owner = type(module).__name__
            for name, fn in mapping.items():
                if name in entries:
                    raise AssemblyError(
                        f"diagnostic {name!r} is contributed by "
                        f"both {owners[name]} and {owner}; "
                        "diagnostic names are unique per model")
                entries[name] = fn
                owners[name] = owner
        return entries

    def __getattr__(self, name: str) -> BoundDiagnostic:
        """Return the named diagnostic bound to the model."""
        if name.startswith("_"):
            raise AttributeError(name)
        entries = self._entries()
        if name not in entries:
            available = ", ".join(sorted(entries)) or "none"
            raise AttributeError(
                f"no diagnostic named {name!r}; available "
                f"diagnostics: {available}")
        return BoundDiagnostic(name, entries[name], self._model)

    def __dir__(self) -> list[str]:
        """List the bound diagnostic names (tab completion)."""
        return sorted(set(super().__dir__()) | set(self._entries()))

    def __repr__(self) -> str:
        """Summary repr listing the bound diagnostics."""
        return (f"<diagnostics: "
                f"{', '.join(sorted(self._entries())) or 'none'}>")


# ================================================================
#  propagator: wrt-target resolution and the identity splice
# ================================================================
class _WrtTarget(NamedTuple):

    """
    One resolved ``propagator`` differentiation target.

    Description
    -----------
    Where a ``theta`` leaf splices into the ``(carry, stepper)`` tree.
    ``kind`` is ``"module"`` (a module parameter, ``slot``/``attr``),
    ``"stepper"`` (the time-stepper leaf ``attr``), or ``"field"`` (a
    PROGNOSTIC initial condition, spliced at ``carry.state[name].
    storage``).

    Parameters
    ----------
    kind : str
        One of ``"module"``, ``"stepper"``, ``"field"``.
    slot : int
        The module tuple index (``-1`` for stepper/field targets).
    attr : str
        The provider's dynamic-leaf attribute ("" for field targets).
    name : str
        The wrt name (a dotted parameter name or a field name).
    """

    kind: str
    slot: int
    attr: str
    name: str


def _live_leaf(
    target: _WrtTarget, carry: ModelState, stepper: TimeStepper,
) -> object:
    """Return one target's live leaf in ``(carry, stepper)``."""
    if target.kind == "field":
        return carry.state[target.name].storage
    if target.kind == "stepper":
        return getattr(stepper, target.attr)
    return getattr(carry.modules[target.slot], target.attr)


def _splice_targets(
    carry: ModelState,
    stepper: TimeStepper,
    targets: tuple[_WrtTarget, ...],
    theta: tuple[object, ...],
) -> tuple[ModelState, TimeStepper]:
    """
    Splice ``theta`` leaves into ``(carry, stepper)`` by identity.

    Description
    -----------
    Generalizes the ``test_model_autodiff`` pattern to any mix of
    module-parameter, stepper and initial-field targets: each target's
    live leaf is located by object identity in the flattened tree and
    replaced with the traced ``theta`` value, so reverse-mode autodiff
    flows straight into the carry ``_chunk_body`` scans over. The
    treedef is untouched, so the substitution never disturbs the
    scan's fixed-point carry structure.
    """
    leaves, treedef = jax.tree_util.tree_flatten((carry, stepper))
    new_leaves = list(leaves)
    for target, value in zip(targets, theta, strict=True):
        live = _live_leaf(target, carry, stepper)
        matches = [i for i, ref in enumerate(leaves) if ref is live]
        if len(matches) != 1:  # pragma: no cover — structural invariant
            raise RuntimeError(
                f"propagator could not locate a unique leaf for wrt "
                f"target {target.name!r} ({len(matches)} matches); the "
                "carry structure changed underneath the resolver")
        new_leaves[matches[0]] = value
    carry2, stepper2 = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return carry2, stepper2


# ================================================================
#  Model — the composition root and host driver
# ================================================================
class Model:

    """
    Grid + modules + stepper; owns assembly, the carry, the run loop.

    Description
    -----------
    Host-side driver — nothing traces through it; a Model-as-pytree
    would double-flatten the carry (the D2 aliasing bug by
    construction). The constructor IS the nine-step assembly
    (``assemble()``) plus step 8, the carry allocation: PROGNOSTIC
    and DIAGNOSTIC fields zero-valued, AUXILIARY fields through
    ``remat_table.materialize`` (defaults evaluated with the
    assembly-time parameters — the one shared code path), the
    stepper's fresh ``init``, a float64-anchored clock, and the
    cleared panic ledger. No ``setup()`` exists.

    The step-7 ``ReshardingReport`` is layout-only; if a pre-built
    leaf ever needs re-homing into the negotiated layout, the
    sanctioned path is the true-shape re-pad (what ``set_fields`` /
    ``set_state`` do), never a bare ``device_put``.

    Parameters
    ----------
    grid : Grid
        The assembly root (frozen after step 7).
    modules : tuple
        The module tuple (``fr.Module`` instances).
    time_stepper : TimeStepper
        REQUIRED, no default — no physics-free dt exists.
    io : tuple, optional
        Standing output config; bound at run start, never at
        assembly (wave 5). Rejects ``Snapshots`` (run-config only)
        (default: ()).
    state_type : type | None, optional
        The State vocabulary class; ``None`` reads the
        module-supplied one, falling back to ``VectorField``
        (default: None).
    name : str | None, optional
        Report/log attribution (default: None).
    chunk_size : int, optional
        advance()'s host-sync granularity C — compiled chunk
        lengths are {C, 1} only. [Spec concretization of the
        max_chunk auto ~256 knob at the Model level; ``run()``'s
        trigger-driven plan subdivides it at wave 5]
        (default: 256).
    async_chunk_compile : bool, optional
        When True, a first advance on a not-yet-compiled configuration
        serves a cheap unroll-1 chunk executable while the full-unroll
        one compiles in a background thread, cutting time-to-first-
        step; the cache entry swaps to the full executable at a later
        chunk boundary. Steady-state results and performance are
        unchanged (default: False).
    """

    def __init__(
        self,
        *,
        grid: Grid,
        modules: tuple,
        time_stepper: TimeStepper,
        io: tuple = (),
        state_type: type | None = None,
        name: str | None = None,
        chunk_size: int = _DEFAULT_CHUNK,
        async_chunk_compile: bool = False,
        term_filter: Callable | None = None,
    ) -> None:
        """Assemble (steps 1-7, 9) and allocate the carry (step 8)."""
        if isinstance(chunk_size, bool) or not isinstance(
                chunk_size, int) or chunk_size < 1:
            raise ValueError(
                f"chunk_size must be an int >= 1, got {chunk_size!r}")
        if not isinstance(async_chunk_compile, bool):
            raise ValueError(  # noqa: TRY004 — mirror chunk_size check
                "async_chunk_compile must be a bool, got "
                f"{async_chunk_compile!r}")
        for stream in io:
            if isinstance(stream, Snapshots):
                raise IOCollisionError(
                    "io= rejects Snapshots — snapshots are "
                    "run-config only (one resume path, never two); "
                    "pass them to run(snapshots=...)")
        modules = tuple(modules)
        self._artifacts: AssemblyArtifacts = assemble(
            grid=grid, modules=modules, time_stepper=time_stepper,
            state_type=state_type, name=name, term_filter=term_filter)
        self._grid = grid
        self._stepper = time_stepper
        self._name = name
        self._io = tuple(io)
        self._chunk_size = chunk_size
        self._async_chunk_compile = async_chunk_compile
        # -- step 8: allocate the carry ---------------------------
        state = self._allocate_state(modules)
        # built once: a cached zero-template BUILDER (structure
        # only, one stable jitted trace) reused by every re-warm —
        # caching the built template itself would keep one zero
        # PROGNOSTIC vector of full padded fields alive for the
        # model's lifetime, while a fresh composer build would pay
        # the grid's pad-and-sync store path per update_parameters
        # call (a per-call retrace under multi-device)
        template = self._artifacts.composer.tendency_template()
        self._template_build: Callable[[], VectorField] | None = (
            _template_builder(template) if template is not None
            else None)
        del template  # the builder's product is init-consumed only
        stepper_state: StepperState = self._fresh_stepper_state()
        carry = ModelState(
            state=state, modules=modules,
            stepper_state=stepper_state, clock=Clock(),
            panic=_fresh_panic())
        self._carry = _canonicalize(carry)
        # armed once per fresh/committed carry: the first advance
        # defragments the pool before dispatching a chunk (GPU only)
        self._defrag_pending = True
        # host mirrors (cheap reads; no device sync)
        self._panicked = False
        self._panic_it: int | None = None
        self._host_elapsed = np.float64(0.0)
        self._debug_carry: ModelState | None = None
        self._debug_steps = 0
        # memoized jitted read-only tendency executables (2.8): keyed
        # by (kept-term-key frozenset, constraints) so repeated matvec
        # calls with one filter hit ONE compiled entry
        self._tendency_cache: dict[tuple, Callable] = {}

    # ================================================================
    #  Allocation helpers (step 8)
    # ================================================================
    def _zero_field(self, record: FieldRecord) -> ScalarField:
        """Zero-valued field of one record (allocation + reset)."""
        return self._grid.create_field(
            record.space, metadata=record.metadata)

    def _allocate_state(
        self, modules: tuple,
    ) -> VectorField | None:
        """Build the step-8 state: PROG/DIAG zeros, AUX defaults."""
        table = self._artifacts.field_table
        if not len(table):
            return None
        aux = self._artifacts.remat_table.materialize(
            modules, self._grid)
        fields = {
            record.name: (aux[record.name]
                          if record.lifecycle is Lifecycle.AUXILIARY
                          else self._zero_field(record))
            for record in table}
        state_type = self._artifacts.record.state_type
        return state_type(fields)

    def _commit(self, carry: ModelState) -> None:
        """Canonicalize and swap in a host-built carry; re-arm defrag."""
        self._carry = _canonicalize(carry)
        # the new carry re-fragments the pool (fresh buffers + the
        # donated old ones freed): re-arm the one-time pre-chunk defrag
        self._defrag_pending = True

    def _should_defragment(self) -> bool:
        """
        Whether the one-time carry defrag applies (GPU, not disabled).

        Description
        -----------
        The predicate half of the defrag, split from
        :meth:`_defragment_carry` so a test can force the repack on a
        non-GPU backend by monkeypatching it. The kill switch
        ``FRIDOM_DISABLE_DEFRAG`` is read at call time; the backend
        gate keeps the repack off cpu/tpu, where the pool is not the
        contiguous-arena BFC that motivates it.
        """
        if not _defrag_enabled():
            return False
        return jax.default_backend() == "gpu"

    def _defragment_carry(self) -> None:
        """
        Sequentially repack the carry's big device buffers (one-time).

        Description
        -----------
        The advective chunk allocates its XLA temp workspace as a
        SINGLE CONTIGUOUS arena (tens of GiB at production sizes); BFC
        cannot place that arena in a pool fragmented by the
        construction / ``set_fields`` churn even when the free bytes
        suffice. Copying each big leaf in turn — ``jnp.copy`` ->
        ``block_until_ready`` -> ``delete`` the original — migrates the
        live cubes into the small holes and coalesces the freed
        high-address stretch into one region large enough for the
        arena. The sequential copy-then-free is load-bearing: a bulk
        ``tree_map`` copy allocates the whole second set before freeing
        any original and so does NOT defragment. ``jnp.copy`` preserves
        each leaf's sharding, so the repack is value- and layout-
        preserving; small / scalar / non-array leaves are left as-is.
        """
        leaves, treedef = jax.tree_util.tree_flatten(self._carry)
        # id-seen guard: an aliased leaf is copied once and its buffer
        # freed once (never delete()d twice)
        copied: dict[int, jax.Array] = {}
        repacked: list[object] = []
        for leaf in leaves:
            if not isinstance(leaf, jax.Array) or \
                    leaf.nbytes < _DEFRAG_MIN_BYTES:
                repacked.append(leaf)
                continue
            key = id(leaf)
            seen = copied.get(key)
            if seen is not None:
                repacked.append(seen)
                continue
            new = jnp.copy(leaf)
            new.block_until_ready()
            leaf.delete()
            copied[key] = new
            repacked.append(new)
        self._carry = jax.tree_util.tree_unflatten(treedef, repacked)

    @property
    def _binding_table(self) -> Any:
        """The frozen parameter binding table."""
        return self._artifacts.binding_table

    # ================================================================
    #  Read surface (host, chunk boundaries)
    # ================================================================
    @property
    def state(self) -> VectorField:
        """
        The carry's state vector, read-only (copy-on-read).

        Description
        -----------
        DECISION (open thread 2, flagged): the returned state's
        leaves are COPIED — the carry's own buffers are donated to
        the next ``advance()``, so a live view would reference
        deleted buffers after the next chunk. Copy cost is one
        device-to-device pass per read.
        """
        state = self._carry.state
        if state is None:
            raise AttributeError(
                "this composition declares no fields (legal, "
                "CS-13), so there is no state vector to read")
        return _copy_leaves(state)

    @property
    def parameters(self) -> ParameterView:
        """Read-only live mapping over the binding table."""
        return ParameterView(self)

    @property
    def diagnostics(self) -> DiagnosticsNamespace:
        """Bound package diagnostics (lazy parameter resolution)."""
        return DiagnosticsNamespace(self)

    def module(
        self,
        module_type: type[M],
        *,
        name: str | None = None,
    ) -> M:
        """
        Typed lookup of a live carry module (unpublished knobs).

        Parameters
        ----------
        module_type : type
            The module class to match (isinstance).
        name : str | None, optional
            Disambiguator matched against a module's ``name``
            attribute, when present (default: None).

        Returns
        -------
        Module
            The single matching live module.

        Raises
        ------
        LookupError
            No match, or an ambiguous match (candidates listed).
        """
        matches = [module for module in self._carry.modules
                   if isinstance(module, module_type)]
        if name is not None:
            matches = [module for module in matches
                       if getattr(module, "name", None) == name]
        if not matches:
            present = ", ".join(
                type(m).__name__ for m in self._carry.modules)
            raise LookupError(
                f"no live module matches {module_type.__name__!r}"
                f"{f' with name={name!r}' if name else ''}; "
                f"assembled modules: {present or 'none'}")
        if len(matches) > 1:
            candidates = ", ".join(
                type(m).__name__ for m in matches)
            raise LookupError(
                f"ambiguous module lookup {module_type.__name__!r}:"
                f" candidates {candidates}; disambiguate with "
                "name= or a more specific type")
        return matches[0]

    @property
    def modules(self) -> tuple:
        """The live carry modules, in assembly order (read-only).

        The plural of :meth:`module`: a host-side read for consumers
        that must inspect the whole module list rather than look one
        up by type (``fr.model.require_linear_operator``). Assembled
        modules are read-only from the host — change leaf values
        through ``update_parameters``.
        """
        return tuple(self._carry.modules)

    @property
    def grid(self) -> Grid:
        """The (frozen) grid this model was assembled on."""
        return self._grid

    @property
    def clock(self) -> Clock:
        """
        Host read of the carry clock (copy-on-read).

        Description
        -----------
        [Spec addition: needed by the Session's CS-18 clock
        assertion and progress reporting.]
        """
        return _copy_leaves(self._carry.clock)

    @property
    def name(self) -> str | None:
        """Report/log attribution (two models, one process)."""
        return self._name

    @property
    def panicked(self) -> bool:
        """Cheap host-readable panic flag (CS-6; no device sync)."""
        return self._panicked

    @property
    def carry(self) -> ModelState:
        """
        Opaque in-memory carry snapshot (CS-12; copy-on-read).

        Description
        -----------
        The disk-free twin of :meth:`snapshot`: leaves are COPIED
        so the snapshot survives later donations. Read-only; the
        sanctioned setter is a leave-open.
        """
        return _copy_leaves(self._carry)

    @property
    def report(self) -> AssemblyReport:
        """The assembly report (run-start addendum: wave 5)."""
        return self._artifacts.report

    @property
    def fingerprint(self) -> Fingerprint:
        """The restart fingerprint (structure, never leaves)."""
        return self._artifacts.fingerprint

    @property
    def field_table(self) -> FieldTable:
        """The resolved field table (names, spaces, lifecycles)."""
        return self._artifacts.field_table

    # ================================================================
    #  State factory (host; section 6.1)
    # ================================================================
    def blank_state(
        self,
        **inits: Callable | np.ndarray | ScalarField | complex,
    ) -> VectorField:
        r"""
        Build a fresh PROGNOSTIC state, optionally initialized.

        Description
        -----------
        The State factory every IC recipe and transform builds on
        ([`04_run_loop_io.md`] §6.1): the PROGNOSTIC subset of the
        field table, each component a zero-valued field born on the
        grid (sharded like the carry). Thin sugar over the field
        table — one :meth:`state_space` + ``grid.create_field`` per
        PROGNOSTIC component, packed into the model's state class.

        Per-component keywords initialize named components through
        ``VectorField.set`` — a callable of the physical
        coordinates, a space-checked field, a scalar, or a
        true-shape array; every unnamed component stays zero:

        .. code-block:: python

            jet = model.blank_state(
                u=lambda x, y: jnp.exp(-(y - 0.5) ** 2))

        Parameters
        ----------
        **inits : Callable | np.ndarray | ScalarField | complex, optional
            Initial values for PROGNOSTIC components, by name.

        Returns
        -------
        VectorField
            A fresh state carrying the PROGNOSTIC components, in
            declaration order.

        Raises
        ------
        ValueError
            If this composition declares no PROGNOSTIC field (legal,
            CS-13), so there is no state to build, or an ``inits``
            name is not a PROGNOSTIC component.
        """
        table = self._artifacts.field_table
        names = table.prognostic
        if not names:
            raise ValueError(
                "this composition declares no PROGNOSTIC field "
                "(legal, CS-13), so there is no blank state to build")
        unknown = tuple(name for name in inits if name not in names)
        if unknown:
            raise ValueError(
                f"blank_state: unknown or non-PROGNOSTIC field(s) "
                f"{unknown}; the PROGNOSTIC components are "
                f"{tuple(names)} (AUXILIARY writes go through "
                "set_aux)")
        state_type = self._artifacts.record.state_type
        state = state_type(
            {name: self._zero_field(table[name]) for name in names})
        return state.set(**inits) if inits else state

    def state_space(self, name: str) -> TensorProductSpace:
        r"""
        Return the function space of a declared component.

        Description
        -----------
        The space an IC recipe feeds ``grid.create_field`` to build
        one component by hand ([`04_run_loop_io.md`] §6.1). Thin
        sugar over ``model.field_table[name].space``.

        Parameters
        ----------
        name : str
            A declared field name.

        Returns
        -------
        TensorProductSpace
            The resolved bare space of ``name``.

        Raises
        ------
        MissingFieldError
            If no module declares ``name``.
        """
        return self._artifacts.field_table[name].space

    @property
    def chunk_size(self) -> int:
        """advance()'s host-sync granularity C."""
        return self._chunk_size

    @property
    def async_chunk_compile(self) -> bool:
        """Whether the first advance serves a cheap async tier."""
        return self._async_chunk_compile

    def __repr__(self) -> str:
        """Return the assembly report's header."""
        return self._artifacts.report.header

    # ================================================================
    #  Lifecycle mutators (host, chunk boundaries; section 6.5)
    # ================================================================
    def _rehome(
        self,
        incumbent: ScalarField,
        value: Callable | np.ndarray | ScalarField,
        label: str,
    ) -> ScalarField:
        """
        Re-home one incoming value onto the declared component.

        Description
        -----------
        Thin delegation to the shared
        :func:`fridom.spatial.fields.vector_field.rehome_component`
        coercion (also behind ``VectorField.set``): the sanctioned
        re-home is the true-shape re-pad, never a bare
        ``device_put`` (CS-8); the incumbent's metadata is kept.
        """
        return rehome_component(incumbent, value, label=label)

    def set_fields(
        self,
        **fields: Callable | np.ndarray | ScalarField,
    ) -> None:
        """
        Write user ICs onto PROGNOSTIC components; clears panic.

        Description
        -----------
        Non-PROGNOSTIC names error, pointing at :meth:`set_aux`;
        incoming values are re-homed per the decomposition (the
        true-shape re-pad). Clears the panic flag (a resume path).

        Parameters
        ----------
        **fields : Callable | np.ndarray | ScalarField
            Per-component ICs: a function of the physical
            coordinates, a true-shape array, or a field.
        """
        table = self._artifacts.field_table
        prognostic = set(table.prognostic)
        for name in fields:
            if name not in table.names:
                raise ValueError(
                    f"set_fields: unknown field {name!r}; declared "
                    f"fields: {table.names}")
            if name not in prognostic:
                raise ValueError(
                    f"set_fields: {name!r} is "
                    f"{table[name].lifecycle.name}, not PROGNOSTIC;"
                    " host writes to consented AUXILIARY/DIAGNOSTIC"
                    " components go through set_aux")
        state = self._carry.state
        updates = {
            name: self._rehome(state[name], value,
                               f"set_fields({name})")
            for name, value in fields.items()}
        new_state = self._derive_initial_fields(
            state.replace(**updates), frozenset(fields))
        self._commit(self._carry.replace(
            state=new_state, panic=_fresh_panic()))
        self._panicked = False
        self._panic_it = None

    def _derive_initial_fields(
        self, state: VectorField, provided: frozenset[str],
    ) -> VectorField:
        """
        Apply module-derived IC updates for slaved PROGNOSTIC fields.

        Description
        -----------
        After :meth:`set_fields` writes the user's ICs, each bound
        module may seed the PROGNOSTIC components it owns that are
        slaved to a just-set master (``Module.derive_initial_fields``,
        the default a no-op — e.g. ``hy.SplitExplicitFreeSurface``
        derives its barotropic transports ``U, V`` from the velocity
        IC). The returned fields are re-homed exactly like the user's
        own values; a module never overrides a component the user set
        in the same call (it reads ``provided`` to gate). Host-side,
        outside any jit trace.

        Parameters
        ----------
        state : VectorField
            The state after the user's fields are applied.
        provided : frozenset[str]
            The component names the user set in this ``set_fields``
            call.

        Returns
        -------
        VectorField
            The state with the derived components applied (``state``
            unchanged when no module derives anything).
        """
        derived: dict[str, ScalarField] = {}
        for module in self._carry.modules:
            derived.update(module.derive_initial_fields(state, provided))
        if not derived:
            return state
        rehomed = {
            name: self._rehome(state[name], value,
                               f"set_fields(derived {name})")
            for name, value in derived.items()}
        return state.replace(**rehomed)

    def set_state(self, state: VectorField) -> None:
        """
        Overwrite the PROGNOSTIC subset from a state vector.

        Description
        -----------
        Missing PROGNOSTIC components are left untouched (amended
        V-C12); AUXILIARY/DIAGNOSTIC components in the input are
        ignored with a debug log. Re-homes; clears the panic flag.

        Parameters
        ----------
        state : VectorField
            The (possibly partial) source state.
        """
        table = self._artifacts.field_table
        prognostic = set(table.prognostic)
        extras = tuple(name for name in state.component_names
                       if name not in prognostic)
        if extras:
            _log.debug(
                "set_state: ignoring non-PROGNOSTIC components %s",
                extras)
        incumbent = self._carry.state
        updates = {
            name: self._rehome(incumbent[name], state[name],
                               f"set_state({name})")
            for name in table.prognostic if name in state}
        self._commit(self._carry.replace(
            state=incumbent.replace(**updates),
            panic=_fresh_panic()))
        self._panicked = False
        self._panic_it = None

    def set_aux(
        self,
        *,
        rewarm: bool = False,
        **fields: ScalarField | np.ndarray,
    ) -> None:
        """
        Write consented components (V-C1, the exchange path).

        Description
        -----------
        AUXILIARY or DIAGNOSTIC components with
        ``host_writable=True`` only — the coupler exchange path and
        the S6-accumulator reset. Re-homes without a host round-trip
        (CS-8). No warm-up re-ramp by default (exchange data is
        forcing). Leaves the panic flag UNTOUCHED (not a resume
        path).

        Parameters
        ----------
        rewarm : bool, optional
            Re-init the stepper's warm-up after the write
            (default: False).
        **fields : ScalarField | np.ndarray
            The consented components to overwrite.
        """
        table = self._artifacts.field_table
        writable = set(table.host_writable)
        for name in fields:
            if name not in table.names:
                raise ValueError(
                    f"set_aux: unknown field {name!r}; declared "
                    f"fields: {table.names}")
            if name not in writable:
                raise ValueError(
                    f"set_aux: {name!r} is not host-writable; host"
                    " writes require the owner's declaration "
                    "consent (host_writable=True on the "
                    "FieldDeclaration). Consented components: "
                    f"{table.host_writable or 'none'}")
        state = self._carry.state
        updates = {
            name: self._rehome(state[name], value,
                               f"set_aux({name})")
            for name, value in fields.items()}
        carry = self._carry.replace(state=state.replace(**updates))
        if rewarm:
            carry = carry.replace(
                stepper_state=self._fresh_stepper_state())
        self._commit(carry)

    def _fresh_stepper_state(self) -> StepperState:
        """Return a fresh ``stepper.init`` product (the re-warm)."""
        build = self._template_build
        return self._stepper.init(build()) if build is not None else ()

    @staticmethod
    def _fresh_clone(provider: object) -> object:
        """
        Shallow copy with a fresh identity (an unbound provider).

        Description
        -----------
        The bind-once/freeze guard is keyed on object identity, so a
        fresh shallow copy is unbound and may be handed to a child
        assembly. Leaf values are shared (arrays are immutable).
        """
        clone = object.__new__(type(provider))
        clone.__dict__.update(provider.__dict__)
        return clone

    @staticmethod
    def _replace_leaf(
        provider: object, attr: str, value: object,
    ) -> object:
        """Functional single-leaf write on a frozen provider."""
        old = getattr(provider, attr)
        if isinstance(old, jax.Array) and isinstance(
                value, int | float | complex | np.number
                | np.ndarray | jax.Array):
            value = jnp.asarray(value, dtype=old.dtype)
        clone = Model._fresh_clone(provider)
        object.__setattr__(clone, attr, value)
        return clone

    def update_parameters(
        self,
        updates: Mapping[str, object],
        *,
        rewarm: bool = True,
    ) -> None:
        """
        Write bound parameter leaves functionally onto the carry.

        Description
        -----------
        Names resolve through the binding table (including
        ``fr.params.TIME_STEP`` — the stepper is a provider; the
        sign flip replaces ``run_backward``). Structure preservation
        is validated (same treedef -> no recompile; a scalar<->Ramp
        swap is a treedef change -> re-assemble). Owner-derived
        AUXILIARY fields of changed owners are re-materialized
        through the ONE shared path (host-writable components
        exempt, CS-2). Re-ramps the multistep warm-up by default
        (old-physics buffers; dt sign flips require it). Does NOT
        clear the panic flag (changing nu after a NaN is not a
        resume path).

        Parameters
        ----------
        updates : Mapping[str, object]
            ``dotted name -> new leaf value``.
        rewarm : bool, optional
            Re-init the stepper's warm-up (default: True).
        """
        table = self._binding_table
        modules = list(self._carry.modules)
        stepper = self._stepper
        changed_owners: set[int] = set()
        for name, value in dict(updates).items():
            entry = table[name]  # hinted MissingParameterError
            if entry.slot is None:
                raise AssemblyError(
                    f"parameter {name!r} is an identity-defaulted "
                    "constant (no provider owns a leaf); provide it"
                    " from a module, or re-assemble")
            if entry.slot == "stepper":
                stepper = self._replace_leaf(stepper, entry.attr,
                                             value)
            else:
                modules[entry.slot] = self._replace_leaf(
                    modules[entry.slot], entry.attr, value)
                changed_owners.add(entry.slot)
        new_modules = tuple(modules)
        old_def = jax.tree_util.tree_structure(
            (self._carry.modules, self._stepper))
        new_def = jax.tree_util.tree_structure(
            (new_modules, stepper))
        if old_def != new_def:
            raise AssemblyError(
                "update_parameters changed a value SPEC (e.g. "
                "scalar <-> Ramp) — a treedef change; post-assembly"
                " writes preserve structure, re-assemble instead")
        self._stepper = stepper
        carry = self._carry.replace(modules=new_modules)
        if changed_owners:
            remat = self._artifacts.remat_table.materialize(
                new_modules, self._grid,
                owners=frozenset(changed_owners))
            if remat:
                carry = carry.replace(
                    state=carry.state.replace(**remat))
        if rewarm:
            carry = carry.replace(
                stepper_state=self._fresh_stepper_state())
        self._commit(carry)  # panic deliberately KEPT

    def reset(self) -> None:
        """
        Re-warm, re-zero, restart the clock (the OptimalBalance op).

        Description
        -----------
        PROGNOSTIC and DIAGNOSTIC components return to zeros
        (including consented DIAGNOSTIC accumulators); AUXILIARY is
        NEVER touched (the three-operation matrix, 02_rules). The
        stepper re-inits (re-warm), the clock resets (which is what
        restarts a Ramp leg), the panic flag clears. Invariant:
        ``reset(); set_state(z)`` == fresh assembly +
        ``set_state(z)``, bitwise.
        """
        table = self._artifacts.field_table
        carry = self._carry
        state = carry.state
        if state is not None:
            zeros = {
                record.name: self._zero_field(record)
                for record in table
                if record.lifecycle is not Lifecycle.AUXILIARY}
            if zeros:
                state = state.replace(**zeros)
        self._commit(ModelState(
            state=state, modules=carry.modules,
            stepper_state=self._fresh_stepper_state(),
            clock=carry.clock.reset(), panic=_fresh_panic()))
        self._panicked = False
        self._panic_it = None
        self._host_elapsed = np.float64(0.0)

    # ================================================================
    #  The differentiable run surface (propagator)
    # ================================================================
    def propagator(
        self,
        *,
        wrt: Sequence[str] = (),
        steps: int,
        remat: bool | None = None,
    ) -> Callable[..., ModelState]:
        r"""
        Build a pure, differentiable ``theta -> ModelState`` run.

        Description
        -----------
        The supported spelling for reverse-mode autodiff through a
        model run (the private ``_chunk_body`` recipe of
        ``design/research/jax_grad_run_investigation.md``, made
        public). Returns a callable ``run(theta, state=None)`` that
        advances ``steps`` steps from the committed carry (snapshotted
        at build) and returns the final ``ModelState`` — carry
        donation OFF and no host panic sync, so the caller may wrap it
        directly in ``jax.jit`` / ``jax.grad`` / ``jax.value_and_grad``
        (a NaN gradient is a bug, not an expected outcome). The panic
        pair rides the returned carry; the returned state is
        ``result.state``.

        ``theta`` is a tuple aligned with ``wrt``: each value is
        spliced by identity into its target leaf (a module parameter, a
        stepper leaf such as ``stepper.dt``, or a PROGNOSTIC initial
        field), so ``jax.grad(lambda th: loss(run(th)))(theta0)`` is
        the gradient of the loss with respect to those leaves. The
        stepper starts from a FRESH warm-up (``_fresh_stepper_state``),
        so the gradient includes the multistep ramp; ``state=None``
        integrates the committed state, and a supplied ``State`` splices
        its PROGNOSTIC components in first.

        A ``wrt`` name is refused, host-side at build, with a taught
        error when it cannot yield a correct gradient: an unknown name,
        an identity-defaulted constant (no leaf to differentiate), a
        parameter frozen into ``exp(L dt)`` by an exponential stepper
        (silently stale), or a parameter whose owner materializes
        AUXILIARY coefficient fields (splicing the scalar leaf leaves
        the field stale — a silently-zero gradient). See
        :class:`fridom.transforms.Propagator` for the unrelated host
        ``State -> State`` run transform (Tier-2, not differentiable).

        Parameters
        ----------
        wrt : Sequence[str], optional
            Differentiation targets, in ``theta`` order: dotted bound
            parameter names (``"mixing.kappa"``, ``fr.params.TIME_STEP``)
            or PROGNOSTIC field names (``"b"``). Empty ``wrt`` builds a
            pure forward run taking ``theta=()`` (default: ()).
        steps : int
            The number of steps to advance (a positive int).
        remat : bool | None, optional
            Checkpoint the scan body (``jax.checkpoint``, unroll 1) so
            the reverse pass recomputes each step instead of taping it —
            trades compute for O(``steps``) adjoint memory. ``None`` /
            ``False`` keep the plain scan (default: None).

        Returns
        -------
        Callable
            ``run(theta=(), state=None) -> ModelState``, a pure
            function of ``theta`` (and the optional initial ``state``).

        Raises
        ------
        ValueError
            If ``steps`` is not a positive int.
        MissingParameterError
            If a ``wrt`` name is neither a bound parameter nor a
            PROGNOSTIC field.
        AssemblyError
            If a ``wrt`` parameter is an identity-defaulted constant,
            frozen into ``L`` by the stepper, or owned by a module that
            materializes AUXILIARY fields.
        NotImplementedError
            If the composition declares no PROGNOSTIC state.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) \
                or steps < 1:
            raise ValueError(
                f"propagator steps must be a positive int, got {steps!r}")
        if not self._artifacts.field_table.prognostic:
            raise NotImplementedError(
                "propagator() needs a PROGNOSTIC state to advance; this "
                "composition declares none (stage-only schedules land "
                "with wave 5)")
        wrt = tuple(wrt)
        targets = self._resolve_wrt_targets(wrt)
        record = self._artifacts.record
        base_carry = _copy_leaves(self._carry)
        base_stepper = self._stepper
        fresh = self._fresh_stepper_state()
        remat_on = bool(remat)

        def run(
            theta: Sequence[object] = (),
            state: VectorField | None = None,
        ) -> ModelState:
            """Advance ``steps`` steps as a pure function of ``theta``."""
            theta = tuple(theta)
            if len(theta) != len(wrt):
                raise ValueError(
                    f"propagator expected {len(wrt)} theta value(s) for "
                    f"wrt={wrt}, got {len(theta)}")
            carry = base_carry
            if state is not None:
                carry = self._carry_with_state(carry, state)
            carry = carry.replace(stepper_state=fresh)
            carry, stepper = _splice_targets(
                carry, base_stepper, targets, theta)
            return _chunk_body(record, steps, carry, stepper,
                               remat=remat_on)

        return run

    def _resolve_wrt_targets(
        self, wrt: tuple[str, ...],
    ) -> tuple[_WrtTarget, ...]:
        """Resolve/validate every ``wrt`` name (the taught refusals)."""
        table = self._binding_table
        prognostic = self._artifacts.field_table.prognostic
        remat_owners = {
            entry.owner
            for entry in self._artifacts.remat_table.entries}
        freezes = bool(getattr(
            self._stepper, "freezes_linear_operator", False))
        l_feeding = self._l_feeding_names() if freezes else frozenset()
        targets: list[_WrtTarget] = []
        for name in wrt:
            if name in table:
                targets.append(self._resolve_param_target(
                    name, table[name], remat_owners, l_feeding))
            elif name in prognostic:
                targets.append(_WrtTarget("field", -1, "", str(name)))
            else:
                bound = ", ".join(table.names) or "none"
                fields = ", ".join(prognostic) or "none"
                raise MissingParameterError(
                    f"propagator wrt target {str(name)!r} is neither a "
                    "bound parameter nor a PROGNOSTIC field; bound "
                    f"parameters: {bound}; prognostic fields: {fields}")
        return tuple(targets)

    def _resolve_param_target(
        self,
        name: str,
        entry: ParameterBinding,
        remat_owners: set[int],
        l_feeding: frozenset[str],
    ) -> _WrtTarget:
        """Refuse a non-differentiable parameter, or build its target."""
        if entry.slot is None:
            raise AssemblyError(
                f"propagator cannot differentiate {str(name)!r}: it is "
                "an identity-defaulted constant (no provider owns a "
                "leaf, so there is nothing to differentiate); provide "
                "it from a module, or re-assemble")
        # frozen-L before materialized: an L-parameter is stale under an
        # exponential stepper regardless of a stepper swap, so it earns
        # the more specific message (and the materialized refusal, which
        # also fires under a non-freezing stepper, otherwise masks it).
        if str(name) in l_feeding:
            raise AssemblyError(
                f"propagator cannot differentiate {str(name)!r}: it "
                "feeds the linear operator L, which "
                f"{type(self._stepper).__name__} freezes into an "
                "eigenbasis at assembly (freezes_linear_operator=True), "
                "so exp(L dt) is a snapshot and the gradient would be "
                "silently stale. Differentiating an L-frozen parameter "
                "is a v1 limitation (design/plans/active/"
                "time_dependent_fields.md, TDF-D4)")
        if entry.slot != "stepper" and entry.slot in remat_owners:
            fields = ", ".join(
                entry_.field
                for entry_ in self._artifacts.remat_table.entries
                if entry_.owner == entry.slot)
            owner = type(self._carry.modules[entry.slot]).__name__
            raise AssemblyError(
                f"propagator cannot differentiate {str(name)!r}: its "
                f"owner modules[{entry.slot}] ({owner}) materializes "
                f"AUXILIARY coefficient field(s) ({fields}) at "
                "assembly, and splicing the scalar leaf alone leaves "
                "those fields stale, so the gradient would be silently "
                "zero. Materialized-owner parameters are refused in v1 "
                "(in-trace rematerialization is the planned unlock, "
                "design/plans/active/differentiability_plan.md 5.2)")
        if entry.slot == "stepper":
            return _WrtTarget("stepper", -1, entry.attr, str(name))
        return _WrtTarget("module", entry.slot, entry.attr, str(name))

    def _l_feeding_names(self) -> frozenset[str]:
        """Union of every module's linear-operator parameter names."""
        names: set[str] = set()
        for module in self._carry.modules:
            names.update(module.linear_operator_parameters())
        return frozenset(names)

    def _carry_with_state(
        self, carry: ModelState, state: VectorField,
    ) -> ModelState:
        """Splice a supplied State's PROGNOSTIC fields into ``carry``."""
        table = self._artifacts.field_table
        incumbent = carry.state
        updates = {
            name: self._rehome(incumbent[name], state[name],
                               f"propagator state[{name}]")
            for name in table.prognostic if name in state}
        return carry.replace(state=incumbent.replace(**updates))

    # ================================================================
    #  The run loop (section 6.3)
    # ================================================================
    def _chunk_plan(self, steps: int) -> Iterator[int]:
        """Greedy C-chunks, then length-1 tails ({C, 1} only)."""
        full, tail = divmod(steps, self._chunk_size)
        for _ in range(full):
            yield self._chunk_size
        for _ in range(tail):
            yield 1

    def advance(
        self,
        steps: int,
        *,
        sync: bool = True,
        debug_nan: bool = False,
    ) -> AdvanceResult:
        """
        Advance ``steps`` steps — the public IO-free primitive.

        Description
        -----------
        A chunked scan over the DONATED carry via the shared
        :func:`step_chunk` (the stepper rides non-donated); the
        per-chunk panic-flag read is the ONLY host synchronization
        (CS-4 — the committed carry may hold pending arrays).
        Raises ``PanicError`` at the abort boundary (never
        mid-chunk) and on entry with a panicked carry. Under an
        x64-off run the traced clock is re-anchored from the
        authoritative host float64 clock at every chunk boundary
        (a no-op under the default x64-on run).

        Parameters
        ----------
        steps : int
            Steps to advance (>= 0).
        sync : bool, optional
            Block on the final carry before returning;
            ``sync=False`` is reserved (CS-5, ``PendingAdvance``)
            (default: True).
        debug_nan : bool, optional
            Keep a chunk-start carry copy so :meth:`replay_nan`
            can pinpoint the exact first-bad step with chunk(1)
            [spec concretization: ``run(debug_nan=)`` is wave 5's;
            the advance-level knob backs it] (default: False).

        Returns
        -------
        AdvanceResult
            The typed frozen return (CS-3).

        Raises
        ------
        PanicError
            On entry with a panicked carry, or at the first chunk
            boundary where the S5 flag reads true.
        """
        self._check_advance_entry(steps, sync=sync)
        if not debug_nan:
            self._debug_carry = None
            self._debug_steps = 0
        # one-time pre-chunk defragmentation: re-coalesce the pool so
        # XLA's single contiguous chunk temp arena can be placed
        if steps > 0 and self._defrag_pending \
                and self._should_defragment():
            self._defragment_carry()
            self._defrag_pending = False
        started = time.perf_counter()
        record = self._artifacts.record
        host_dt = float(self._stepper.dt)
        x64 = dtype_real() is np.float64
        done = 0
        for length in self._chunk_plan(steps):
            if debug_nan:
                self._debug_carry = _copy_leaves(self._carry)
                self._debug_steps = length
            carry = step_chunk(
                record, self._carry, self._stepper, length,
                async_compile=self._async_chunk_compile)
            self._carry = carry
            done += length
            # the authoritative host-side float64 clock
            self._host_elapsed += length * np.float64(host_dt)
            if not x64:
                # x64-off: re-anchor the traced elapsed (02_rules)
                self._carry = carry.replace(
                    clock=carry.clock.reanchored(
                        float(self._host_elapsed)))
            # boundary sequence: the panic read is the ONLY sync
            if bool(self._carry.panic.flag):
                self._abort(done, started)
        if debug_nan:
            self._debug_carry = None
            self._debug_steps = 0
        if sync:
            jax.block_until_ready(
                jax.tree_util.tree_leaves(self._carry))
        return AdvanceResult(
            steps_done=done, panicked=False, panic_it=None,
            wall_seconds=time.perf_counter() - started)

    def _check_advance_entry(self, steps: int, *, sync: bool) -> None:
        """Discharge advance()'s entry contracts (raising checks)."""
        if not sync:
            raise NotImplementedError(
                "advance(sync=False) is reserved (CS-5): the "
                "PendingAdvance handle's semantics are a leave-open")
        if isinstance(steps, bool) or not isinstance(steps, int) \
                or steps < 0:
            raise ValueError(
                f"steps must be a non-negative int, got {steps!r}")
        if self._panicked:
            raise PanicError(model_name=self._name,
                             first_bad_it=self._panic_it)
        if not self._artifacts.field_table.prognostic:
            raise NotImplementedError(
                "advancing a PROGNOSTIC-free composition (stage-"
                "only schedules) lands with wave 5")

    def _abort(self, done: int, started: float) -> None:
        """Raise the typed abort at a panicked chunk boundary."""
        self._panicked = True
        first_bad = int(self._carry.panic.it)
        self._panic_it = first_bad
        partial_result = AdvanceResult(
            steps_done=done, panicked=True, panic_it=first_bad,
            wall_seconds=time.perf_counter() - started)
        raise PanicError(model_name=self._name,
                         first_bad_it=first_bad,
                         partial=partial_result)

    def replay_nan(self) -> int | None:
        """
        Replay the panicked chunk with chunk(1) to the first NaN.

        Description
        -----------
        Requires a preceding ``advance(..., debug_nan=True)`` that
        panicked: the retained chunk-start carry copy is re-stepped
        one step at a time until the S5 flag flips, pinpointing the
        exact first-bad iteration. The model's own (panicked) carry
        is left untouched for autopsy.

        Returns
        -------
        int | None
            The first-bad iteration, or ``None`` if the replay
            stays finite (a cross-run nondeterminism red flag).

        Raises
        ------
        RuntimeError
            Without a retained debug carry.
        """
        if self._debug_carry is None:
            raise RuntimeError(
                "no chunk-start carry is retained; run "
                "advance(..., debug_nan=True) first")
        record = self._artifacts.record
        carry = _copy_leaves(self._debug_carry)
        for _ in range(self._debug_steps):
            carry = step_chunk(record, carry, self._stepper, 1)
            if bool(carry.panic.flag):
                return int(carry.panic.it)
        return None

    def _plan_run_steps(
        self,
        steps: int | None,
        runlen: float | np.timedelta64 | None,
        end_time: float | np.timedelta64 | None,
    ) -> int:
        """Reduce a run target to a step count (sign-agnostic).

        Description
        -----------
        Exactly one of ``steps``/``runlen``/``end_time`` (none or
        several is ambiguous — ``RunTargetError``). Durations reduce
        by ``steps = ceil((end - t0)/dt - eps)`` with the precondition
        ``(end - t0)*dt > 0``; ``runlen`` is an unsigned duration, the
        direction taken from the dt sign.
        """
        given = [steps is not None, runlen is not None,
                 end_time is not None]
        if sum(given) != 1:
            raise RunTargetError(
                "run() takes exactly one of steps=/runlen=/end_time="
                f" (got {sum(given)} of them); none or several is "
                "ambiguous")
        dt = float(self._stepper.dt)
        if dt == 0.0:
            raise RunTargetError(
                "the time step is zero; run() cannot plan a target")
        if steps is not None:
            if isinstance(steps, bool) or not isinstance(steps, int) \
                    or steps < 0:
                raise RunTargetError(
                    f"steps= must be a non-negative int; got {steps!r}")
            return steps
        if runlen is not None:
            runlen_s = abs(_run_target_seconds(runlen, name="runlen"))
            if runlen_s == 0.0:
                raise RunTargetError(
                    f"runlen= must be a nonzero duration; got "
                    f"{runlen!r}")
            return max(1, math.ceil(runlen_s / abs(dt) - _TARGET_EPS))
        end_s = _run_target_seconds(end_time, name="end_time")
        t0 = float(self._carry.clock.time)
        delta = end_s - t0
        if delta * dt <= 0.0:
            raise RunTargetError(
                f"run(end_time={end_time!r}) cannot be reached from "
                f"t0={t0} with dt={dt}: the precondition "
                "(end - t0)*dt > 0 fails (flip fr.params.TIME_STEP "
                "via update_parameters for a backward leg)")
        return max(1, math.ceil(delta / dt - _TARGET_EPS))

    def run(
        self,
        steps: int | None = None,
        *,
        runlen: float | np.timedelta64 | None = None,
        end_time: float | np.timedelta64 | None = None,
        outputs: tuple = (),
        snapshots: Any = None,
        max_chunk: int | None = None,
        progress: bool | Any = True,
        jit: bool = True,
        profile: str | None = None,
        debug_nan: bool = False,
        raise_on_nan: bool = False,
    ) -> RunResult:
        """Advance to a run target, driving IO — sugar over advance.

        Description
        -----------
        Reimplemented as a single-model ``fr.ops.Session`` loop (the
        facade law: ``run(n)`` is bitwise-identical to a hand-written
        Session loop with the same chunk plan, and adds zero jit-cache
        entries). Exactly one of
        ``steps``/``runlen``/``end_time`` sets the target
        (:class:`RunTargetError` on a bad or ambiguous target). The
        model's standing ``io=`` streams plus ``outputs=`` bind at run
        start. ``raise_on_nan=False`` (the default) catches the
        Session's ``PanicError`` and returns ``RunResult(NAN_ABORT)``
        (notebooks want the carry); ``raise_on_nan=True`` re-raises.
        ``run()`` never exits the process.

        Parameters
        ----------
        steps : int or None, optional
            Advance exactly this many steps (default: None).
        runlen : float or np.timedelta64 or None, optional
            Advance this (unsigned) model-time duration (default:
            None).
        end_time : float or np.timedelta64 or None, optional
            Advance until this absolute model time (default: None).
        outputs : tuple, optional
            Per-run output streams, added to the model's ``io=``
            (default: ()).
        snapshots : Snapshots or None, optional
            The restart-snapshot run config (default: None).
        max_chunk : int or None, optional
            Host-sync granularity override (default: None).
        progress : bool or ProgressReporter, optional
            Progress rendering (default: True).
        jit : bool, optional
            Reserved chunk-dispatch policy; ``jit=False`` is not wired
            in wave 5 (default: True).
        profile : str or None, optional
            Reserved profiling target; accepted, not wired in wave 5
            (default: None).
        debug_nan : bool, optional
            Keep a chunk-start carry copy for ``replay_nan``
            (default: False).
        raise_on_nan : bool, optional
            Re-raise ``PanicError`` instead of returning
            ``RunResult(NAN_ABORT)`` (default: False).

        Returns
        -------
        RunResult
            The aggregated single-model run result.
        """
        from fridom.model.ops.session import (  # noqa: PLC0415 — avoids the ops<->model import cycle
            Session,
        )
        if profile is not None:
            _log.debug(
                "run(profile=%r) is accepted but not wired in wave 5",
                profile)
        session = Session(
            self, outputs=tuple(self._io) + tuple(outputs),
            snapshots=snapshots, progress=progress,
            max_chunk=max_chunk, jit=jit, debug_nan=debug_nan)
        key = self._name if self._name is not None else "model"
        try:
            with session as active:
                # plan AFTER __enter__ so a snapshot resume re-plans
                # the remaining steps against the absolute target
                # (end_time= is the natural resumable spelling)
                n_steps = self._plan_run_steps(steps, runlen, end_time)
                active.advance({self: n_steps})
                results = dict(active.result)
            return results[key]
        except PanicError:
            if raise_on_nan:
                raise
            # the Session's bookkeeping already recorded NAN_ABORT and
            # the partial step count; aggregate it post-exit
            return session._aggregate()[key]  # noqa: SLF001 — sibling-owned helper

    def tendency(
        self,
        state: VectorField,
        *,
        t: float | None = None,
        filter: Callable | None = None,  # noqa: A002 — spec-fixed keyword
        constraints: bool = True,
    ) -> VectorField:
        """
        Host-callable, jitted, read-only composed tendency (2.8).

        Description
        -----------
        The linear-stability / JVP matvec surface (08 §10.6 S1): the
        input's PROGNOSTIC components are overlaid onto the carry's
        current full state (so AUXILIARY inputs are available), the
        SELF_UPDATE and DIAGNOSE stages run first at ``t`` (amended
        V-H8 — the result reflects recomputed diagnostics), the
        filtered EXPLICIT terms accumulate (IMPLICIT terms via their
        forward apply ``L @ state``), and the CONSTRAINT stages apply
        to the result iff ``constraints`` — the accumulated PROGNOSTIC
        sums overlay the prepared full state (so auxiliary reads
        resolve) and diagnostic stage writes (e.g. the nonhydro
        pressure ``p``) stay on that overlay, never entering the
        returned PROGNOSTIC-only tendency (the H1 accumulator fix).
        **Never advances the carry.** The composed tendency is jitted
        separately, keyed by the kept-term set and ``constraints``
        (its own cache entry).

        Parameters
        ----------
        state : VectorField
            The state to evaluate the tendency at; its declared
            components overlay the carry's current state.
        t : float | None, optional
            The stage time for parameter/diagnostic evaluation;
            ``None`` reads the carry clock (default: None).
        filter : Callable | None, optional
            A term predicate (``fr.terms``) restricting the evaluated
            terms — per-term budgets, linear matvecs (default: None).
        constraints : bool, optional
            Whether the CONSTRAINT stages apply to the result
            (default: True).

        Returns
        -------
        VectorField
            The PROGNOSTIC-only composed tendency (``d state / d t``).
        """
        schedule = self._artifacts.schedule
        if not schedule.prognostic:
            raise NotImplementedError(
                "model.tendency needs PROGNOSTIC fields; a stage-only"
                " / field-free composition has no state tendency")
        base_state = self._carry.state
        if base_state is None:  # pragma: no cover — prognostic implies state
            raise AttributeError("this composition declares no fields")
        kept = self._resolve_tendency_filter(filter)
        run = self._tendency_executable(kept, constraints)
        t_val = (self._carry.clock.time if t is None
                 else jnp.asarray(t, dtype=dtype_real()))
        return run(state, base_state, self._carry.modules,
                   self._stepper, t_val)

    def _terms_by_key(self, modules: tuple) -> dict[str, object]:
        """Map each term's attribution key to its ``TendencyTerm``."""
        return {
            f"{type(modules[slot]).__name__}/{term.name}": term
            for slot, term in _collect_terms(modules)}

    def _resolve_tendency_filter(
        self, term_filter: Callable | None,
    ) -> frozenset[str]:
        """Resolve the kept-term key set for a tendency filter."""
        entries = self._artifacts.schedule.kind_entries(None)
        if term_filter is None:
            return frozenset(entry.key for entry in entries)
        modules = self._carry.modules
        term_by_key = self._terms_by_key(modules)
        _validate_filter_names(term_filter, tuple(term_by_key))
        wants_module = getattr(term_filter, "wants_module", False)
        kept = set()
        for entry in entries:
            term = term_by_key[entry.key]
            module = modules[entry.slot]
            keep = (term_filter(entry.key, term, module) if wants_module
                    else term_filter(entry.key, term))
            if keep:
                kept.add(entry.key)
        return frozenset(kept)

    def _tendency_executable(
        self, kept: frozenset[str], constraints: bool,
    ) -> Callable:
        """Build (memoized) the jitted composed-tendency function."""
        cache_key = (kept, constraints)
        cached = self._tendency_cache.get(cache_key)
        if cached is not None:
            return cached
        schedule = self._artifacts.schedule
        prognostic = schedule.prognostic
        term_entries = schedule.kind_entries(None)
        do_constrain = bool(
            constraints and schedule.kind_entries(StageKind.CONSTRAINT))

        def run(
            state: VectorField,
            base_state: VectorField,
            modules: tuple,
            stepper: TimeStepper,
            t: Any,
        ) -> VectorField:
            """Overlay, prepare at t, accumulate the kept terms."""
            overlay = {name: state[name]
                       for name in state.component_names
                       if name in base_state}
            full = base_state.replace(**overlay)
            table = schedule.binding_table
            params = (table.eval_params(modules, stepper, t)
                      if table is not None else {})
            ctx = StepContext(params=params, clock=t, dt=stepper.dt,
                              stage_dt=stepper.dt)
            bound = schedule.bind(modules)
            full = bound.prepare(full, ctx)
            sums = VectorField({name: zero_like(full[name])
                               for name in prognostic})
            for entry in term_entries:
                if entry.key not in kept:
                    continue
                module = modules[entry.slot]
                if entry.treatment is Treatment.IMPLICIT:
                    result = entry.implicit.apply(module, full, ctx)
                else:
                    result = evaluate_entry(entry, module, full, ctx)
                sums = apply_add(entry, sums, result)
            if do_constrain:
                # the CONSTRAINT stages see the accumulated PROGNOSTIC
                # sums overlaid on the prepared full state (auxiliary
                # reads resolve; diagnostic writes such as the nonhydro
                # pressure land on the overlay and are dropped from the
                # PROGNOSTIC-only result — the H1 accumulator fix)
                constrained = bound.constrain(
                    full.replace(**{name: sums[name]
                                    for name in prognostic}), ctx)
                sums = VectorField({name: constrained[name]
                                    for name in prognostic})
            return sums

        jitted = jax.jit(run)
        self._tendency_cache[cache_key] = jitted
        return jitted

    def constrain(
        self,
        state: VectorField,
        *,
        t: float | None = None,
    ) -> VectorField:
        """
        Host-callable, jitted, read-only CONSTRAINT application (H1).

        Description
        -----------
        The public projector matvec the numeric eigenmode probes
        compose around the raw tendency: the input's PROGNOSTIC
        components are overlaid onto the carry's current full state
        (so AUXILIARY inputs are available), the CONSTRAINT stages run
        at ``t``, and the PROGNOSTIC subset of the constrained state
        is returned — diagnostic stage writes (e.g. the nonhydro
        pressure ``p``) stay on the internal overlay. For the nonhydro
        pressure projection this realizes the (M-orthogonal) Leray
        projector; a model without CONSTRAINT stages returns the
        input's PROGNOSTIC subset unchanged. **Never advances the
        carry.**

        Parameters
        ----------
        state : VectorField
            The state to constrain; its declared components overlay
            the carry's current state.
        t : float | None, optional
            The stage time for parameter evaluation; ``None`` reads
            the carry clock (default: None).

        Returns
        -------
        VectorField
            The PROGNOSTIC-only constrained state.
        """
        schedule = self._artifacts.schedule
        if not schedule.prognostic:
            raise NotImplementedError(
                "model.constrain needs PROGNOSTIC fields; a stage-"
                "only / field-free composition has no state to "
                "constrain")
        base_state = self._carry.state
        if base_state is None:  # pragma: no cover — prognostic implies state
            raise AttributeError("this composition declares no fields")
        run = self._constrain_executable()
        t_val = (self._carry.clock.time if t is None
                 else jnp.asarray(t, dtype=dtype_real()))
        return run(state, base_state, self._carry.modules,
                   self._stepper, t_val)

    def _constrain_executable(self) -> Callable:
        """Build (memoized) the jitted CONSTRAINT-stage function."""
        cache_key = ("constrain",)
        cached = self._tendency_cache.get(cache_key)
        if cached is not None:
            return cached
        schedule = self._artifacts.schedule
        prognostic = schedule.prognostic
        has_constraint = bool(
            schedule.kind_entries(StageKind.CONSTRAINT))

        def run(
            state: VectorField,
            base_state: VectorField,
            modules: tuple,
            stepper: TimeStepper,
            t: Any,
        ) -> VectorField:
            """Overlay, run the CONSTRAINT stages, strip to PROG."""
            overlay = {name: state[name]
                       for name in state.component_names
                       if name in base_state}
            full = base_state.replace(**overlay)
            if has_constraint:
                table = schedule.binding_table
                params = (table.eval_params(modules, stepper, t)
                          if table is not None else {})
                ctx = StepContext(params=params, clock=t,
                                  dt=stepper.dt, stage_dt=stepper.dt)
                bound = schedule.bind(modules)
                full = bound.constrain(full, ctx)
            return VectorField(
                {name: full[name] for name in prognostic})

        jitted = jax.jit(run)
        self._tendency_cache[cache_key] = jitted
        return jitted

    def variant(
        self,
        *,
        term_filter: Callable | None = None,
        updates: Mapping[str, object] | None = None,
        extra_modules: tuple = (),
        name: str | None = None,
    ) -> Model:
        """
        Re-assemble a derived model on the parent's frozen grid (2.8).

        Description
        -----------
        08 §10.4: declarations are **never** filtered — the variant
        shares the parent's FieldTable, State treedef, shapes, halos
        and layouts (parent<->variant state exchange is copy-free).
        Only tendency terms are filtered (at assembly step 5). The ⊆
        verify lemma guarantees the frozen-grid verify path passes
        (variants never ``GridFrozenError``). ``updates=`` is
        assembly-time and may change value **specs** (scalar -> Ramp,
        ``fr.params.TIME_STEP`` sign) — unlike post-assembly
        ``update_parameters``; the carry treedef may then differ, but
        the State treedef (the load-bearing identity) does not.

        ``extra_modules=`` appends field-free modules (parameter
        providers, the ``fr.modules.TendencyEnvelope`` term
        envelope) to the cloned parent tuple. A module declaring
        fields is refused: it would extend the FieldTable and break
        the shared parent/variant State treedef.

        Parameters
        ----------
        term_filter : Callable | None, optional
            A term predicate (``fr.terms``) removing terms at assembly;
            an empty result / unknown ``named`` key is a build error
            (default: None).
        updates : Mapping[str, object] | None, optional
            Assembly-time parameter value (or spec) changes, resolved
            through the parent's binding table (default: None).
        extra_modules : tuple, optional
            Field-free modules appended (as fresh clones) after the
            parent's modules (default: ``()``).
        name : str | None, optional
            The variant's report/log name (default: ``"{parent}/
            variant"``).

        Returns
        -------
        Model
            The derived model (a full lifecycle citizen).

        Raises
        ------
        AssemblyError
            If an extra module declares fields, or an update names an
            unbound/identity-defaulted parameter.
        """
        # fresh clones: the child assembly binds (and freezes) the
        # module instances it is handed — cloning keeps the parent's
        # carry modules unbound, so variants never consume the parent
        modules = [self._fresh_clone(m) for m in self._carry.modules]
        stepper = self._stepper
        for pname, value in dict(updates or {}).items():
            entry = self._binding_table[pname]  # MissingParameterError
            if entry.slot is None:
                raise AssemblyError(
                    f"parameter {str(pname)!r} is an identity-"
                    "defaulted constant (no provider owns a leaf); "
                    "provide it from a module, or re-assemble")
            if entry.slot == "stepper":
                stepper = self._replace_leaf(stepper, entry.attr, value)
            else:
                modules[entry.slot] = self._replace_leaf(
                    modules[entry.slot], entry.attr, value)
        # extras append AFTER the updates loop (parent binding slots
        # index the parent tuple) and BEFORE the filter-name check
        # (named() filters may reference extra-module terms)
        for extra in tuple(extra_modules):
            if tuple(getattr(extra, "field_declarations", ())):
                raise AssemblyError(
                    f"variant extra module {type(extra).__name__} "
                    "declares fields; extra_modules may not extend "
                    "the FieldTable (the parent/variant shared state "
                    "treedef is the load-bearing identity) — "
                    "assemble a new model instead")
            modules.append(self._fresh_clone(extra))
        _validate_filter_names(
            term_filter, tuple(self._terms_by_key(tuple(modules))))
        if name is not None:
            variant_name = name
        elif self._name:
            variant_name = f"{self._name}/variant"
        else:
            variant_name = "variant"
        return Model(
            grid=self._grid, modules=tuple(modules),
            time_stepper=stepper,
            state_type=self._artifacts.record.state_type,
            name=variant_name, term_filter=term_filter,
            chunk_size=self._chunk_size,
            async_chunk_compile=self._async_chunk_compile)

    # ================================================================
    #  Persistence (section 6.4)
    # ================================================================
    def _carry_leaves(
        self,
    ) -> tuple[tuple[tuple[str, object], ...], object]:
        """
        Flatten the carry to named leaves, fields kept whole.

        Description
        -----------
        The snapshot leaf naming: ``jax.tree_util.keystr`` over the
        carry's path-flatten with ``ScalarField`` treated as a leaf
        — unique and stable for one treedef (which the fingerprint
        check pins). Field leaves are stored as their GATHERED
        true-shape global arrays (device-count-portable; no device
        identity in the record); every other leaf is stored as-is.
        """
        flat, treedef = jax.tree_util.tree_flatten_with_path(
            self._carry,
            is_leaf=lambda node: isinstance(node, ScalarField))
        named = tuple(
            (jax.tree_util.keystr(path), leaf)
            for path, leaf in flat)
        return named, treedef

    def _provided_values(self) -> dict[str, float]:
        """Numeric provided-parameter values for the manifest."""
        values: dict[str, float] = {}
        live = self._binding_table.host_view(
            self._carry.modules, self._stepper)
        for name in live:
            try:
                values[name] = float(live[name])
            except (TypeError, ValueError):
                continue  # Ramps/profiles: leaves, not header data
        return values

    def snapshot(self, path: str | Path) -> None:
        """
        Write the dumb leaf blob: manifest + true-shape leaves.

        Description
        -----------
        The versioned manifest records the fingerprint digest AND
        its source record (mismatches diff, never silently reuse)
        plus the provided-parameter values (at least
        ``fr.params.TIME_STEP``); the leaves are the gathered
        true-shape carry arrays. Never pickles (02_rules).

        Parameters
        ----------
        path : str | Path
            The target snapshot directory (must not exist).
        """
        named, _ = self._carry_leaves()
        decomposition = self._grid.decomposition
        leaves: dict[str, np.ndarray] = {}
        for key, leaf in named:
            if isinstance(leaf, ScalarField):
                gathered = decomposition.gather(
                    leaf._data, leaf.function_space)  # noqa: SLF001 — storage seam
                leaves[key] = np.asarray(gathered)
            else:
                leaves[key] = np.asarray(leaf)
        clock = self._carry.clock
        manifest = SnapshotManifest(
            version=FORMAT_VERSION,
            model_name=self._name,
            time=float(clock.time),
            iteration=int(clock.it),
            provided_parameters=self._provided_values(),
            fingerprint=self.fingerprint.digest,
            fingerprint_record=dict(self.fingerprint.source))
        write_snapshot(path, manifest, leaves)

    def load_snapshot(self, path: str | Path) -> None:
        """
        Assemble-then-load: check, overwrite all carry leaves.

        Description
        -----------
        Fingerprint check first (``SnapshotMismatchError`` diffs;
        never silent reuse), then the dt check (SIGN mismatch
        errors, magnitude warns), then every carry leaf is
        overwritten BITWISE and re-homed per the current device
        count (field leaves re-enter through the true-shape re-pad).
        Clears the panic flag.

        Parameters
        ----------
        path : str | Path
            A committed snapshot directory.
        """
        manifest = read_manifest(path)
        check_fingerprint(
            manifest, digest=self.fingerprint.digest,
            record=dict(self.fingerprint.source))
        check_dt(manifest, float(self._stepper.dt))
        stored = read_leaves(path, manifest)
        named, treedef = self._carry_leaves()
        keys = tuple(key for key, _ in named)
        missing = tuple(key for key in keys if key not in stored)
        extra = tuple(key for key in stored if key not in keys)
        if missing or extra:
            raise SnapshotMismatchError(
                "snapshot leaves do not match the assembled carry "
                f"(missing: {missing or 'none'}; unexpected: "
                f"{extra or 'none'}) — fingerprint version skew?")
        rebuilt = []
        for key, incumbent in named:
            array = stored[key]
            if isinstance(incumbent, ScalarField):
                # the true-shape re-pad: pad + halo sync re-home
                rebuilt.append(
                    incumbent.with_data(jnp.asarray(array)))
            else:
                rebuilt.append(jnp.asarray(array))
        carry: ModelState = jax.tree_util.tree_unflatten(
            treedef, rebuilt)
        carry = carry.replace(panic=_fresh_panic())
        self._commit(carry)
        self._panicked = False
        self._panic_it = None
        self._host_elapsed = np.float64(
            float(self._carry.clock.elapsed))
