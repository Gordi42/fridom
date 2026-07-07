"""
The Model composition root and the run loop.

Description
-----------
Wave 4.2: ``Model`` (the host-side composition root and driver),
``ModelState`` (the full dynamic carry), ``step_chunk`` (THE one
framework-level jitted entry), ``ParameterView``,
``DiagnosticsNamespace`` and ``BoundDiagnostic``. Owning class spec:
``notes/framework2/model/classes/model.md``; run-loop rules:
``notes/framework2/model/02_rules.md`` (Clock precision, the
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
import time
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.io.snapshots import (
    FORMAT_VERSION,
    SnapshotManifest,
    Snapshots,
    check_dt,
    check_fingerprint,
    read_leaves,
    read_manifest,
    write_snapshot,
)
from fridom.framework2.io.streams import (
    IOCollisionError,
    SnapshotMismatchError,
)
from fridom.framework2.model.assembly import assemble
from fridom.framework2.model.clock import Clock
from fridom.framework2.model.context import StepContext
from fridom.framework2.model.declarations import Lifecycle
from fridom.framework2.model.errors import AssemblyError
from fridom.framework2.model.results import (
    AdvanceResult,
    PanicError,
    RunResult,
    RunTargetError,
)
from fridom.framework2.model.schedule import BoundSchedule
from fridom.framework2.model.stages import StageKind
from fridom.framework2.model.time_dependent import resolve_at

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.assembly import (
        AssemblyArtifacts,
        AssemblyRecord,
        Fingerprint,
    )
    from fridom.framework2.model.field_table import (
        FieldRecord,
        FieldTable,
    )
    from fridom.framework2.model.parameters import ParameterDeclaration
    from fridom.framework2.model.report import AssemblyReport
    from fridom.framework2.model.time_steppers.base import (
        StepperState,
        TimeStepper,
    )

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


# ================================================================
#  PanicState — the S5 ledger entry
# ================================================================
class PanicState(NamedTuple):

    """
    The carried S5 NaN-seam record: sticky flag + first-failure it.

    Description
    -----------
    A NamedTuple (hence automatically a pytree): ``flag`` is the
    sticky in-trace panic bit (once true, stays true for the rest of
    the chunk), ``it`` records the clock iteration of the FIRST
    non-finite step. The chunk body reduces into it per step; the
    host reads it at chunk boundaries only (the one sanctioned host
    synchronization, CS-4).

    Parameters
    ----------
    flag : jax.Array
        Sticky boolean: any non-finite value seen so far.
    it : jax.Array
        First-failure iteration (integer at the global width);
        meaningful only while ``flag`` is set.
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
def _strengthen(leaf: object) -> jax.Array:
    """Coerce one leaf to a strongly-typed array of its own dtype."""
    array = jnp.asarray(leaf)
    # astype with an explicit dtype strips the weak type; a no-op
    # for already-strong leaves
    return array.astype(array.dtype)


@jax.jit
def _canonicalize(tree: Any) -> Any:
    """
    Normalize a carry's leaves to the canonical jit-output form.

    Description
    -----------
    Host-born leaves (``grid.create_field`` products, ``asarray``
    scalars) may carry same-layout-but-differently-spelled shardings
    and weakly-typed avals; jit outputs are the canonical form. One
    pass through this jitted identity commits every leaf, so every
    carry the model holds keys the SAME compiled-chunk cache entry.
    """
    return jax.tree_util.tree_map(_strengthen, tree)


def _copy_leaves(tree: Any) -> Any:
    """Deep-copy a pytree's array leaves (the debug_nan copy)."""
    return jax.tree_util.tree_map(jnp.copy, tree)


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
) -> ModelState:
    """
    Trace one chunk: ``lax.scan`` of the composed step body.

    Description
    -----------
    Per step: ``schedule.bind`` over the carry's current modules
    (via the live view), ``stepper.step``, then the chunk body's
    per-step epilogue — the S5 ``isfinite`` reduction into the
    sticky ``panic`` pair (no ``lax.cond`` no-op wrapper) and the S6
    DIAGNOSTIC stages. The epilogue is the CHUNK BODY's, never the
    stepper's. NOTE (wave-5 seam): the S6 ctx carries no tendency
    sums yet — the treatment partition threads them at 2.5.
    """
    schedule = record.schedule
    run_diagnostics = bool(
        schedule.kind_entries(StageKind.DIAGNOSTIC))

    def one_step(
        carry: ModelState, _: None,
    ) -> tuple[ModelState, None]:
        """One composed step plus the S5/S6 epilogue."""
        bound = _LiveBoundSchedule(schedule, carry.modules, stepper)
        stepper_state, state, clock = stepper.step(
            carry.stepper_state, carry.state, bound, carry.clock)
        # ---- S5: the isfinite reduction into the sticky pair ----
        finite = _all_finite(state)
        newly_bad = jnp.logical_and(~carry.panic.flag, ~finite)
        panic = PanicState(
            flag=jnp.logical_or(carry.panic.flag, ~finite),
            it=jnp.where(newly_bad, clock.it, carry.panic.it))
        # ---- S6: DIAGNOSTIC stages (post-NaN-seam epilogue) -----
        if run_diagnostics:
            ctx = bound.context(clock, dt=stepper.dt,
                                stage_dt=stepper.dt)
            state = bound.diagnostics(state, ctx)
        return ModelState(state, carry.modules, stepper_state,
                          clock, panic), None

    out, _ = jax.lax.scan(one_step, model_state, xs=None, length=n)
    return out


#: the ONE jitted entry — one per process, never a per-assembly
#: closure (which would silently defeat the shared jit cache); the
#: assembly record and the chunk length are static, the carry is
#: donated, the stepper is a loop-invariant NON-donated input
_JITTED_CHUNK = jax.jit(
    _chunk_body, static_argnums=(0, 1), donate_argnums=(2,))

#: AOT-compiled chunk executables, keyed by (record, n, structure)
_CHUNK_EXECUTABLES: Final[dict[tuple, Any]] = {}

#: per-key compile accounting: key -> (seconds, memory analysis)
_CHUNK_COMPILE_LOG: Final[dict[tuple, tuple[float, object]]] = {}


def _leaf_signature(leaf: object) -> tuple:
    """One leaf's contribution to the compiled-chunk cache key."""
    if isinstance(leaf, jax.Array):
        return (tuple(leaf.shape), str(leaf.dtype),
                bool(leaf.weak_type), str(leaf.sharding))
    return ("host", type(leaf).__name__)


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


def step_chunk(
    record: AssemblyRecord,
    carry: ModelState,
    stepper: TimeStepper,
    n: int,
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

    Returns
    -------
    ModelState
        The advanced carry.
    """
    key = _chunk_key(record, carry, stepper, n)
    compiled = _CHUNK_EXECUTABLES.get(key)
    if compiled is None:
        started = time.perf_counter()
        compiled = _JITTED_CHUNK.lower(
            record, n, carry, stepper).compile()
        seconds = time.perf_counter() - started
        try:
            memory = compiled.memory_analysis()
        except Exception:  # noqa: BLE001 — backend-optional report
            memory = None  # pragma: no cover
        _CHUNK_EXECUTABLES[key] = compiled
        _CHUNK_COMPILE_LOG[key] = (seconds, memory)
    return compiled(carry, stepper)


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
    ) -> None:
        """Assemble (steps 1-7, 9) and allocate the carry (step 8)."""
        if isinstance(chunk_size, bool) or not isinstance(
                chunk_size, int) or chunk_size < 1:
            raise ValueError(
                f"chunk_size must be an int >= 1, got {chunk_size!r}")
        for stream in io:
            if isinstance(stream, Snapshots):
                raise IOCollisionError(
                    "io= rejects Snapshots — snapshots are "
                    "run-config only (one resume path, never two); "
                    "pass them to run(snapshots=...)")
        modules = tuple(modules)
        self._artifacts: AssemblyArtifacts = assemble(
            grid=grid, modules=modules, time_stepper=time_stepper,
            state_type=state_type, name=name)
        self._grid = grid
        self._stepper = time_stepper
        self._name = name
        self._io = tuple(io)
        self._chunk_size = chunk_size
        # -- step 8: allocate the carry ---------------------------
        state = self._allocate_state(modules)
        # built once: a constant zero PROGNOSTIC vector, reused by
        # every re-warm (a fresh build would pay the grid's
        # pad-and-sync store path per update_parameters call —
        # a per-call retrace under multi-device)
        self._tendency_template = (
            self._artifacts.composer.tendency_template())
        template = self._tendency_template
        stepper_state: StepperState = (
            time_stepper.init(template) if template is not None
            else ())
        carry = ModelState(
            state=state, modules=modules,
            stepper_state=stepper_state, clock=Clock(),
            panic=_fresh_panic())
        self._carry = _canonicalize(carry)
        # host mirrors (cheap reads; no device sync)
        self._panicked = False
        self._panic_it: int | None = None
        self._host_elapsed = np.float64(0.0)
        self._debug_carry: ModelState | None = None
        self._debug_steps = 0

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
        """Canonicalize and swap in a host-built carry."""
        self._carry = _canonicalize(carry)

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

    @property
    def chunk_size(self) -> int:
        """advance()'s host-sync granularity C."""
        return self._chunk_size

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
        The sanctioned re-home is the true-shape re-pad
        (``with_data`` routes through ``decomposition.pad`` + the
        halo sync), never a bare ``device_put``; device-resident
        fields stay on device (CS-8). The incumbent's metadata is
        kept.
        """
        if isinstance(value, ScalarField):
            incoming = value.function_space.bare
            declared = incumbent.function_space.bare
            if incoming != declared:
                raise ValueError(
                    f"{label}: the incoming field lives on "
                    f"{incoming!r}, but the declared component "
                    f"space is {declared!r}")
            return incumbent.with_data(value.data)
        if callable(value):
            built = self._grid.create_field(
                incumbent.function_space, init=value)
            return incumbent.with_data(built.data)
        array = jnp.asarray(value)
        if tuple(array.shape) != tuple(incumbent.shape):
            raise ValueError(
                f"{label}: expected the true shape "
                f"{incumbent.shape}, got {tuple(array.shape)}")
        return incumbent.with_data(array)

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
        self._commit(self._carry.replace(
            state=state.replace(**updates), panic=_fresh_panic()))
        self._panicked = False
        self._panic_it = None

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
        template = self._tendency_template
        return (self._stepper.init(template)
                if template is not None else ())

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
        clone = object.__new__(type(provider))
        clone.__dict__.update(provider.__dict__)
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
        started = time.perf_counter()
        record = self._artifacts.record
        host_dt = float(self._stepper.dt)
        x64 = dtype_real() is np.float64
        done = 0
        for length in self._chunk_plan(steps):
            if debug_nan:
                self._debug_carry = _copy_leaves(self._carry)
                self._debug_steps = length
            carry = step_chunk(record, self._carry, self._stepper,
                               length)
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
        from fridom.framework2.ops.session import (  # noqa: PLC0415 — avoids the ops<->model import cycle
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

    def tendency(self, *_args: object, **_kwargs: object) -> object:
        """``model.tendency`` lands at wave 7 (2.8)."""
        raise NotImplementedError(
            "Model.tendency lands at wave 7 with the read-only "
            "composed tendency (ROADMAP 2.8)")

    def variant(self, *_args: object, **_kwargs: object) -> Model:
        """``model.variant`` (derived models) lands at wave 7 (2.8)."""
        raise NotImplementedError(
            "Model.variant lands at wave 7 with the term-predicate "
            "algebra (ROADMAP 2.8)")

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
