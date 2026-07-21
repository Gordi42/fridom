"""
The gridded zarr writer (tensorstore sink).

Description
-----------
Wave 5 C: ``fr.io.Writer`` (front kwargs fixed at the 2.4 spec; sink
behind the seam). Owning class spec:
``design/specs/model/classes/io_ops.md`` (the Writer section).

A ``Writer`` is trigger-driven, human-facing gridded output: the
store it writes is a **zarr-format** store that opens in xarray/xgcm
with **no post-processing**. The layout contract is the grid
cluster's ``f.xr`` export (``fridom.spatial.export``): at bind this
writer builds each variable's store from the **values-free**
``export_layout`` (``ExportLayout``) and reads the labels/coords/attrs
straight off it — the xgcm-style staggered dim names, the
``c_grid_axis_shift`` comodo attrs, the coordinate values from
``grid.evaluation_nodes``, and the per-variable ``FieldMetadata``
attrs are therefore never duplicated here, and bind imports no
``xarray``. The writer only adds the time axis (a CF
``seconds since <start_date>`` coordinate plus an ``iteration``
coordinate) and the append/truncate mechanics.

SINK ENGINE (owner directive, deviates from io_ops.md): the write
backend is **tensorstore**, not the ``zarr`` Python package — this
module never imports ``zarr``. tensorstore's zarr (v2) driver writes
the array chunks and ``.zarray`` metadata; the small zarr sidecars
that make the store xarray-openable (``.zgroup``, the group and
per-array ``.zattrs`` carrying ``_ARRAY_DIMENSIONS`` and the CF/xgcm
attributes) are written here as plain JSON. xarray's read path still
pulls ``zarr`` under the hood — that is expected and lives on the
reader, not this writer. Cross-boundary async is opt-in
(``async_writes``, see :meth:`Writer.write`); file-split writes stay
designed-for behind this seam.

The bind/write split (``gather_free_output_plan.md``): bind is
layout-only (``export_layout`` — no gather, no ``xarray``), and the
default spatial chunks come from the decomposition's write-aligned
grid (``chunk_hint``). The write path is the shard-wise sink: each
firing writes its locally-owned true-DOF tiles straight into the
store via ``decomposition.shard_writes`` (one contiguous host copy
per shard, with the halo ghosts, stagger reserve, and cell padding
stripped), so no gather happens anywhere in the writer and the global
array is never formed on any device. Coefficient-space and complex
fields raise at bind, inheriting the ``f.xr`` iteration-1 restriction,
with a pointer to ``.data``.
"""
# Wave 5 C: Writer (tensorstore zarr-append decomposed-slice sink)
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np

from fridom.model.io.streams import reject_walltime_trigger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping, Sequence

    from fridom.model.io.triggers import Trigger
    from fridom.spatial.export import ExportLayout
    from fridom.spatial.fields.scalar_field import ScalarField

# nanoseconds per second (the CF reference-date conversion)
_NS_PER_S = 1_000_000_000

# the accepted create/append vocabulary (V-S1; "w-" is the default)
_MODES = ("w", "w-", "a")

# the xgcm stagger-position vocabulary of the export layout: a
# staggered dim exports as ``<name>_<position>``; stripping the
# suffix recovers the coordinate's unit-factor row name
_POSITIONS = ("left", "right", "outer", "inner")

# iteration fill for an un-committed slice. The iteration axis is
# written last, after a firing's variable tiles commit, so a real
# iteration value marks a fully-durable slice; an interrupted firing
# keeps this sentinel and is dropped on reopen / truncate. It is out of
# range for any real iteration (INT64_MIN), so it cannot collide with a
# genuine step index (which may be negative for a backward run).
_UNWRITTEN = int(np.iinfo(np.int64).min)


# ================================================================
#  Multi-process seams (mockable; single-process no-ops)
# ================================================================
def _process_index() -> int:
    """Return this process's index in the jax group (0 if single)."""
    return jax.process_index()


def _process_count() -> int:
    """Return the jax process-group size (1 in a single process)."""
    return jax.process_count()


def _barrier(tag: str) -> None:
    """
    Synchronize the process group at a named point (multi-process).

    Description
    -----------
    Orders rank 0's metadata mutations against the other ranks' shard
    writes. A no-op when ``_process_count() == 1``, so the
    single-controller path takes no synchronization and its store
    output stays byte-for-byte identical.

    Parameters
    ----------
    tag : str
        A label identifying the barrier; must match across processes.
    """
    if _process_count() > 1:
        from jax.experimental import multihost_utils  # noqa: PLC0415
        multihost_utils.sync_global_devices(tag)


# ================================================================
#  Unit-factor metadata (the §D writer stamp)
# ================================================================
class _UnitsStamp(NamedTuple):

    """
    The bind-time snapshot of the model's unit-factor metadata.

    Description
    -----------
    Computed once at bind (``_units_stamp``) — the one deliberate
    snapshot in the §D design (the live surface is
    ``model.units``): identical on every rank, written by rank 0.

    Parameters
    ----------
    global_attrs : dict
        The ``fridom_scaling*`` store-level attributes.
    per_name : dict
        Per-row ``dimensional_factor`` attribute dicts, keyed by
        the factor-row name (components, coordinates, ``"t"``).
    nondimensional : bool
        Whether the model is nondimensional (drives the CF
        time-axis option-(b) rewrite).
    """

    global_attrs: dict[str, Any]
    per_name: dict[str, dict[str, Any]]
    nondimensional: bool


def _units_stamp(model: Any) -> _UnitsStamp | None:
    """
    Build the unit-factor metadata stamp of a model (or ``None``).

    Description
    -----------
    Pure and best-effort (metadata changes no data, so it must
    never fail a run): a model without a ``units`` surface — or one
    whose surface errors — yields ``None`` and the store is written
    byte-identically to the pre-§D layout. Per row:
    ``dimensional_factor`` (omitted when unresolvable),
    ``dimensional_units`` and ``dimensional_factor_expr`` (always),
    ``dimensional_factor_time_dependent`` only for Ramp-valued
    rows. Globally: the scaling class, variant flag, stored
    reference scales, the constant-resolvable ``T_ref`` and
    ``epsilon``, and the bound constants the tables reference.
    """
    try:
        units = model.units
        entries = dict(units.factors)
        per_name = {name: _stamp_row(entry)
                    for name, entry in entries.items()}
        scaling = getattr(model, "scaling", None)
        nondimensional = bool(
            getattr(scaling, "nondimensional", False))
        global_attrs = _stamp_globals(
            units, scaling, nondimensional, entries)
        return _UnitsStamp(global_attrs, per_name, nondimensional)
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return None


def _stamp_row(entry: Any) -> dict[str, Any]:
    """Build one row's ``dimensional_factor`` attribute dict."""
    attrs: dict[str, Any] = {}
    value = getattr(entry, "value", None)
    if value is not None:
        attrs["dimensional_factor"] = float(value)
    attrs["dimensional_units"] = str(getattr(entry, "unit", ""))
    attrs["dimensional_factor_expr"] = str(
        getattr(entry, "expr", ""))
    if getattr(entry, "time_dependent", False):
        attrs["dimensional_factor_time_dependent"] = True
    return attrs


def _stamp_globals(
    units: Any, scaling: Any, nondimensional: bool,
    entries: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the ``fridom_scaling*`` store-level attribute dict."""
    global_attrs: dict[str, Any] = {
        "fridom_scaling_nondimensional": nondimensional}
    if scaling is not None:
        global_attrs["fridom_scaling"] = type(scaling).__name__
        for scale in ("L", "U", "g"):
            stored = getattr(scaling, scale, None)
            if stored is not None:
                global_attrs[f"fridom_scaling_{scale}"] = float(
                    stored)
    t_ref = entries.get("T_ref")
    if getattr(t_ref, "value", None) is not None:
        global_attrs["fridom_scaling_T_ref"] = float(t_ref.value)
    bound = getattr(units, "bound_constants", None)
    constants = dict(bound()) if callable(bound) else {}
    if constants:
        global_attrs["fridom_scaling_parameters"] = {
            name: float(value)
            for name, value in constants.items()}
    epsilon = constants.get("scaling.nonlinearity")
    if epsilon is not None:
        global_attrs["fridom_scaling_epsilon"] = float(epsilon)
    return global_attrs


def _strip_position(dim: str) -> str:
    """Strip the xgcm stagger suffix off an exported dim name."""
    for position in _POSITIONS:
        suffix = f"_{position}"
        if dim.endswith(suffix):
            return dim[: -len(suffix)]
    return dim


# ================================================================
#  Writer
# ================================================================
class Writer:

    """
    Trigger-driven gridded output stream (zarr, xarray/xgcm-openable).

    Description
    -----------
    Reusable configuration until :meth:`bind`, a single store after
    it (host infrastructure — never a pytree, never traced, never
    hashed into the restart fingerprint). Implements the
    ``OutputStream`` protocol.

    Parameters
    ----------
    path : str | Path
        The zarr store path (the stream owns it exclusively).
    fields : Sequence[str] | None, optional
        Field names to write. ``None`` selects the lifecycle default:
        every PROGNOSTIC and DIAGNOSTIC field; AUXILIARY is opt-in by
        name (default: None).
    derived : Mapping[str, Callable] | None, optional
        Named derived outputs, each a pure ``(model_state) -> Field``
        callable evaluated at output cadence (default: None).
    trigger : Trigger
        The firing trigger (walltime-bearing triggers are rejected at
        bind — data streams need a deterministic output grid).
    mode : {"w", "w-", "a"}, optional
        Store creation mode. ``"w-"`` creates and fails if the path
        exists (the confirm-at-first-use default: failing loudly
        beats clobbering a previous segment); ``"w"`` overwrites;
        ``"a"`` appends to an existing store. On snapshot resume the
        run machinery flips a bound writer to ``"a"`` and calls
        :meth:`truncate_after` (default: "w-").
    chunks : Mapping[str, int] | None, optional
        Per-dimension zarr chunk sizes, keyed by dim name (``"time"``
        and the exported spatial dims); unset dims chunk whole
        (spatial) or in single steps (``"time"``) (default: None).
    attrs : Mapping[str, str] | None, optional
        Extra global attributes merged into the store (default: None).
    units_metadata : bool, optional
        Stamp the model's unit-factor metadata (§D) into the store:
        ``fridom_scaling*`` global attributes plus per-variable /
        coordinate / time ``dimensional_factor`` (+ unit / expr)
        attributes, snapshotted at bind. On a nondimensional model
        the CF time coordinate's ``units`` becomes ``"1"`` and the
        calendar anchor is dropped (model time is in units of
        ``T_ref``; ``dimensional_factor = T_ref`` is stamped
        alongside — owner ruling, option b). It changes no data;
        models without a ``units`` surface are stamped with nothing
        (default: True).
    async_writes : bool, optional
        Overlap the spatial disk writes with the model integration:
        each firing defers its writes and drains the previous firing
        first (one firing in flight — see :meth:`write`), instead of
        blocking on every write before returning. Every firing still
        commits its ``iteration`` label last, so a hard kill leaves the
        store crash-consistent (the interrupted tail slice is dropped
        on reopen); the default additionally blocks, so no output at
        all trails the model on a clean abort. Ignored under a real
        multi-process run (``jax.process_count() > 1``): deferring
        writes past the per-firing barriers would break the
        one-firing-in-flight invariant, so a distributed writer always
        blocks (single-process async is unchanged) (default: False).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fields: Sequence[str] | None = None,
        derived: Mapping[str, Callable] | None = None,
        trigger: Trigger,
        mode: str = "w-",
        chunks: Mapping[str, int] | None = None,
        attrs: Mapping[str, str] | None = None,
        units_metadata: bool = True,
        async_writes: bool = False,
    ) -> None:
        """Configure the stream; no file IO happens here."""
        if mode not in _MODES:
            raise ValueError(
                f"mode must be one of {_MODES}, got {mode!r}")
        self._path = Path(path)
        self._fields = None if fields is None else tuple(fields)
        self._derived = dict(derived) if derived else {}
        self._trigger = trigger
        self.mode = mode
        self._chunks = dict(chunks) if chunks else {}
        self._attrs = dict(attrs) if attrs else {}
        self._units_metadata = bool(units_metadata)
        # the bind-time unit-factor stamp (None until bind / opted out)
        self._units: _UnitsStamp | None = None
        self._async_writes = bool(async_writes)
        # bind state (all reset until bind)
        self._bound = False
        self._n = 0
        # multi-process coordination (resolved at bind): rank 0 owns
        # every metadata mutation, and async falls back to blocking
        # when distributed (see :meth:`write`).
        self._rank0 = True
        self._distributed = False
        self._async_effective = self._async_writes
        self._time: Any = None
        self._iteration: Any = None
        self._vars: dict[str, Any] = {}
        self._spatial: dict[str, tuple[int, ...]] = {}
        # values-free export layouts, keyed by output name (bind)
        self._templates: dict[str, ExportLayout] = {}
        # ordered (out_name, evaluator) pairs, resolved at bind
        self._outputs: list[tuple[str, Callable]] = []
        # the last firing's deferred work when async_writes is on (one
        # firing in flight): (var_writes, nt, time_value, it_value),
        # committed at the next firing and at truncate_after / close.
        # None in the blocking default. The var-write source arrays are
        # retained until their writes commit (tensorstore requires the
        # source valid until the copy is done).
        self._pending: tuple[list[tuple[Any, Any]], int, float, int]\
            | None = None

    # ================================================================
    #  Protocol slots
    # ================================================================
    @property
    def trigger(self) -> Trigger:
        """The stream's firing trigger (protocol slot)."""
        return self._trigger

    @property
    def path(self) -> Path:
        """The resolved store path."""
        return self._path

    def bind(self, model: Any) -> None:
        """
        Resolve fields/derived, dry-evaluate, create the store.

        Description
        -----------
        Called at RUN START (never at assembly). Rejects a
        walltime-bearing trigger, resolves the write set against the
        model's field table (lifecycle defaults or explicit
        ``fields=``), dry-evaluates every output on the model's carry
        (validating that no output is coefficient-space or complex),
        and creates/opens the store per :attr:`mode` — writing the
        static coordinates once.

        Parameters
        ----------
        model : fr.Model
            The bound model (duck-typed: ``carry`` — the model_state
            with ``state``/``clock`` — and, for the lifecycle default,
            a ``field_table``).
        """
        if self._bound:
            raise RuntimeError(
                f"this Writer is already bound to {self._path}; a "
                "Writer owns one store after bind (build a fresh "
                "Writer for a second store)")
        reject_walltime_trigger(
            self._trigger, stream=f"Writer({self._path})")
        self._rank0 = _process_index() == 0
        self._distributed = _process_count() > 1
        self._async_effective = (
            self._async_writes and not self._distributed)
        model_state = self._model_state(model)
        self._outputs = self._resolve_outputs(model, model_state)
        templates: dict[str, ExportLayout] = {}
        hints: dict[str, tuple[int, ...]] = {}
        for name, evaluate in self._outputs:
            field = evaluate(model_state)
            layout = self._layout(field, name)
            templates[name] = layout
            hints[name] = self._chunk_hint(field, layout)
        self._templates = templates
        self._open_store(templates, hints, model_state, model)
        self._bound = True

    def write(self, model_state: Any) -> None:
        """
        Append one time slice from the boundary-synced carry.

        Description
        -----------
        The spatial variables write shard-wise true-DOF tiles via
        ``shard_writes`` (no gather anywhere). Every firing commits in
        a fixed order — **variable tiles first, then the time label,
        then the iteration label last** (:meth:`_commit`) — so the
        ``iteration`` axis is the durable commit marker: a real
        iteration value implies its whole slice is on disk. An
        interrupted slice keeps the iteration fill sentinel and is
        dropped on reopen (:meth:`_reopen`) / :meth:`truncate_after`.

        By default the whole firing commits before returning, so no
        output trails the model. With ``async_writes`` the variable
        writes are **deferred**: this firing commits the previous one
        first (backpressure — one firing in flight, so the source
        buffers cannot pile up), then issues its own variable writes,
        which commit in the background while the model integrates
        toward the next firing (where they, then its labels, commit).
        The device->host copy in ``shard_writes`` is unaffected — it
        stays on the calling thread; only the disk commit overlaps.

        Parameters
        ----------
        model_state : object
            The synced carry: ``state`` (the component fields) plus
            ``clock`` (the time/iteration axes are written from it).
        """
        self._require_bound()
        clock = model_state.clock
        nt = self._n
        if self._distributed:
            self._write_distributed(nt, model_state, clock)
        else:
            self._write_local(nt, model_state, clock)
        self._n = nt + 1

    def _write_local(
        self, nt: int, model_state: Any, clock: Any,
    ) -> None:
        """
        Single-process append (the default blocking/async path).

        Description
        -----------
        The original single-controller write: one process owns every
        array, grows each axis up front (the new slot reads the
        iteration fill sentinel until its firing commits), issues the
        shard writes, and commits — blocking, or deferring one firing
        when ``async_writes`` is on.
        """
        # grow every axis up front (the new slot reads the iteration
        # fill sentinel until its firing commits, keeping the store
        # openable with consistent dims); the labels are written last.
        self._time = _grow(self._time, (nt + 1,))
        self._iteration = _grow(self._iteration, (nt + 1,))
        if self._async_effective:
            self._drain()  # commit the previous firing (one in flight)
        var_writes: list[tuple[Any, Any]] = []
        for name, evaluate in self._outputs:
            shape = (nt + 1, *self._spatial[name])
            self._vars[name] = _grow(self._vars[name], shape)
            self._write_shards(nt, name, evaluate(model_state), var_writes)
        firing = (var_writes, nt, float(np.asarray(clock.time)),
                  int(np.asarray(clock.it)))
        if self._async_effective:
            self._pending = firing
        else:
            self._commit(firing)

    def _write_distributed(
        self, nt: int, model_state: Any, clock: Any,
    ) -> None:
        """
        Coordinated blocking append across a multi-process group.

        Description
        -----------
        Rank 0 owns every metadata mutation: it grows ``time``,
        ``iteration`` and each variable to ``nt + 1``, then a barrier
        lets the non-rank-0 recheck handles observe the grown shape.
        Every rank then writes and blocks on its own disjoint shard
        tiles; a second barrier orders that data before rank 0 writes
        the ``time`` then ``iteration`` labels (the commit marker,
        last), so the ``iteration`` axis stays the durable marker for a
        fully-committed slice across the whole group.
        """
        if self._rank0:
            self._time = _grow(self._time, (nt + 1,))
            self._iteration = _grow(self._iteration, (nt + 1,))
            for name in self._vars:
                self._vars[name] = _grow(
                    self._vars[name], (nt + 1, *self._spatial[name]))
        _barrier("writer-resized")
        var_writes: list[tuple[Any, Any]] = []
        for name, evaluate in self._outputs:
            self._write_shards(nt, name, evaluate(model_state), var_writes)
        _block_writes(var_writes, self._path, nt)
        _barrier("writer-data")
        if self._rank0:
            self._time[nt] = float(np.asarray(clock.time))
            self._iteration[nt] = int(np.asarray(clock.it))

    def _commit(
        self, firing: tuple[list[tuple[Any, Any]], int, float, int],
    ) -> None:
        """
        Commit one firing: variable tiles, then labels (iteration last).

        Description
        -----------
        Blocks on the firing's variable writes, then writes ``time``
        and finally ``iteration`` — so the iteration label lands only
        after everything it labels is durable. A crash before this
        finishes leaves the slice's iteration at the fill sentinel,
        which reopen/truncate treat as never-written.
        """
        var_writes, nt, time_value, it_value = firing
        _block_writes(var_writes, self._path, nt)
        self._time[nt] = time_value
        self._iteration[nt] = it_value

    def _drain(self) -> None:
        """Commit the deferred previous firing, if any (async)."""
        pending, self._pending = self._pending, None
        if pending is not None:
            self._commit(pending)

    def _write_shards(
        self, nt: int, name: str, field: ScalarField,
        writes: list[tuple[Any, Any]],
    ) -> None:
        """
        Queue the async shard writes of one output at time index `nt`.

        Description
        -----------
        Each locally-owned true-DOF tile (``decomposition.shard_writes``)
        is mapped from storage axes to the exported dims through
        ``layout.kept_axes``: the dropped (constant-factor) positions
        are squeezed off ``values``, the same positions are dropped from
        the target slices, and the surviving window is written into the
        ``[nt]`` slice. Zero-extent tiles (the empty last shard of
        ``Inner`` / bounded ``FaceAvg`` spaces on non-divisible cell
        counts) are skipped.

        Parameters
        ----------
        nt : int
            The time index of this firing.
        name : str
            The output variable name.
        field : ScalarField
            The evaluated field to write.
        writes : list
            The per-firing accumulator of ``(write_future, source)``
            pairs (blocked on in ``write`` or ``_drain``); the source
            tile is retained so it stays valid until its write commits.
        """
        layout = self._templates[name]
        space = field.function_space
        decomp = field.grid.decomposition
        var = self._vars[name]
        for target, values in decomp.shard_writes(
                field._data, space):  # noqa: SLF001 — storage seam
            dropped = tuple(axis for axis in range(len(target))
                            if axis not in layout.kept_axes)
            block = np.squeeze(values, axis=dropped) if dropped else values
            if block.size == 0:
                continue  # empty last shard (Inner / bounded FaceAvg)
            kept = tuple(target[axis] for axis in layout.kept_axes)
            writes.append((var[(nt, *kept)].write(block), block))

    def truncate_after(self, iteration: int) -> None:
        """
        Drop appended slices past the given iteration coordinate.

        Description
        -----------
        Iteration-keyed resume alignment (V-S1): the ``iteration``
        coordinate is monotone in both time directions, so the kept
        prefix is every slice whose iteration is ``<= iteration``.
        The store stays open for continued appends (fork-free resume).

        Parameters
        ----------
        iteration : int
            The snapshot iteration; slices past it are dropped.
        """
        self._require_bound()
        # deferred writes must commit before the arrays are resized down
        self._drain()
        stored = np.asarray(self._iteration.read().result())
        # ignore a trailing un-committed slice (fill sentinel) from a
        # prior crash — the store is trimmed to the committed prefix
        n = _committed_length(stored)
        keep = int(np.searchsorted(stored[:n], iteration, side="right"))
        # ``keep`` is derived identically on every rank (shared
        # filesystem, process-local read). Only rank 0 performs the
        # down-resize; the read barrier keeps no rank reading across
        # rank 0's trim, and the trim barrier makes the shorter length
        # visible (the non-rank-0 recheck handles) before any append.
        _barrier("writer-truncate-read")
        if self._rank0:
            self._time = _grow(self._time, (keep,))
            self._iteration = _grow(self._iteration, (keep,))
            for name in self._vars:
                shape = (keep, *self._spatial[name])
                self._vars[name] = _grow(self._vars[name], shape)
        _barrier("writer-truncate-trim")
        self._n = keep

    def close(self) -> None:
        """
        Release the store; partial output survives.

        Description
        -----------
        Idempotent. Any deferred writes are drained first, so the
        store on disk is complete before the handles drop; an abort
        before ``close`` still leaves every already-committed slice
        readable.
        """
        if not self._bound:
            return
        self._drain()
        # no rank races ahead of another's final commit before handles drop
        _barrier("writer-close")
        self._bound = False
        self._time = None
        self._iteration = None
        self._vars = {}
        self._spatial = {}
        self._templates = {}
        self._pending = None

    # ================================================================
    #  Bind helpers — output resolution
    # ================================================================
    @staticmethod
    def _model_state(model: Any) -> Any:
        """Return the model's carry (the dry-evaluation model_state)."""
        carry = getattr(model, "carry", None)
        if carry is None:
            raise TypeError(
                "Writer.bind expects a model exposing a carry "
                "(model_state with state/clock); got "
                f"{type(model).__name__}")
        return carry

    def _resolve_outputs(
        self, model: Any, model_state: Any,
    ) -> list[tuple[str, Callable]]:
        """Resolve the ordered (name, evaluator) write set."""
        state = getattr(model_state, "state", None)
        names = self._resolve_field_names(model, state)
        outputs: list[tuple[str, Callable]] = [
            (name, _state_getter(name)) for name in names]
        chosen = dict(outputs)
        for name, function in self._derived.items():
            if name in chosen:
                raise ValueError(
                    f"derived output {name!r} collides with a "
                    "selected field name; give the derived output a "
                    "distinct key")
            outputs.append((name, function))
        if not outputs:
            raise ValueError(
                f"Writer({self._path}) resolved no outputs; pass "
                "fields=/derived= or declare PROGNOSTIC/DIAGNOSTIC "
                "fields")
        return outputs

    def _resolve_field_names(
        self, model: Any, state: Any,
    ) -> tuple[str, ...]:
        """Explicit ``fields=`` or the PROGNOSTIC+DIAGNOSTIC default."""
        table = _field_table(model)
        if self._fields is not None:
            known = _known_names(table, state)
            unknown = tuple(n for n in self._fields
                            if known is not None and n not in known)
            if unknown:
                raise ValueError(
                    f"Writer({self._path}) was asked to write unknown "
                    f"field(s) {unknown}; the model declares "
                    f"{tuple(known)}")
            return self._fields
        if table is None:
            raise ValueError(
                f"Writer({self._path}) needs an explicit fields= list "
                "(the model exposes no field_table for the lifecycle "
                "default)")
        # lifecycle default: PROGNOSTIC + DIAGNOSTIC, AUXILIARY opt-in
        return tuple(table.prognostic) + tuple(table.diagnostic)

    def _layout(self, field: ScalarField, name: str) -> ExportLayout:
        """Reject unwritable fields; return the values-free layout."""
        space = field.function_space
        from fridom.spatial.scalars import (  # noqa: PLC0415
            Scalars,
        )
        from fridom.spatial.spaces.coefficient import (  # noqa: PLC0415
            CoefficientSpace,
        )
        if any(isinstance(f, CoefficientSpace) for f in space.factors):
            raise NotImplementedError(
                f"Writer cannot write the coefficient-space output "
                f"{name!r}: spectral coefficients have no xarray/xgcm "
                "layout in iteration 1 (transform back first, or read "
                "field.data directly)")
        if space.scalars is Scalars.COMPLEX:
            raise NotImplementedError(
                f"Writer cannot write the complex output {name!r}: "
                "zarr/CF has no complex layout in iteration 1 (write "
                "field.real/field.imag, or read field.data directly)")
        from fridom.spatial.export import export_layout  # noqa: PLC0415
        return export_layout(field)

    @staticmethod
    def _chunk_hint(
        field: ScalarField, layout: ExportLayout,
    ) -> tuple[int, ...]:
        """
        Default per-dim spatial chunks from the decomposition.

        Description
        -----------
        The decomposition's write-aligned chunk grid
        (``chunk_hint`` — per-shard cells on a blocked axis, the full
        extent on an unblocked one) mapped to the exported dims through
        ``layout.kept_axes``. Under this default every
        ``shard_writes`` tile is chunk-aligned, so no write reads a
        chunk back. On a single device it equals the full spatial
        shape (the previous default). Explicit user ``chunks=`` still
        wins per dim.
        """
        hint = field.grid.decomposition.chunk_hint(
            field.function_space)
        return tuple(hint[axis] for axis in layout.kept_axes)

    # ================================================================
    #  Bind helpers — store creation
    # ================================================================
    def _open_store(
        self, templates: Mapping[str, ExportLayout],
        hints: Mapping[str, tuple[int, ...]], model_state: Any,
        model: Any,
    ) -> None:
        """Create (or reopen for append) the tensorstore zarr store."""
        # the §D unit-factor stamp: computed identically on every
        # rank at bind (the one deliberate snapshot); rank 0 writes
        # it below. A mode="a" reopen keeps the existing store
        # metadata untouched (no restamp).
        self._units = (_units_stamp(model)
                       if self._units_metadata else None)
        exists = self._path.exists()
        # The existence check + mode decision is evaluated identically
        # on every rank (shared filesystem) before any rank mutates the
        # store, so all ranks agree on the branch taken below.
        if self.mode == "w-" and exists:
            raise FileExistsError(
                f"Writer({self._path}) is mode='w-' and the store "
                "already exists; failing loudly (use mode='w' to "
                "overwrite or mode='a' to append)")
        if self.mode == "a" and exists:
            self._reopen(templates)
            return
        # every rank has evaluated ``exists``; barrier before rank 0
        # mutates so no rank's check races rank 0's create/clobber.
        _barrier("writer-exists")
        if self._rank0:
            if self.mode == "w" and exists:
                shutil.rmtree(self._path)
            self._path.mkdir(parents=True, exist_ok=True)
            self._write_group(model)
            self._write_coords(templates)
            self._write_time_axis(model_state)
            self._write_variables(templates, hints)
        _barrier("writer-skeleton")
        if not self._rank0:
            self._open_skeleton(templates)
        self._n = 0

    def _open_skeleton(
        self, templates: Mapping[str, ExportLayout],
    ) -> None:
        """
        Open the rank-0-created skeleton on a non-rank-0 process.

        Description
        -----------
        Rank 0 owns store creation; the other ranks open (never create)
        the ``time``/``iteration``/variable arrays with
        ``recheck_cached_metadata`` so they observe rank 0's later
        resizes, and record the spatial shapes locally off the layouts.
        """
        self._time = _open_array(self._path / "time", recheck=True)
        self._iteration = _open_array(
            self._path / "iteration", recheck=True)
        for name, layout in templates.items():
            self._vars[name] = _open_array(
                self._path / name, recheck=True)
            self._spatial[name] = tuple(int(s) for s in layout.shape)

    def _reopen(self, templates: Mapping[str, ExportLayout]) -> None:
        """Reopen an existing store for append (resume)."""
        # non-rank-0 handles recheck so they see rank 0's down-resize
        recheck = self._distributed and not self._rank0
        self._time = _open_array(self._path / "time", recheck=recheck)
        self._iteration = _open_array(
            self._path / "iteration", recheck=recheck)
        for name, layout in templates.items():
            store = (self._path / name)
            if not store.exists():
                raise ValueError(
                    f"cannot append to {self._path}: it has no "
                    f"variable {name!r} (schema mismatch on resume)")
            self._vars[name] = _open_array(store, recheck=recheck)
            self._spatial[name] = tuple(int(s) for s in layout.shape)
        # a prior async run may have died mid-firing, leaving a trailing
        # slice whose iteration is the fill sentinel; drop it so the
        # append is gap-free and every array shares one time length. The
        # committed length is read identically on every rank; only rank
        # 0 down-resizes, bracketed by barriers (no rank reads across
        # the trim; the trimmed length is visible before any append).
        n = _committed_length(
            np.asarray(self._iteration.read().result()))
        _barrier("writer-reopen-read")
        if self._rank0:
            self._time = _grow(self._time, (n,))
            self._iteration = _grow(self._iteration, (n,))
            for name in self._vars:
                self._vars[name] = _grow(
                    self._vars[name], (n, *self._spatial[name]))
        _barrier("writer-reopen-trim")
        self._n = n

    def _write_group(self, model: Any) -> None:
        """Write ``.zgroup`` and the global ``.zattrs`` (provenance)."""
        _write_json(self._path / ".zgroup", {"zarr_format": 2})
        attrs: dict[str, Any] = {"Conventions": "CF-1.10"}
        digest = getattr(
            getattr(model, "fingerprint", None), "digest", None)
        if digest is not None:
            attrs["fridom_fingerprint"] = digest
        version = _fridom_version()
        if version is not None:
            attrs["fridom_version"] = version
        if self._units is not None:
            attrs.update(self._units.global_attrs)
        attrs.update(self._attrs)  # user attrs win
        _write_json(self._path / ".zattrs", attrs)

    def _write_coords(
        self, templates: Mapping[str, ExportLayout],
    ) -> None:
        """Write the static spatial coordinate arrays once."""
        seen: set[str] = set()
        for layout in templates.values():
            for dim in layout.dims:
                if dim in seen:
                    continue
                seen.add(dim)
                values = np.asarray(layout.coords[dim])
                store = _create_array(
                    self._path / dim, values.shape,
                    (max(1, values.shape[0]),), values.dtype)
                store[...] = values
                zattrs = {"_ARRAY_DIMENSIONS": [dim]}
                zattrs.update(layout.coord_attrs[dim])
                if self._units is not None:
                    # the stagger suffix strips back to the factor
                    # row name; no matching row -> no stamp
                    zattrs.update(self._units.per_name.get(
                        _strip_position(dim), {}))
                _write_json(self._path / dim / ".zattrs", zattrs)

    def _write_time_axis(self, model_state: Any) -> None:
        """Create the empty CF time axis + the iteration coordinate."""
        chunk = max(1, int(self._chunks.get("time", 1)))
        self._time = _create_array(
            self._path / "time", (0,), (chunk,), np.dtype(np.float64))
        time_attrs = {"_ARRAY_DIMENSIONS": ["time"]}
        time_attrs.update(_time_attrs(model_state.clock))
        if self._units is not None:
            time_attrs.update(self._units.per_name.get("t", {}))
            if self._units.nondimensional:
                # CF option (b), owner ruling 2026-07-21: model time
                # is in units of T_ref, so the old "seconds since"
                # claim was dimensionally false — declare the axis
                # dimensionless and drop the calendar anchor (the
                # dimensional_factor above carries T_ref)
                time_attrs["units"] = "1"
                time_attrs.pop("calendar", None)
        _write_json(self._path / "time" / ".zattrs", time_attrs)
        self._iteration = _create_array(
            self._path / "iteration", (0,), (chunk,),
            np.dtype(np.int64), fill_value=_UNWRITTEN)
        _write_json(
            self._path / "iteration" / ".zattrs",
            {"_ARRAY_DIMENSIONS": ["time"], "long_name": "iteration"})

    def _write_variables(
        self, templates: Mapping[str, ExportLayout],
        hints: Mapping[str, tuple[int, ...]],
    ) -> None:
        """Create the empty (time, *space) variable arrays."""
        time_chunk = max(1, int(self._chunks.get("time", 1)))
        for name, layout in templates.items():
            spatial = tuple(int(s) for s in layout.shape)
            self._spatial[name] = spatial
            # default spatial chunks = the decomposition's write-aligned
            # grid (chunk_hint); explicit user chunks= win per dim.
            chunks = (time_chunk, *(
                max(1, int(self._chunks.get(dim, default)))
                for dim, default in zip(
                    layout.dims, hints[name], strict=True)))
            self._vars[name] = _create_array(
                self._path / name, (0, *spatial), chunks,
                layout.dtype)
            zattrs = {"_ARRAY_DIMENSIONS": ["time", *layout.dims]}
            zattrs.update(layout.attrs)
            if self._units is not None:
                zattrs.update(self._units.per_name.get(name, {}))
            # CF auxiliary-coordinate promotion: xarray reads the
            # iteration variable back as a coordinate, no post-proc.
            zattrs["coordinates"] = "iteration"
            _write_json(self._path / name / ".zattrs", zattrs)

    # ================================================================
    #  Small utilities
    # ================================================================
    def _require_bound(self) -> None:
        """Raise if used before bind."""
        if not self._bound:
            raise RuntimeError(
                f"Writer({self._path}) is not bound; call bind(model) "
                "at run start before write/truncate_after")


# ================================================================
#  tensorstore helpers (zarr v2 driver; no zarr import)
# ================================================================
def _create_array(
    path: Path, shape: tuple[int, ...], chunks: tuple[int, ...],
    dtype: np.dtype, *, fill_value: float | None = None,
) -> Any:
    """Create (overwriting) a zarr-v2 array via tensorstore."""
    import tensorstore as ts  # noqa: PLC0415
    metadata: dict[str, Any] = {
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": np.dtype(dtype).str,
    }
    if fill_value is not None:
        metadata["fill_value"] = fill_value
    return ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": metadata,
        "create": True,
        "delete_existing": True,
    }).result()


def _open_array(path: Path, *, recheck: bool = False) -> Any:
    """
    Open an existing zarr-v2 array via tensorstore (append/read).

    Description
    -----------
    ``recheck`` sets ``recheck_cached_metadata: true`` so the handle
    re-reads the ``.zarray`` metadata on every operation. Non-rank-0
    handles open with it in a multi-process run so they observe rank
    0's resizes; the single-process default keeps the cached metadata.

    Parameters
    ----------
    path : Path
        The zarr array directory.
    recheck : bool, optional
        Re-read the array metadata on each access (default: False).

    Returns
    -------
    Any
        The opened tensorstore handle.
    """
    import tensorstore as ts  # noqa: PLC0415
    spec: dict[str, Any] = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
        "open": True,
    }
    if recheck:
        spec["recheck_cached_metadata"] = True
    return ts.open(spec).result()


def _grow(store: Any, shape: tuple[int, ...]) -> Any:
    """Resize a tensorstore array to ``shape`` (grow or shrink)."""
    return store.resize(exclusive_max=list(shape)).result()


def _block_writes(
    writes: list[tuple[Any, Any]], path: Path, nt: int,
) -> None:
    """
    Block until every queued write has committed.

    Description
    -----------
    Waits on each ``(write_future, source)`` pair's commit future; the
    source tile is held only to keep it valid until the write is done
    (tensorstore borrows the source until then) and is otherwise
    unused here. A failed write (deferred in ``async_writes`` mode)
    surfaces at the drain that awaits it, so it is re-raised with the
    store path and the step index it belongs to.
    """
    for future, _source in writes:
        try:
            future.result()
        except Exception as error:
            msg = (f"Writer({path}) failed to commit an output write "
                   f"at step {nt}")
            raise RuntimeError(msg) from error


def _committed_length(iterations: np.ndarray) -> int:
    """
    Count the committed slices in an ``iteration`` axis.

    Description
    -----------
    Slices commit their iteration label last, so the committed prefix
    runs up to the first fill sentinel (``_UNWRITTEN``) — the marker of
    a firing interrupted by a crash. A clean store has no sentinel and
    the whole axis counts.
    """
    unwritten = np.flatnonzero(iterations == _UNWRITTEN)
    return int(unwritten[0]) if unwritten.size else int(iterations.size)


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    """Write one zarr JSON sidecar (``.zgroup``/``.zattrs``)."""
    with path.open("w") as handle:
        json.dump(dict(obj), handle)


# ================================================================
#  Module-level helpers
# ================================================================
def _state_getter(name: str) -> Callable:
    """Return an evaluator reading one component off the state."""
    def evaluate(model_state: Any) -> ScalarField:
        return model_state.state[name]
    return evaluate


def _field_table(model: Any) -> Any:
    """Duck-typed field-table access (public, then assembly seam)."""
    table = getattr(model, "field_table", None)
    if table is not None:
        return table
    artifacts = getattr(model, "_artifacts", None)
    return getattr(artifacts, "field_table", None)


def _known_names(table: Any, state: Any) -> tuple[str, ...] | None:
    """Return the known field names for validation (table or state)."""
    if table is not None:
        return tuple(table.names)
    if state is not None:
        return tuple(state.component_names)
    return None


def _time_attrs(clock: Any) -> dict[str, str]:
    """CF time-axis attributes; a calendar anchor when present."""
    attrs = {
        "long_name": "time",
        "standard_name": "time",
        "axis": "T",
    }
    start_date = getattr(clock, "start_date", None)
    if start_date is None:
        attrs["units"] = "seconds"
        return attrs
    start = float(np.asarray(clock.start))
    reference = np.datetime64(start_date) - np.timedelta64(
        round(start * _NS_PER_S), "ns")
    # Whole-second reference date: ns precision breaks CF readers
    # (e.g. Julia's CFTime overflows on nanosecond epochs).
    reference = reference.astype("datetime64[s]")
    attrs["units"] = f"seconds since {reference}"
    attrs["calendar"] = "proleptic_gregorian"
    return attrs


def _fridom_version() -> str | None:
    """Best-effort fridom package version for provenance."""
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version("fridom")
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return None
