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
from typing import TYPE_CHECKING, Any

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
    async_writes : bool, optional
        Overlap the spatial disk writes with the model integration:
        each firing defers its writes and drains the previous firing
        first (one firing in flight — see :meth:`write`), instead of
        blocking on every write before returning. The default blocks,
        so an aborted run always leaves a complete final slice
        (default: False).
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
        self._async_writes = bool(async_writes)
        # bind state (all reset until bind)
        self._bound = False
        self._n = 0
        self._time: Any = None
        self._iteration: Any = None
        self._vars: dict[str, Any] = {}
        self._spatial: dict[str, tuple[int, ...]] = {}
        # values-free export layouts, keyed by output name (bind)
        self._templates: dict[str, ExportLayout] = {}
        # ordered (out_name, evaluator) pairs, resolved at bind
        self._outputs: list[tuple[str, Callable]] = []
        # deferred (write_future, source) pairs of the last firing when
        # async_writes is on (one firing in flight); drained at the next
        # firing and at truncate_after / close. Empty in the blocking
        # default. The source array is retained until its write commits
        # (tensorstore requires the source valid until the copy is done).
        self._pending: list[tuple[Any, Any]] = []

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
        The scalar time/iteration axes always write synchronously
        (tiny, and :meth:`truncate_after` reads the iteration axis
        back). The spatial variables write shard-wise true-DOF tiles
        via ``shard_writes`` (no gather anywhere).

        By default every write is blocked on before returning, so an
        aborted run leaves a complete final slice. With
        ``async_writes`` the tensorstore writes are **deferred**: this
        firing drains the previous firing first (backpressure — one
        firing in flight, so the source buffers cannot pile up) and
        then queues its own writes, which commit in the background
        while the model integrates toward the next firing. The
        device->host copy in ``shard_writes`` is unaffected — it stays
        on the calling thread; only the disk commit overlaps.

        Parameters
        ----------
        model_state : object
            The synced carry: ``state`` (the component fields) plus
            ``clock`` (the time/iteration axes are written from it).
        """
        self._require_bound()
        clock = model_state.clock
        nt = self._n
        self._time = _grow(self._time, (nt + 1,))
        self._time[nt] = float(np.asarray(clock.time))
        self._iteration = _grow(self._iteration, (nt + 1,))
        self._iteration[nt] = int(np.asarray(clock.it))
        if self._async_writes:
            self._drain()  # finish the previous firing (one in flight)
        writes: list[tuple[Any, Any]] = []
        for name, evaluate in self._outputs:
            shape = (nt + 1, *self._spatial[name])
            self._vars[name] = _grow(self._vars[name], shape)
            self._write_shards(nt, name, evaluate(model_state), writes)
        if self._async_writes:
            self._pending = writes
        else:
            _block_writes(writes)
        self._n = nt + 1

    def _drain(self) -> None:
        """Block on the deferred writes of the previous firing."""
        pending, self._pending = self._pending, []
        _block_writes(pending)

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
        keep = int(np.searchsorted(stored, iteration, side="right"))
        self._time = _grow(self._time, (keep,))
        self._iteration = _grow(self._iteration, (keep,))
        for name in self._vars:
            shape = (keep, *self._spatial[name])
            self._vars[name] = _grow(self._vars[name], shape)
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
        self._bound = False
        self._time = None
        self._iteration = None
        self._vars = {}
        self._spatial = {}
        self._templates = {}
        self._pending = []

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
        exists = self._path.exists()
        if self.mode == "w-" and exists:
            raise FileExistsError(
                f"Writer({self._path}) is mode='w-' and the store "
                "already exists; failing loudly (use mode='w' to "
                "overwrite or mode='a' to append)")
        if self.mode == "a" and exists:
            self._reopen(templates)
            return
        if self.mode == "w" and exists:
            shutil.rmtree(self._path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._write_group(model)
        self._write_coords(templates)
        self._write_time_axis(model_state)
        self._write_variables(templates, hints)
        self._n = 0

    def _reopen(self, templates: Mapping[str, ExportLayout]) -> None:
        """Reopen an existing store for append (resume)."""
        self._time = _open_array(self._path / "time")
        self._iteration = _open_array(self._path / "iteration")
        for name, layout in templates.items():
            store = (self._path / name)
            if not store.exists():
                raise ValueError(
                    f"cannot append to {self._path}: it has no "
                    f"variable {name!r} (schema mismatch on resume)")
            self._vars[name] = _open_array(store)
            self._spatial[name] = tuple(int(s) for s in layout.shape)
        self._n = int(self._time.domain[0].exclusive_max)

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
        attrs.update(self._attrs)
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
                _write_json(self._path / dim / ".zattrs", zattrs)

    def _write_time_axis(self, model_state: Any) -> None:
        """Create the empty CF time axis + the iteration coordinate."""
        chunk = max(1, int(self._chunks.get("time", 1)))
        self._time = _create_array(
            self._path / "time", (0,), (chunk,), np.dtype(np.float64))
        time_attrs = {"_ARRAY_DIMENSIONS": ["time"]}
        time_attrs.update(_time_attrs(model_state.clock))
        _write_json(self._path / "time" / ".zattrs", time_attrs)
        self._iteration = _create_array(
            self._path / "iteration", (0,), (chunk,),
            np.dtype(np.int64))
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
    dtype: np.dtype,
) -> Any:
    """Create (overwriting) a zarr-v2 array via tensorstore."""
    import tensorstore as ts  # noqa: PLC0415
    return ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": np.dtype(dtype).str,
        },
        "create": True,
        "delete_existing": True,
    }).result()


def _open_array(path: Path) -> Any:
    """Open an existing zarr-v2 array via tensorstore (append/read)."""
    import tensorstore as ts  # noqa: PLC0415
    return ts.open({
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": str(path)},
        "open": True,
    }).result()


def _grow(store: Any, shape: tuple[int, ...]) -> Any:
    """Resize a tensorstore array to ``shape`` (grow or shrink)."""
    return store.resize(exclusive_max=list(shape)).result()


def _block_writes(writes: list[tuple[Any, Any]]) -> None:
    """
    Block until every queued write has committed.

    Description
    -----------
    Waits on each ``(write_future, source)`` pair's commit future; the
    source tile is held only to keep it valid until the write is done
    (tensorstore borrows the source until then) and is otherwise
    unused here.
    """
    for future, _source in writes:
        future.result()


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
