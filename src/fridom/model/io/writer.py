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
cluster's ``f.xr`` export (``fridom.spatial.export``): this
writer produces one ``xarray.DataArray`` per variable through
``scalar_to_dataarray`` and reads the labels/coords/attrs straight off
it — the xgcm-style staggered dim names, the ``c_grid_axis_shift``
comodo attrs, the coordinate values from ``grid.evaluation_nodes``,
and the per-variable ``FieldMetadata`` attrs are therefore never
duplicated here. The writer only adds the time axis (a CF
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
reader, not this writer. Async / decomposed-slice / file-split writes
stay designed-for behind this seam.

The ``decomposition.gather`` to the global true shape happens inside
``scalar_to_dataarray`` (the it-1 sink gathers to rank 0 and writes;
single process here). Coefficient-space and complex fields raise,
inheriting the ``f.xr`` iteration-1 restriction, with a pointer to
``.data``.
"""
# Wave 5 C: Writer (tensorstore zarr-append gather sink behind it)
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
        # bind state (all reset until bind)
        self._bound = False
        self._n = 0
        self._time: Any = None
        self._iteration: Any = None
        self._vars: dict[str, Any] = {}
        self._spatial: dict[str, tuple[int, ...]] = {}
        # ordered (out_name, evaluator) pairs, resolved at bind
        self._outputs: list[tuple[str, Callable]] = []

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
        templates = {
            name: self._template(evaluate(model_state), name)
            for name, evaluate in self._outputs}
        self._open_store(templates, model_state, model)
        self._bound = True

    def write(self, model_state: Any) -> None:
        """
        Append one time slice from the boundary-synced carry.

        Parameters
        ----------
        model_state : object
            The synced carry: ``state`` (the component fields) plus
            ``clock`` (the time/iteration axes are written from it).
        """
        self._require_bound()
        clock = model_state.clock
        # single process here: the decomposition.gather inside
        # scalar_to_dataarray already lands rank-0 global arrays; the
        # rank-0 write guard is the decomposed-slice sink swap's seam.
        slices = {
            name: np.asarray(self._template(
                evaluate(model_state), name).values)
            for name, evaluate in self._outputs}
        nt = self._n
        self._time = _grow(self._time, (nt + 1,))
        self._time[nt] = float(np.asarray(clock.time))
        self._iteration = _grow(self._iteration, (nt + 1,))
        self._iteration[nt] = int(np.asarray(clock.it))
        for name, values in slices.items():
            shape = (nt + 1, *self._spatial[name])
            self._vars[name] = _grow(self._vars[name], shape)
            self._vars[name][nt] = values
        self._n = nt + 1

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
        Idempotent. tensorstore writes commit synchronously, so an
        abort mid-run leaves a readable store; ``close`` only drops
        the in-memory handles.
        """
        if not self._bound:
            return
        self._bound = False
        self._time = None
        self._iteration = None
        self._vars = {}
        self._spatial = {}

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

    def _template(self, field: ScalarField, name: str) -> Any:
        """Reject unwritable fields; return the ``f.xr`` DataArray."""
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
        from fridom.spatial.export import (  # noqa: PLC0415
            scalar_to_dataarray,
        )
        return scalar_to_dataarray(field)

    # ================================================================
    #  Bind helpers — store creation
    # ================================================================
    def _open_store(
        self, templates: Mapping[str, Any], model_state: Any,
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
        self._write_variables(templates)
        self._n = 0

    def _reopen(self, templates: Mapping[str, Any]) -> None:
        """Reopen an existing store for append (resume)."""
        self._time = _open_array(self._path / "time")
        self._iteration = _open_array(self._path / "iteration")
        for name, da in templates.items():
            store = (self._path / name)
            if not store.exists():
                raise ValueError(
                    f"cannot append to {self._path}: it has no "
                    f"variable {name!r} (schema mismatch on resume)")
            self._vars[name] = _open_array(store)
            self._spatial[name] = tuple(
                int(s) for s in np.asarray(da.values).shape)
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

    def _write_coords(self, templates: Mapping[str, Any]) -> None:
        """Write the static spatial coordinate arrays once."""
        seen: set[str] = set()
        for da in templates.values():
            for dim in da.dims:
                if dim in seen:
                    continue
                seen.add(dim)
                values = np.asarray(da.coords[dim].values)
                store = _create_array(
                    self._path / dim, values.shape,
                    (max(1, values.shape[0]),), values.dtype)
                store[...] = values
                zattrs = {"_ARRAY_DIMENSIONS": [dim]}
                zattrs.update(da.coords[dim].attrs)
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

    def _write_variables(self, templates: Mapping[str, Any]) -> None:
        """Create the empty (time, *space) variable arrays."""
        time_chunk = max(1, int(self._chunks.get("time", 1)))
        for name, da in templates.items():
            values = np.asarray(da.values)
            spatial = tuple(int(s) for s in values.shape)
            self._spatial[name] = spatial
            chunks = (time_chunk, *(
                max(1, int(self._chunks.get(dim, size)))
                for dim, size in zip(da.dims, spatial, strict=True)))
            self._vars[name] = _create_array(
                self._path / name, (0, *spatial), chunks,
                values.dtype)
            zattrs = {"_ARRAY_DIMENSIONS": ["time", *da.dims]}
            zattrs.update(da.attrs)
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
