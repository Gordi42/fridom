r"""
Fused distributed channel-eigenbasis contraction (slab-decomposed).

Description
-----------
The multi-device realization of the channel-eigenmode application
``Q scale(Q^H M z)`` over the periodic Fourier planes, the sibling of
the fused spectral solve in ``operators/distributed_solve.py``. The
model-side engine (``model/_eigenbasis.py`` ``_contract_planes``)
Fourier-transforms the periodic axes with the plain (GSPMD) transform;
when the grid's default layout shards a periodic axis, XLA:GPU lowers
the sharded-transform-axis FFT through its distributed Cooley-Tukey
decomposition whose twiddle constants are ``complex64`` against the
``complex128`` data -- an upstream HLO-verifier fault
(``design/research/multidevice_test_faults.md``). This module runs the
whole forward/contract/backward pipeline inside one ``jax.shard_map``
region so every FFT axis is device-local when its transform runs, the
same manual lowering the distributed spectral solve owns:

1. the local Hermitian ``rfft`` on the engine's half axis
   (``periodic_axis``, the transpose partner ``b``) runs per shard while
   the sharded axis ``a`` is still distributed;
2. one ``jax.lax.all_to_all`` transposing the decomposition (``a``
   becomes local, the half axis ``b`` becomes sharded);
3. the ``a``-stage full ``fft`` on the now-local axis;
4. the per-plane column contraction ``Q scale(Q^H M z)`` -- an einsum
   of the (per-shard-sliced) eigenvector basis ``q`` and complex weights
   ``w`` against the coefficient planes, no reduction over the sharded
   mode axis, so no collective;
5. the mirrored inverse path (``all_to_all`` back, half-axis synthesis
   last, real part), so the output sharding equals the input sharding.

The coefficient frame is fixed by the engine: ``q`` is laid out in the
``rfftn`` frame (the full spectrum on the sharded axis ``a``, the half
spectrum on ``periodic_axis``, both in their original array positions),
so this lowering keeps that exact frame -- the half axis is never
relocated, and the transpose partner ``b`` (the half axis) is padded on
its **coefficient** extent (``n // 2 + 1``), with empty trailing pad
shards allowed (the pad lanes are transient, never stored; zero-padded
``q`` and ``w`` make their output exactly zero).

Scope: the 3-D channel -- exactly two periodic axes (the sharded axis
``a`` and the half axis ``b = periodic_axis``) plus one bounded axis,
whose position (walled x / y / z) is free. The resolution
(:func:`resolve_distributed_contraction`) is memoized per grid on static
keys only (axis names, component segment layout); ``q``, ``w`` and the
metric enter as **arguments** at apply time (dynamic; the ``shard_map``
``in_specs`` slice the replicated basis per shard, a per-device memory
win). The plan declines (returns None -- the caller keeps the existing
GSPMD path or the taught error) on a single device, a non-1-D mesh, a
layout that shards nothing / only the bounded axis, a channel without
exactly two periodic axes (the 2-D channel, or a hypothetical >3-D one),
or a layout that shards the half (``rfft``) axis itself (incompatible
with the engine's fixed half-spectrum frame under one reshard).
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.spatial.operators.transform import axis_slice

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField

#: the 3-D channel: the sharded axis ``a`` plus the half axis ``b``. A
#: single periodic axis (the 2-D channel) has no transpose partner; a
#: hypothetical >3-D channel is out of scope. Both keep the taught error.
_PERIODIC_AXES = 2


def _ceil_mult(n: int, shards: int) -> int:
    """Round ``n`` up to the nearest multiple of ``shards``."""
    return -(-n // shards) * shards


def _tail_pad(arr: jax.Array, axis: int, count: int) -> jax.Array:
    """Zero-pad ``count`` slots at the tail of ``axis`` (``count>=0``)."""
    pads = [(0, 0)] * arr.ndim
    pads[axis] = (0, count)
    return jnp.pad(arr, pads)


# ================================================================
#  The static plan (geometry + cached shard_map region)
# ================================================================
class ContractPlan:

    r"""
    One channel's slab-decomposed eigenbasis contraction pipeline.

    Description
    -----------
    Static structure resolved once per grid + component layout by
    :func:`resolve_distributed_contraction` (build through it, not this
    plumbing constructor): the device mesh, the slab geometry -- sharded
    axis ``a``, half axis ``b`` (the engine's ``rfft`` / transpose
    partner), bounded axis -- and the padded balanced all-to-all extents.
    The jit-wrapped ``jax.shard_map`` region is built here once (stable
    identity: eager re-application adds zero compiles); the eigenvector
    basis ``q``, the complex column weights ``w`` and the diagonal metric
    enter :meth:`apply` as arguments.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        The decomposition's device mesh (1-D).
    axis_name : str
        The device-mesh axis name.
    a_arr : int
        The sharded periodic coordinate's array axis (nodal frame).
    b_arr : int
        The half (``rfft``) axis's array axis (nodal component frame).
    bounded_arr : int
        The bounded (walled) coordinate's array axis.
    b_plane : int
        The half axis's array axis in the plane frame of ``q`` / ``w``
        (its periodic-axis index; the bounded axis is absorbed into the
        stacked column, so the plane frame drops it).
    half_n : int
        The half axis's true nodal extent (the ``irfft`` length).
    n_a : int
        The sharded axis's true nodal extent.
    pad_a : int
        The sharded axis's padded-even extent (``ceil_mult(n_a)``).
    n_b : int
        The half axis's true coefficient extent (``half_n // 2 + 1``).
    pad_b : int
        The half axis's padded coefficient extent.
    slices : Mapping[str, slice]
        Per-component segment slices into the stacked column axis.
    components : tuple[str, ...]
        The component names, the stacked-column segment order.
    """

    def __init__(
        self,
        mesh: jax.sharding.Mesh,
        axis_name: str,
        *,
        a_arr: int,
        b_arr: int,
        bounded_arr: int,
        b_plane: int,
        half_n: int,
        n_a: int,
        pad_a: int,
        n_b: int,
        pad_b: int,
        slices: Mapping[str, slice],
        components: tuple[str, ...],
    ) -> None:
        """Store the geometry and build the shard_map region."""
        self._mesh: jax.sharding.Mesh = mesh
        self._axis_name: str = axis_name
        self._a: int = a_arr
        self._b: int = b_arr
        self._bounded: int = bounded_arr
        self._b_plane: int = b_plane
        self._half_n: int = half_n
        self._n_a: int = n_a
        self._pad_a: int = pad_a
        self._n_b: int = n_b
        self._pad_b: int = pad_b
        self._slices: Mapping[str, slice] = slices
        self._components: tuple[str, ...] = components

        # the 3-D channel: rank-3 nodal components, rank-4 ``q`` planes
        # (two periodic mode axes + two column axes), rank-3 weights.
        nodal = [None, None, None]
        nodal[a_arr] = axis_name
        self._nodal_spec = jax.sharding.PartitionSpec(*nodal)
        q_spec = [None, None, None, None]
        q_spec[b_plane] = axis_name
        self._q_spec = jax.sharding.PartitionSpec(*q_spec)
        w_spec = [None, None, None]
        w_spec[b_plane] = axis_name
        self._w_spec = jax.sharding.PartitionSpec(*w_spec)
        self._metric_spec = jax.sharding.PartitionSpec()

        comp_specs = dict.fromkeys(components, self._nodal_spec)
        self._region = jax.jit(jax.shard_map(
            self._body, mesh=mesh,
            in_specs=(comp_specs, self._q_spec, self._w_spec,
                      self._metric_spec),
            out_specs=comp_specs))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def a_padded(self) -> bool:
        """
        Whether the sharded axis needs the padded-even field frame.

        Description
        -----------
        True when the sharded coordinate ``a`` has an indivisible extent
        (``pad_a != n_a``): :meth:`apply` then reads/writes the
        padded-even nodal frame (``decomposition.unpad_even`` /
        ``pad_even``) instead of the true frame, whose indivisible
        sharded-axis trim would gather the cube. The half axis's padding
        is transient inside the region and does not affect the field
        frame.
        """
        return self._pad_a != self._n_a

    # ================================================================
    #  The per-shard region (runs under jax.shard_map)
    # ================================================================
    def _forward_component(self, piece: jax.Array) -> jax.Array:
        """Local half rfft, all-to-all, sharded-axis fft."""
        c = jnp.fft.rfft(piece, axis=self._b, norm="forward")
        if self._pad_b != self._n_b:
            c = _tail_pad(c, self._b, self._pad_b - self._n_b)
        c = jax.lax.all_to_all(
            c, self._axis_name, split_axis=self._b,
            concat_axis=self._a, tiled=True)
        if self._pad_a != self._n_a:
            c = axis_slice(c, self._a, 0, self._n_a)
        return jnp.fft.fft(c, axis=self._a, norm="forward")

    def _backward_component(self, c: jax.Array) -> jax.Array:
        """Inverse sharded-axis fft, all-to-all back, half irfft."""
        c = jnp.fft.ifft(c, axis=self._a, norm="forward")
        if self._pad_a != self._n_a:
            c = _tail_pad(c, self._a, self._pad_a - self._n_a)
        c = jax.lax.all_to_all(
            c, self._axis_name, split_axis=self._a,
            concat_axis=self._b, tiled=True)
        if self._pad_b != self._n_b:
            c = axis_slice(c, self._b, 0, self._n_b)
        return jnp.fft.irfft(
            c, n=self._half_n, axis=self._b, norm="forward").real

    def _body(
        self,
        comps: dict[str, jax.Array],
        q: jax.Array,
        w: jax.Array,
        metric: jax.Array,
    ) -> dict[str, jax.Array]:
        """Forward, per-plane column contraction, backward (one shard)."""
        trans = {name: self._forward_component(comps[name])
                 for name in self._components}
        z = jnp.concatenate(
            [jnp.moveaxis(trans[name], self._bounded, -1)
             for name in self._components], axis=-1)
        amp = jnp.einsum("...dj,d,...d->...j", jnp.conj(q), metric, z)
        out = jnp.einsum("...dj,...j->...d", q, w * amp)
        result = {}
        for name in self._components:
            seg = jnp.moveaxis(
                out[..., self._slices[name]], -1, self._bounded)
            result[name] = self._backward_component(seg)
        return result

    # ================================================================
    #  Application
    # ================================================================
    def _pad_modes(self, arr: jax.Array) -> jax.Array:
        """Zero-pad the plane-frame half-axis mode axis to ``pad_b``."""
        if self._pad_b == self._n_b:
            return arr
        return _tail_pad(arr, self._b_plane, self._pad_b - self._n_b)

    def apply(
        self,
        fields: Mapping[str, ScalarField],
        q: jax.Array,
        weights: jax.Array,
        metric: jax.Array,
    ) -> dict[str, ScalarField]:
        r"""
        Apply ``Q diag(w) Q^H M`` on the component fields (no gather).

        Description
        -----------
        Extracts each component's nodal data (the true frame on a
        divisible sharded axis, the padded-even frame otherwise -- never
        gathering the indivisible sharded axis), runs the fused
        forward/contract/backward region, and returns the components on
        the same nodal spaces. The basis ``q`` and weights ``w`` are
        zero-padded on the half-axis mode axis to the padded extent
        before the call (the ``shard_map`` ``in_specs`` slice them per
        shard); the metric is replicated.

        Parameters
        ----------
        fields : Mapping[str, ScalarField]
            The component nodal fields (default layout, ``a`` sharded).
        q : jax.Array
            The M-orthonormal eigenvector planes, shape
            ``(*modes, D, D)`` in the engine's ``rfftn`` frame.
        weights : jax.Array
            The complex column weights, shape ``(*modes, D)`` (the mask
            0/1 for a projection, ``f(omega)`` for a spectral function).
        metric : jax.Array
            The diagonal energy metric ``M``, shape ``(D,)``.

        Returns
        -------
        dict[str, ScalarField]
            The contracted real component fields.
        """
        qf = self._pad_modes(q)
        wf = self._pad_modes(weights.astype(qf.dtype))
        metric = jnp.asarray(metric)
        if self.a_padded:
            decomposition = next(iter(fields.values())).grid.decomposition
            pieces = {
                name: decomposition.unpad_even(
                    f.storage, f.function_space)
                for name, f in fields.items()}
            out = self._region(pieces, qf, wf, metric)
            return {
                name: fields[name].with_storage(decomposition.pad_even(
                    out[name], fields[name].function_space))
                for name in self._components}
        pieces = {name: jnp.asarray(fields[name].data)
                  for name in self._components}
        out = self._region(pieces, qf, wf, metric)
        return {name: fields[name].with_data(out[name])
                for name in self._components}


# ================================================================
#  Resolution (from the grid layout; memoized per grid)
# ================================================================
#: per-grid memo of resolved plans, keyed on the static contraction
#: signature (the ``distributed_solve`` ``WeakKeyDictionary`` idiom: a
#: dropped grid auto-evicts its memo)
_PLANS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _axis_cells(grid: object, name: str) -> int:
    """Origin cell count of the mesh factor carrying ``name``."""
    return next(m for m in grid.factors if name in m.names).n_cells


def build_distributed_contraction(
    grid: object,
    *,
    bounded_axis: str,
    periodic_axis: str,
    components: tuple[str, ...],
    slices: Mapping[str, slice],
) -> ContractPlan | None:
    r"""
    Build the fused contraction plan from the grid layout, or None.

    Description
    -----------
    Derives the slab geometry -- sharded axis ``a``, half axis
    ``b = periodic_axis`` -- from the decomposition's default layout and
    the engine's fixed half axis. Returns None (the caller keeps the
    existing path / taught error) when the operand is single-device, the
    mesh is not 1-D, the layout shards nothing or only the bounded axis,
    the channel does not have exactly two periodic axes (the 2-D channel,
    or a >3-D one), or the layout shards the half axis itself (the
    ``rfft`` would need real data on the sharded axis, unreachable under
    one reshard).

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition and mesh factors.
    bounded_axis : str
        The bounded (walled) coordinate name.
    periodic_axis : str
        The engine's half-spectrum (``rfft``) periodic coordinate name.
    components : tuple[str, ...]
        The component names, the stacked-column segment order.
    slices : Mapping[str, slice]
        Per-component segment slices into the stacked column axis.

    Returns
    -------
    ContractPlan | None
        The reusable contraction pipeline, or None when ineligible.
    """
    decomposition = grid.decomposition
    if getattr(decomposition, "device_count", 1) <= 1:
        return None
    mesh = decomposition.device_mesh
    if len(mesh.axis_names) != 1:
        return None
    device_axes = decomposition.default_layout.device_axes
    names = grid.names
    periodic = tuple(n for n in names if n != bounded_axis)
    if len(device_axes) != 1 or len(periodic) != _PERIODIC_AXES:
        return None
    (a_name, axis_name) = device_axes[0]
    # decline when the sharded coordinate is the bounded axis (the
    # existing GSPMD path already handles that), a non-periodic axis, or
    # the engine's fixed half (rfft) axis -- the rfft would need real
    # data on the sharded axis, unreachable under one reshard.
    if a_name not in periodic or a_name == periodic_axis:
        return None
    shards = int(mesh.shape[axis_name])
    half_n = _axis_cells(grid, periodic_axis)
    n_b = half_n // 2 + 1
    n_a = _axis_cells(grid, a_name)
    return ContractPlan(
        mesh, axis_name,
        a_arr=names.index(a_name), b_arr=names.index(periodic_axis),
        bounded_arr=names.index(bounded_axis),
        b_plane=periodic.index(periodic_axis), half_n=half_n,
        n_a=n_a, pad_a=_ceil_mult(n_a, shards),
        n_b=n_b, pad_b=_ceil_mult(n_b, shards),
        slices=dict(slices), components=components)


def resolve_distributed_contraction(
    grid: object,
    *,
    bounded_axis: str,
    periodic_axis: str,
    components: tuple[str, ...],
    slices: Mapping[str, slice],
) -> ContractPlan | None:
    """
    Resolve (and memoize) the distributed contraction plan, or None.

    Description
    -----------
    Memoized per grid on the static contraction signature (axis names,
    component segment layout) only -- never on ``q`` / weights, which are
    dynamic apply-time arguments (a per-call closure rebuild would force
    recompiles). See :func:`build_distributed_contraction` for the
    decline conditions.

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition and mesh factors.
    bounded_axis : str
        The bounded (walled) coordinate name.
    periodic_axis : str
        The engine's half-spectrum (``rfft``) periodic coordinate name.
    components : tuple[str, ...]
        The component names, the stacked-column segment order.
    slices : Mapping[str, slice]
        Per-component segment slices into the stacked column axis.

    Returns
    -------
    ContractPlan | None
        The memoized plan, or None when ineligible.
    """
    key = (bounded_axis, periodic_axis, tuple(components),
           tuple((name, slices[name].start, slices[name].stop)
                 for name in components))
    memo = _PLANS.setdefault(grid, {})
    if key not in memo:
        memo[key] = build_distributed_contraction(
            grid, bounded_axis=bounded_axis, periodic_axis=periodic_axis,
            components=components, slices=slices)
    return memo[key]
