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

The synthesis-only features (single-mode ``mode()`` states,
random-phase channel states) build their coefficient columns host-side
and need only that inverse path: :meth:`ContractPlan.synthesize` runs
it alone (no forward, no contraction), taking store-frame coefficient
columns to real fields on the grid's own layout.

The coefficient frame is fixed by the engine: ``q`` is laid out in the
``rfftn`` frame (the full spectrum on the sharded axis ``a``, the half
spectrum on ``periodic_axis``, both in their original array positions),
so this lowering keeps that exact frame -- the half axis is never
relocated, and the transpose partner ``b`` (the half axis) is padded on
its **coefficient** extent (``n // 2 + 1``), with empty trailing pad
shards allowed (the pad lanes are transient, never stored; zero-padded
``q`` and ``w`` make their output exactly zero).

This module serves two channel geometries. The 3-D channel
(:class:`ContractPlan`) -- exactly two periodic axes (the sharded axis
``a`` and the half axis ``b = periodic_axis``) plus one bounded axis,
whose position (walled x / y / z) is free. The **2-D channel**
(:class:`Channel2DPlan`) -- one periodic axis plus one bounded axis,
the default multi-device layout of every 2-D channel: with no second
periodic axis to absorb the shardedness, the transpose parks it on the
**bounded** axis (the owner's transpose pipeline, two ``all_to_all``
moves), so the local ``rfft`` runs and the per-``kx`` contraction stays
local. The resolution (:func:`resolve_distributed_contraction`) is
memoized per grid on static keys only (axis names, component segment
layout); ``q``, ``w`` and the metric enter as **arguments** at apply
time (dynamic; the ``shard_map`` ``in_specs`` slice the replicated basis
per shard, a per-device memory win). The plan declines (returns None --
the caller keeps the existing GSPMD path or the taught error) on a
single device, a non-1-D mesh, a layout that shards nothing / only the
bounded axis, a channel with more than two periodic axes (a
hypothetical >3-D one), or -- for the 3-D path -- a layout that shards
the half (``rfft``) axis itself (defensive; the engine designates a
**local** half axis at build time, so a 3-D channel never builds a
basis in that frame).
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.spatial.operators.distributed_transform import (
    TransposeGeometry,
    transpose_backward,
    transpose_forward,
)
from fridom.spatial.operators.transform import axis_slice

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.spatial.fields.scalar_field import ScalarField

#: the 3-D channel: the sharded axis ``a`` plus the half axis ``b``.
#: A single periodic axis is the 2-D channel, served by the transpose
#: pipeline of :class:`Channel2DPlan`; a hypothetical >3-D channel is
#: out of scope (keeps the taught error).
_PERIODIC_AXES = 2
_ONE_PERIODIC_AXIS = 1


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

        # the backward-only synthesis region (host-built coefficient
        # columns -> real fields, no forward / contraction): reuses the
        # contraction's inverse pipeline (``_backward_component``). The
        # input shards the half axis ``b`` -- the internal frame the
        # forward's ``all_to_all`` produces, so a coefficient column laid
        # out with ``b`` sharded and ``a`` full needs no transpose here;
        # the output shards ``a``, the field storage frame.
        coeff_spec = [None, None, None]
        coeff_spec[b_arr] = axis_name
        self._coeff_spec = jax.sharding.PartitionSpec(*coeff_spec)
        self._backward_region = jax.jit(jax.shard_map(
            self._backward_body, mesh=mesh,
            in_specs=(dict.fromkeys(components, self._coeff_spec),),
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

    def _backward_body(
        self,
        coeffs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Per-component inverse pipeline (one shard): synthesis only.

        The synthesis counterpart of :meth:`_body`: the coefficient
        columns are already built (host-side, in the engine's store
        frame -- the half axis ``b`` sharded), so only the mirrored
        inverse path runs, no forward transform and no contraction.
        """
        return {name: self._backward_component(coeffs[name])
                for name in self._components}

    # ================================================================
    #  Application
    # ================================================================
    def _pad_modes(self, arr: jax.Array) -> jax.Array:
        """Zero-pad the plane-frame half-axis mode axis to ``pad_b``."""
        if self._pad_b == self._n_b:
            return arr
        return _tail_pad(arr, self._b_plane, self._pad_b - self._n_b)

    def _pad_half_axis(self, arr: jax.Array) -> jax.Array:
        """Zero-pad the field-frame half axis (``b``) to ``pad_b``."""
        if self._pad_b == self._n_b:
            return arr
        return _tail_pad(arr, self._b, self._pad_b - self._n_b)

    def _wrap(
        self,
        out: dict[str, jax.Array],
        fields: Mapping[str, ScalarField],
    ) -> dict[str, ScalarField]:
        """Wrap region outputs as fields on the components' nodal spaces.

        On an indivisible sharded axis the region delivers the
        padded-even storage frame (``pad_even``); otherwise the true
        frame (``with_data``). Shared by :meth:`apply` and
        :meth:`synthesize`.
        """
        if self.a_padded:
            decomposition = next(iter(fields.values())).grid.decomposition
            return {
                name: fields[name].with_storage(decomposition.pad_even(
                    out[name], fields[name].function_space))
                for name in self._components}
        return {name: fields[name].with_data(out[name])
                for name in self._components}

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
        else:
            pieces = {name: jnp.asarray(fields[name].data)
                      for name in self._components}
        out = self._region(pieces, qf, wf, metric)
        return self._wrap(out, fields)

    def synthesize(
        self,
        coeffs: Mapping[str, jax.Array],
        fields: Mapping[str, ScalarField],
    ) -> dict[str, ScalarField]:
        r"""
        Inverse-transform coefficient columns to real fields (no gather).

        Description
        -----------
        The synthesis (backward-only) entry, the inverse of the
        contraction :meth:`apply` shares with the engine's host path:
        the caller has already built each component's coefficient column
        in the engine's store frame (the sharded axis ``a`` full
        spectrum, the half axis ``b`` the Hermitian half spectrum, the
        bounded axis nodal), so only the fused inverse pipeline runs.
        The half axis is zero-padded to the balanced extent and the
        ``shard_map`` ``in_specs`` shard it -- the internal frame the
        forward's ``all_to_all`` produces, so the store-frame column
        needs no transpose. The output lands on the components' own
        nodal spaces (the grid's default layout; the sharded axis is
        never gathered).

        Parameters
        ----------
        coeffs : Mapping[str, jax.Array]
            Per-component coefficient columns in the store frame
            (``a`` full spectrum, ``b`` the ``n_b`` half spectrum,
            bounded axis nodal), complex.
        fields : Mapping[str, ScalarField]
            Template physical nodal fields (the components' spaces),
            for the output layout / wrapping.

        Returns
        -------
        dict[str, ScalarField]
            The synthesized real component fields.
        """
        padded = {name: self._pad_half_axis(jnp.asarray(coeffs[name]))
                  for name in self._components}
        out = self._backward_region(padded)
        return self._wrap(out, fields)


# ================================================================
#  The 2-D channel plan (transpose partner = the bounded axis)
# ================================================================
class Channel2DPlan:

    r"""
    One 2-D channel's transpose-pipeline eigenbasis contraction.

    Description
    -----------
    The sibling of :class:`ContractPlan` for the **2-D channel** (one
    periodic axis ``a`` plus one bounded axis) -- the default layout of
    every multi-device 2-D channel, which shards its single periodic
    axis. With no second periodic axis to absorb the shardedness, the
    ``rfft`` cannot run locally under one reshard; instead the fused
    region transposes **through the bounded axis** (the owner's
    transpose pipeline, the phase-3 plan
    ``design/plans/active/gspmd_transform_illegality_plan.md``): per
    component,

    1. ``all_to_all`` parks the shardedness on the bounded axis (the
       periodic axis ``a`` becomes local);
    2. the local Hermitian ``rfft`` on ``a`` produces the half spectrum
       ``kx``;
    3. ``all_to_all`` re-shards ``kx`` and localizes the bounded axis,

    so every device holds full bounded columns for its own ``kx`` rows
    and the per-``kx`` dense contraction ``Q diag(w) Q^H M`` runs purely
    locally (the basis ``q`` sliced per ``kx`` shard by the ``shard_map``
    ``in_specs``, no reduction over a sharded axis). Only ``all_to_all``
    collectives -- never an ``all_gather``. The transpose engine is the
    shared ``transpose_forward`` / ``transpose_backward`` of
    ``operators.distributed_transform`` (``a`` the Hermitian half axis,
    the bounded axis the transpose partner); because the components'
    bounded extents differ (``u`` on ``n_y`` faces, ``v`` on
    ``n_y - 1``), each carries its own transpose geometry, differing
    only in the partner extent.

    Build through :func:`resolve_distributed_contraction`, not this
    plumbing constructor.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        The decomposition's device mesh (1-D).
    axis_name : str
        The device-mesh axis name.
    a_arr : int
        The periodic (sharded) coordinate's array axis -- also the
        ``rfft`` half axis.
    bounded_arr : int
        The bounded coordinate's array axis (the transpose partner).
    half_n : int
        The periodic axis's true nodal extent (the ``irfft`` length).
    n_kx : int
        The periodic axis's half-spectrum extent (``half_n // 2 + 1``).
    pad_a : int
        The periodic axis's nodal padded-even extent.
    pad_kx : int
        The half-spectrum axis's padded split extent.
    bounded_n : Mapping[str, int]
        Per-component bounded-axis extent.
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
        bounded_arr: int,
        half_n: int,
        n_kx: int,
        pad_a: int,
        pad_kx: int,
        bounded_n: Mapping[str, int],
        slices: Mapping[str, slice],
        components: tuple[str, ...],
    ) -> None:
        """Store the geometry and build the shard_map regions."""
        self._mesh: jax.sharding.Mesh = mesh
        self._axis_name: str = axis_name
        self._a: int = a_arr
        self._bounded: int = bounded_arr
        self._half_n: int = half_n
        self._n_kx: int = n_kx
        self._pad_a: int = pad_a
        self._pad_kx: int = pad_kx
        self._slices: Mapping[str, slice] = slices
        self._components: tuple[str, ...] = components
        shards = int(mesh.shape[axis_name])
        # per-component transpose geometry (a = the periodic half axis,
        # the bounded axis the transpose partner -- extent per component)
        self._geoms: dict[str, TransposeGeometry] = {
            name: TransposeGeometry(
                axis_name=axis_name, a=a_arr, b=bounded_arr,
                a_half=True, a_n=half_n, a_spec_n=n_kx,
                b_n=bounded_n[name], pad_a=pad_a, pad_a_spec=pad_kx,
                pad_b=_ceil_mult(bounded_n[name], shards),
                local_stages=(), real=True)
            for name in components}

        ndim = 2
        nodal = [None, None]
        nodal[a_arr] = axis_name
        self._nodal_spec = jax.sharding.PartitionSpec(*nodal[:ndim])
        # q planes (kx, D, D) and weights (kx, D) shard the kx plane axis
        self._q_spec = jax.sharding.PartitionSpec(axis_name, None, None)
        self._w_spec = jax.sharding.PartitionSpec(axis_name, None)
        self._metric_spec = jax.sharding.PartitionSpec()

        comp_specs = dict.fromkeys(components, self._nodal_spec)
        self._region = jax.jit(jax.shard_map(
            self._body, mesh=mesh,
            in_specs=(comp_specs, self._q_spec, self._w_spec,
                      self._metric_spec),
            out_specs=comp_specs))
        # backward-only synthesis: the store-frame coefficient columns
        # (kx full-half spectrum sharded, bounded nodal) are the internal
        # frame the forward's transpose produces, so no extra reshard
        coeff_spec = [None, None]
        coeff_spec[a_arr] = axis_name
        self._coeff_spec = jax.sharding.PartitionSpec(*coeff_spec[:ndim])
        self._backward_region = jax.jit(jax.shard_map(
            self._backward_body, mesh=mesh,
            in_specs=(dict.fromkeys(components, self._coeff_spec),),
            out_specs=comp_specs))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def a_padded(self) -> bool:
        """Whether the periodic axis needs the padded-even field frame."""
        return self._pad_a != self._half_n

    # ================================================================
    #  The per-shard region (runs under jax.shard_map)
    # ================================================================
    def _body(
        self,
        comps: dict[str, jax.Array],
        q: jax.Array,
        w: jax.Array,
        metric: jax.Array,
    ) -> dict[str, jax.Array]:
        """Forward transpose, per-kx contraction, backward (one shard)."""
        trans = {name: transpose_forward(comps[name], self._geoms[name])
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
            result[name] = transpose_backward(seg, self._geoms[name])
        return result

    def _backward_body(
        self,
        coeffs: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Per-component inverse transpose (one shard): synthesis only.

        The synthesis counterpart of :meth:`_body`: the coefficient
        columns are already built (host-side, in the store frame -- the
        periodic half axis ``kx`` sharded, the bounded axis nodal), so
        only the mirrored inverse transpose runs, no forward and no
        contraction.
        """
        return {name: transpose_backward(coeffs[name], self._geoms[name])
                for name in self._components}

    # ================================================================
    #  Application
    # ================================================================
    def _pad_modes(self, arr: jax.Array) -> jax.Array:
        """Zero-pad the plane-frame kx mode axis (axis 0) to ``pad_kx``."""
        if self._pad_kx == self._n_kx:
            return arr
        return _tail_pad(arr, 0, self._pad_kx - self._n_kx)

    def _pad_half_axis(self, arr: jax.Array) -> jax.Array:
        """Zero-pad the field-frame kx axis (``a``) to ``pad_kx``."""
        if self._pad_kx == self._n_kx:
            return arr
        return _tail_pad(arr, self._a, self._pad_kx - self._n_kx)

    def _wrap(
        self,
        out: dict[str, jax.Array],
        fields: Mapping[str, ScalarField],
    ) -> dict[str, ScalarField]:
        """Wrap region outputs as fields on the components' nodal spaces.

        On an indivisible periodic axis the region delivers the
        padded-even storage frame (``pad_even``); otherwise the true
        frame (``with_data``). Shared by :meth:`apply` and
        :meth:`synthesize`.
        """
        if self.a_padded:
            decomposition = next(iter(fields.values())).grid.decomposition
            return {
                name: fields[name].with_storage(decomposition.pad_even(
                    out[name], fields[name].function_space))
                for name in self._components}
        return {name: fields[name].with_data(out[name])
                for name in self._components}

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
        The 2-D-channel analogue of :meth:`ContractPlan.apply`: extracts
        each component's nodal data (the true frame on a divisible
        periodic axis, the padded-even frame otherwise), runs the fused
        transpose/contract/transpose region, and returns the components
        on the same nodal spaces. The basis ``q`` and weights ``w`` are
        zero-padded on the ``kx`` plane axis to the padded extent before
        the call (the ``shard_map`` ``in_specs`` slice them per shard);
        the metric is replicated.

        Parameters
        ----------
        fields : Mapping[str, ScalarField]
            The component nodal fields (default layout, periodic axis
            sharded).
        q : jax.Array
            The M-orthonormal eigenvector planes, shape ``(n_kx, D, D)``
            in the ``rfft`` frame.
        weights : jax.Array
            The complex column weights, shape ``(n_kx, D)``.
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
        else:
            pieces = {name: jnp.asarray(fields[name].data)
                      for name in self._components}
        out = self._region(pieces, qf, wf, metric)
        return self._wrap(out, fields)

    def synthesize(
        self,
        coeffs: Mapping[str, jax.Array],
        fields: Mapping[str, ScalarField],
    ) -> dict[str, ScalarField]:
        r"""
        Inverse-transform coefficient columns to real fields (no gather).

        Description
        -----------
        The 2-D-channel analogue of :meth:`ContractPlan.synthesize`: the
        caller has already built each component's coefficient column in
        the store frame (the periodic half axis ``kx`` on its Hermitian
        half spectrum, the bounded axis nodal), which is exactly the
        internal frame the forward transpose produces (``kx`` sharded,
        bounded local), so only the fused inverse transpose runs. The
        ``kx`` axis is zero-padded to the balanced extent and the
        ``shard_map`` ``in_specs`` shard it. The output lands on the
        components' own nodal spaces (the grid's default layout; the
        periodic axis is never gathered).

        Parameters
        ----------
        coeffs : Mapping[str, jax.Array]
            Per-component coefficient columns in the store frame (``kx``
            the ``n_kx`` half spectrum, bounded axis nodal), complex.
        fields : Mapping[str, ScalarField]
            Template physical nodal fields (the components' spaces),
            for the output layout / wrapping.

        Returns
        -------
        dict[str, ScalarField]
            The synthesized real component fields.
        """
        padded = {name: self._pad_half_axis(jnp.asarray(coeffs[name]))
                  for name in self._components}
        out = self._backward_region(padded)
        return self._wrap(out, fields)


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
) -> ContractPlan | Channel2DPlan | None:
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
    ``rfft`` would need real data on the sharded axis; defensive --
    ``channel_eigenpairs`` designates a local half axis at build time,
    so a 3-D channel does not reach this decline).

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
    ContractPlan | Channel2DPlan | None
        The reusable contraction pipeline (the 3-D ``ContractPlan`` or
        the 2-D ``Channel2DPlan``), or None when ineligible.
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
    if len(device_axes) != 1:
        return None
    (a_name, axis_name) = device_axes[0]
    if len(periodic) == _ONE_PERIODIC_AXIS:
        return _build_channel_2d(
            grid, mesh, axis_name, a_name,
            bounded_axis=bounded_axis, periodic_axis=periodic_axis,
            components=components, slices=slices)
    # decline (3-D path) when the channel is not exactly two periodic
    # axes, the sharded coordinate is the bounded axis (the existing
    # GSPMD path already handles that) or a non-periodic axis, or it is
    # the designated half (rfft) axis -- the rfft would need real data
    # on the sharded axis (defensive: the engine designates a local
    # half axis at build time, so a 3-D channel never presents this).
    if (len(periodic) != _PERIODIC_AXES or a_name not in periodic
            or a_name == periodic_axis):
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


def _build_channel_2d(
    grid: object,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    a_name: str,
    *,
    bounded_axis: str,
    periodic_axis: str,
    components: tuple[str, ...],
    slices: Mapping[str, slice],
) -> Channel2DPlan | None:
    r"""
    Build the 2-D channel transpose plan, or None.

    Description
    -----------
    Serves the 2-D channel (one periodic axis, one bounded axis) on a
    1-D device mesh whose default layout shards the periodic axis (the
    single periodic axis is always the engine's half/``rfft`` axis).
    Returns None when the layout shards the bounded axis instead (the
    periodic ``rfft`` axis is then local -- the plain GSPMD path serves
    it), or when the periodic axis's transient bounded-partner shard
    would empty a trailing device (a bounded extent too short to split).

    Parameters
    ----------
    grid : object
        The grid carrying the mesh factors.
    mesh : jax.sharding.Mesh
        The 1-D device mesh.
    axis_name : str
        The device-mesh axis name.
    a_name : str
        The default layout's sharded coordinate name.
    bounded_axis : str
        The bounded coordinate name (the transpose partner).
    periodic_axis : str
        The single periodic coordinate name (the ``rfft`` half axis).
    components : tuple[str, ...]
        The component names, the stacked-column segment order.
    slices : Mapping[str, slice]
        Per-component segment slices (their lengths are the components'
        bounded extents).

    Returns
    -------
    Channel2DPlan | None
        The reusable 2-D channel plan, or None when ineligible.
    """
    if a_name != periodic_axis:
        return None
    names = grid.names
    shards = int(mesh.shape[axis_name])
    half_n = _axis_cells(grid, periodic_axis)
    n_kx = half_n // 2 + 1
    bounded_n = {name: slices[name].stop - slices[name].start
                 for name in components}
    return Channel2DPlan(
        mesh, axis_name,
        a_arr=names.index(periodic_axis),
        bounded_arr=names.index(bounded_axis),
        half_n=half_n, n_kx=n_kx,
        pad_a=_ceil_mult(half_n, shards),
        pad_kx=_ceil_mult(n_kx, shards),
        bounded_n=bounded_n, slices=dict(slices),
        components=components)


def resolve_distributed_contraction(
    grid: object,
    *,
    bounded_axis: str,
    periodic_axis: str,
    components: tuple[str, ...],
    slices: Mapping[str, slice],
) -> ContractPlan | Channel2DPlan | None:
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
    ContractPlan | Channel2DPlan | None
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
