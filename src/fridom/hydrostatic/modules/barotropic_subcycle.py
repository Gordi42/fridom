"""Temporally blocked barotropic stencil on uniform periodic C grids."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.finite_difference import FiniteDifference
from fridom.spatial.operators.staggering import uniform_spacing
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField

_DIMENSIONS = 3
_MIN_BLOCK_CELLS = 2


def supports_blocking(
    pressure: ScalarField, u: ScalarField, v: ScalarField,
    horizontal: tuple[str, str],
) -> bool:
    """Whether the fields carry the exact periodic two-point C stencil.

    Description
    -----------
    Nonuniform meshes, alternate stencils, charts, immersed geometry and
    unsupported layouts retain the field-operator implementation. The
    optimized kernel consumes true arrays, with its own temporary halos.
    """
    grid = pressure.grid
    if (grid.override_keys or grid.chart_coords is not None
            or grid.immersed is not None or grid.mapping is not None):
        return False
    factors = pressure.function_space.bare.factors
    if (tuple(grid.names[:2]) != horizontal
            or len(grid.names) != _DIMENSIONS
            or not isinstance(factors[2], ConstantSpace)):
        return False
    expected = ((NodeSet.CENTER, NodeSet.CENTER),
                (NodeSet.RIGHT, NodeSet.CENTER),
                (NodeSet.CENTER, NodeSet.RIGHT))
    for field, nodes in zip((pressure, u, v), expected, strict=True):
        if field.function_space.layout != pressure.function_space.layout:
            return False
        for axis, node in zip(horizontal, nodes, strict=True):
            factor = field.function_space.bare.factor(axis)
            if (not isinstance(factor, NodalSpace)
                    or factor.node_set is not node
                    or not isinstance(factor.mesh, IntervalMesh)
                    or not factor.mesh.periodic
                    or grid.dispatch.resolve("diff", factor)
                    is not FiniteDifference()):
                return False
    axes = dict(pressure.function_space.layout.device_axes)
    if axes and tuple(axes) != horizontal[:1]:
        return False
    shards = grid.decomposition.device_count if axes else 1
    return (pressure.shape[0] % shards == 0
            and pressure.shape[0] // shards >= _MIN_BLOCK_CELLS)


def periodic_subcycle(
    fields: tuple[ScalarField, ...], dtau: jax.Array, csqr: jax.Array,
    weights: tuple[float, ...], horizontal: tuple[str, str],
) -> tuple[ScalarField, ScalarField, ScalarField]:
    """Advance and average the unchanged forward-backward substeps.

    Description
    -----------
    ``fields`` holds pressure, the two velocities and the two constant
    slow forcings. Across devices each block exchanges all five fields
    in two packed messages. A halo twice the block length covers every
    dependency of the sequential divergence/gradient stencil. Redundant
    work in that halo is discarded; only valid interiors are committed.
    The block divides the substep count, so no extra physical steps run.
    One device uses the same stencil without halo exchanges or blocking.
    """
    pressure, u, v, *_ = fields
    decomposition = pressure.grid.decomposition
    axis_name = dict(pressure.function_space.layout.device_axes).get(
        horizontal[0])
    shards = decomposition.device_count if axis_name else 1
    local_size = pressure.shape[0] // shards
    block = max(k for k in range(1, min(15, local_size // 2, len(weights)) + 1)
                if len(weights) % k == 0)
    width = 2 * block
    dx, dy = (uniform_spacing(pressure.function_space.bare.factor(axis))
              for axis in horizontal)
    values = jnp.stack([field.data for field in fields])
    filtered = jnp.asarray(weights, dtype=values.dtype)
    spec = PartitionSpec(None, axis_name, None, None)

    def step(p: jax.Array, a: jax.Array, b: jax.Array,
             forcing: jax.Array) -> tuple[jax.Array, ...]:
        # Match FiniteDifference's statically scaled stencil weights,
        # including its floating-point operation order.
        divergence = ((-1.0 / dx) * jnp.roll(a, 1, axis=0) + (1.0 / dx) * a
                      + ((-1.0 / dy) * jnp.roll(b, 1, axis=1)
                         + (1.0 / dy) * b))
        p = p - dtau * csqr * divergence
        grad_x = (-1.0 / dx) * p + (1.0 / dx) * jnp.roll(p, -1, axis=0)
        grad_y = (-1.0 / dy) * p + (1.0 / dy) * jnp.roll(p, -1, axis=1)
        a = a - dtau * grad_x + dtau * forcing[0]
        b = b - dtau * grad_y + dtau * forcing[1]
        return p, a, b

    def local_run(local: jax.Array) -> jax.Array:
        initial, forcing = local[:3], local[3:]
        if shards == 1:
            def one(carry: tuple, weight: jax.Array) -> tuple[tuple, None]:
                p, a, b, ap, aa, ab = carry
                p, a, b = step(p, a, b, forcing)
                return (p, a, b, ap + weight * p,
                        aa + weight * a, ab + weight * b), None

            p, a, b = initial
            result, _ = jax.lax.scan(
                one, (p, a, b, jnp.zeros_like(p),
                      jnp.zeros_like(a), jnp.zeros_like(b)), filtered)
            return jnp.stack(result[3:])

        def outer(
            carry: tuple, block_weights: jax.Array,
        ) -> tuple[tuple, None]:
            current, average = carry
            packed = jnp.concatenate((current, forcing), axis=0)
            left = jax.lax.ppermute(
                packed[:, -width:], axis_name,
                [(i, (i + 1) % shards) for i in range(shards)])
            right = jax.lax.ppermute(
                packed[:, :width], axis_name,
                [(i, (i - 1) % shards) for i in range(shards)])
            extended = jnp.concatenate((left, packed, right), axis=1)

            def inner(carry: tuple, weight: jax.Array) -> tuple[tuple, None]:
                current, total = carry
                current = jnp.stack(step(*current, extended[3:]))
                return (current, total + weight * current), None

            (end, total), _ = jax.lax.scan(
                inner, (extended[:3], jnp.zeros_like(extended[:3])),
                block_weights)
            return (end[:, width:-width],
                    average + total[:, width:-width]), None

        (_, average), _ = jax.lax.scan(
            outer, (initial, jnp.zeros_like(initial)),
            filtered.reshape(-1, block))
        return average

    result = jax.shard_map(
        local_run, mesh=decomposition.device_mesh,
        in_specs=spec, out_specs=spec, check_vma=False)(values)
    return (pressure.with_data(result[0]), u.with_data(result[1]),
            v.with_data(result[2]))
