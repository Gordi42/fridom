#!/usr/bin/env python
"""Aliasing demo: how the eigh heap corruption gets blamed on ``sort``.

Companion to ``repro_batched_eigh_heap_corruption.py``. This script
reproduces the *exact symptom* recorded in
``design/research/multidevice_test_faults.md`` item 1 (forced-CPU
multi-device, exit 139, faulthandler pointing into the argsort/``sort``
lowering) -- and shows it is an alias: the corruption is caused by the
batched ``eigh`` on the line above, and the immediately following
``argsort`` is simply the next op whose lowering walks the poisoned heap.

The crash SITE wanders run to run (all heap corruption): observed at
``mlir.make_ir_context``, at ``lax._sort_lower`` ->
``_canonicalize_float_for_sort`` -> ``shaped_abstractify``, and at
``_standard_weak_type_rule`` -- all inside the pjit lowering of THIS
argsort. Delete the ``argsort`` line and the ``eigh`` alone still aborts
(see the companion script); it is necessary and sufficient, the sort is
neither.

Run (crashes with exit 139 on a many-core host)
-----------------------------------------------
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python -X faulthandler repro_forced4_sort_alias.py
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((4,), ("devices",), axis_types=(AxisType.Auto,))
a = jax.random.normal(jax.random.PRNGKey(0), (16, 9, 63, 63),
                      dtype=jnp.float64)
a = jax.device_put((a + jnp.swapaxes(a, -1, -2)) / 2,
                   NamedSharding(mesh, P(None, None, None, None)))
w, v = jnp.linalg.eigh(a)          # ROOT CAUSE: corrupts the heap
order = jnp.argsort(w, axis=-1)    # BLAMED: crash surfaces in its lowering
order.block_until_ready()
print("OK")
