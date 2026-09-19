# Periodic hydrostatic throughput

Two exact-arithmetic rewrites reduce the cost of flat periodic hydrostatic
models. Geometry and operator gates retain the original implementation for
configurations where the equivalence is not established. No time step,
substep count, precision, advection order or filter weight changes.

## Order-matched surface correction

The biased momentum constancy correction formerly computed A(1) over the full
volume because the two-point surface trace did not match the order-coupled
velocity interpolation. On uniform periodic transverse axes the interpolation
commutes with the continuity difference. Interpolating the surface trace with
the same operator therefore reproduces the full correction, up to cancellation
roundoff. Lateral walls, mapped meshes, charts and immersed momentum keep the
full-volume fallback. Tests retain the old naive-trace counterexample and
compare the new result against the full-volume oracle, at both supported orders (3 and 5).

## Barotropic temporal blocking

The flat periodic nodal C-grid subcycle uses the same forward pressure step,
backward velocity step and SM2005 weights. Single-device model runs retain
the field-operator loop to preserve the existing bitwise identity with flat
terrain coordinates: raw-array lowering, though numerically equivalent,
changes compiler rounding. On an x-slab decomposition the temporary arrays carry
a halo of twice the block length. Each block packs pressure, both velocities
and both fixed slow forcings into two neighbor messages; all substeps inside
the block are local. A complete forward/backward substep reaches at most two
cells, so discarding the temporary halo leaves the exact interior dependence
cone. The block divides the number of substeps and its halo never exceeds a
neighbor's interior. No additional physical steps are taken.

The temporary halo belongs only to these 2-D arrays: it does not widen the
3-D model storage. The optimization is restricted to default second-order
nodal differences on periodic uniform horizontal axes, flat geometry, and a
replicated or x-slab layout on multiple devices. FV surface spaces, custom dispatch, other axis
orders and geometries retain the field-operator loop. Random fields test
shard boundaries, and a model-run gradient is checked against finite differences.

## Compiler setting

On JAX 0.10.2 / A100, disabling `multi_output_fusion` improves the unchanged
one-GPU WENO hydrostatic step from about 80.7 to 60.9 ms at 1024²×64.
The distributed harness already requires this flag for correctness. The code
changes are independent of that setting; no process-global XLA configuration
is changed by these modules. Performance comparisons must record the flag.
