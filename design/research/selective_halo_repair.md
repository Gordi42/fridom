# Selective consumption-side halo repair

A separable stencil requires ghosts only along its applied axis. Previously,
a deficient local z claim triggered full Grid.sync, which also completed any
partially valid distributed x halo. The latter can communicate large arrays
without helping the vertical consumer.

Grid.sync keeps its full-fill default. Its optional axes tuple restricts both
the tensor fills/exchanges and the returned validity updates to those axes.
Unselected validity is preserved, never promoted. Empty selection is a no-op;
unknown names are rejected. Explicit Sync and scan materialization callers
keep the default behavior.

Grid.sync_required is an optional consumption-side protocol. The operator
base uses it when present and retains the original Sync fallback for duck
fields/grids. A completely invalid operand still receives a full fill shared
by all consumers. Single-device execution also retains the previous lowering.
Other operands repair only required deficient axes.

The external weak identity cache stores the repaired view. A later consumer
starts from that cached view, preserving earlier repairs, and replaces the
cached value with its enlarged valid region. Neither original field storage
nor its pytree auxiliary metadata is mutated. The existing concrete-to-tracer
cache guard remains in force.

Tensor-factor boundary extensions commute. Filling z over a partially valid
x slab extends exactly its currently valid x layers into z; the invalid x
outer layers remain unclaimed. A subsequent x exchange carries the repaired
z layers into the missing x region and completes correct corners. Operations
remain ordinary JAX gathers/writes/permutations and retain JVP/VJP support.

Regression tests poison unclaimed ghosts with NaNs, assert zero exchanges for
local repair followed by exactly one x-axis repair, compare the full final
storage with an independent full sync, and verify original treedef stability,
cache accumulation, concrete-operand tracer safety, and gradients. Existing
single-device and duck-field behavior stays covered by unchanged tests.
