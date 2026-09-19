# Local periodic stencils over distributed halo storage

## Problem and lowering

TensorDecomposition stores a concatenation of complete halo blocks. The
staggered stencil tails previously ran shape-changing windows over that
global storage array. GSPMD consequently inserted communication at the
boundary between blocks, even though every valid stencil output already
has its inputs in its block's halos.

The shared `_run_storage_kernel` now executes the same window kernel and
alignment inside `shard_map` for uniform periodic interval factors. Other
factors retain the established global lowering. Co-operands are explicitly
passed and partitioned on matching storage dimensions; size-one dimensions
remain replicated. Unsupported co-operand shapes retain the old path.

## Correctness boundary

A valid output slot k reads only slots k-m0 through k-m0+size-1. The input
halo validity and the existing consumed-validity calculation guarantee
these slots belong to its local block. Local and global windows therefore
read the same values in the same arithmetic order for every true DOF and
every claimed valid ghost. Outer ghost slots outside that claim can differ:
local lowering zero-fills them, while the old global spelling could read
across the boundary between two already padded blocks. They must be repaired
before consumption, as required by the existing halo contract.

Random periodic stencil oracles cover both layout axes, uneven cell counts,
full co-operands and transverse broadcast profiles. Compiled stencil checks
assert no collective remains when the input halos are already supplied.
An end-to-end WENO hydrostatic gradient test checks the resulting run against
a central finite difference.

## Avoiding redundant halo repair

`Decomposition.sync` accepts an optional `valid` halo claim. `Grid.sync`
passes the field's existing claim. On multiple devices the tensor backend
omits an axis only when both sides cover its full negotiated width. An
absent or partial claim retains the full fill. Single-device lowering stays
unchanged. Tensor-factor boundary extensions commute: repairing a second
axis therefore preserves an already valid first-axis extension, including
corner values. Tests compare complete storage after partial repair against
an unconditional fill and verify that the local-only repair communicates
nothing. Callers that do not supply a claim retain unconditional behavior.

## Barotropic blocking

The maximum temporal block is raised from 8 to 15 substeps, still restricted
to a divisor of the substep count and to half the local horizontal extent.
For a 30-substep solve with sufficient local cells this changes five blocks
to two. Halo reach remains twice the block length, so each block's true
interior is independent of artificial outer values. No substep, filter
weight, or physical timestep changes. Random-field tests include a larger
local extent that exercises the new 15-substep path.
