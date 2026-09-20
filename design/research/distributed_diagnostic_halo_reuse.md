# Distributed diagnostic halo reuse

Three representation boundaries discarded reusable halo information or
exchanged values immediately before overwriting them.

1. Vector component metadata reattachment rebuilt a scalar field from the
   same grid, space and storage, but omitted the result's `halo_valid`.
   Reattaching names and units now preserves the result claim. It never
   adopts the incumbent component's claim: the incoming operation remains
   responsible for the claim's numerical validity.
2. A cumulative integral on an undistributed tensor axis previously unpadded
   every axis, integrated true columns, then padded every axis again. With
   no Jacobian weight, the integral commutes with extension on transverse
   tensor factors. The distributed path now integrates existing transverse
   ghost columns, preserving precisely their input validity. It slices out
   the integration axis's true interval and resets that axis's claim.
   Geometric weights and one-device execution retain the original path.
3. The scan boundary formerly sealed all state components, including
   diagnostics recomputed by the next step. Distributed diagnostics now
   carry zero claims at both scan entry and exit. This is conservative:
   ordinary consumers still repair halos when needed. Prognostic and
   auxiliary fields retain the materialized seal, and one-device execution
   retains its original policy. The field table's lifecycle determines the
   policy; annotation metadata never selects numerical behavior.

All three changes preserve true values and numerical methods. In particular,
no stencil radius, requested barotropic timestep, or validation tolerance is
reduced. Preserving transverse columns alone did not improve timing while
metadata reattachment still erased their validity; the combined path avoids
downstream exchanges. Resetting every carry field was measured and rejected.

Validation includes independent random column sums on nodal/FV and stretched
meshes, uneven transverse decompositions, partial-claim preservation,
component map/replace/arithmetic, multi-step hydrostatic gradients, and
distributed model comparisons against the unchanged one-device execution.
