# Repair only missing periodic halo layers

`HaloSpec` validity counts consecutive layers outward from a field's true
interior. A partially valid halo does not require a complete replacement.
For an equal-cell periodic interval decomposition, preserve those inner
layers and exchange only the missing outer band on each side. A fully valid
side needs no message. The repaired result retains the existing full-width
halo claim; the stencil and time integration are unchanged.

For width w, c true cells and vL/vR valid layers, the left receive occupies
[0,w-vL) and comes from the previous shard's [c,c+w-vL). The right receive
occupies [w+c+vR,w+c+w) and comes from the next shard's [w+vR,2w).
This construction uses only true donor cells and therefore preserves the
existing reverse-mode differentiation contract of ppermute and slice updates.

The optimization applies only to uniform periodic IntervalMesh factors with
a true count divisible by the device count. Uneven blocks, bounded axes,
mapped factors, absent validity claims and all-invalid claims keep the full
exchange. Single-device behavior remains unchanged.

Tests corrupt every missing-band pair independently on both layout axes and
compare complete storage exactly against an unconditional sync. Additional
cases cover fallback geometries, reverse-mode finite differences and a short
hydrostatic WENO model gradient. The existing partially-valid fixture was
corrected to actually initialize the inner layers it declared valid; its
exact full-storage comparison is retained. A false validity claim is not a
supported input contract.

The partial path additionally requires the full halo width to fit within one
shard's true interior, so every donor slice above contains only true cells.
A direct decomposition with a larger halo keeps the original exchange.

The direct TensorDecomposition test fixtures explicitly select every exposed
device: its default device_ids=None chooses only device zero, even when JAX
exposes four devices. This strengthens the prior local-stencil and valid-axis
fixtures to exercise their intended distributed paths.
