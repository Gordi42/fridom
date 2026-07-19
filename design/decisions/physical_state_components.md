---
status: decided (rulings a, d in flight; b implementing; c scheduled)
date: 2026-07-19
---

# State components are physical on every grid

**Status: owner-ratified design, 2026-07-19** (Silvano, in chat). The
user-facing state components `u`, `v`, `w` denote the **physical**
velocity components (m/s in the ambient frame) on every grid — flat,
stretched, terrain-following, spherical — *all the way down*: the
stored pytree the stepper integrates, tendency-dict keys, checkpoint
contents and io output all hold the physical quantities, so
`state.u == state["u"]` needs no derived-access magic and there is no
layer at which the same name means two things. Chart-native
quantities (the contravariant flux `J\omega` on a mapped column, the
coordinate velocities `d\lambda/dt` on an embedding chart) are
**derived, read-only** and live behind the `state.chart` namespace.

This is the state-vocabulary sibling of
`physical_integral_default.md`: the verbs and names mean physics; the
computational spellings are the explicitly-named expert surface.

## 1. The as-found convention census (2026-07-19)

The three model packages disagreed about what the state names mean on
chart/mapped grids — the source of the "sometimes there is a
`u_physical`, sometimes not" confusion:

| package (chart case)     | horizontal `u, v`            | vertical `w`                       |
| ------------------------ | ---------------------------- | ---------------------------------- |
| shallowwater2, spherical | chart-native (`dlon/dt`)     | —                                  |
| hydrostatic, terrain     | physical (m/s)               | diagnosed chart flux `J\omega`     |
| nonhydro2, mapped        | physical (m/s)               | prognostic **physical** `w`        |

The nonhydro2 evidence is the mapped pressure solve
(`src/fridom/nonhydro2/modules/mapped_pressure.py`): its divergence
RHS `sum_i D_i(J u_i) + D_b(w - sum_i Z_i I(u_i))` *derives* the
contravariant flux by subtracting the slope terms from the stored
`w`, which is only correct when the stored `w` is physical (verified:
that expression is exactly `J\,\nabla_{\rm phys}\cdot u` with
physical components).

**The decisive machinery fact:** the shared advection
(`src/fridom/model/modules/advection.py`, mapped-grids section)
builds its transport from **physical** velocity fluxes
(`F_i = v_i q`, physical divergence with slope corrections; the FV
mapped form `(1/J)[D_i(J F_i) + D_b(F_b - Z_i I(F_i))]` reduces the
vertical member to the contravariant flux only if the queried `w` is
physical). nonhydro2 satisfies that contract; hydrostatic's diagnosed
`J\omega` under the same `Velocity` role is the odd one out. The
invariant is therefore not just naming hygiene — it aligns
hydrostatic with the contract the shared machinery already assumes.

## 2. The rulings

**(a) The invariant.** Stored state components are the physical
quantities on every grid. Consequences: tendency dicts are tendencies
of physical components; checkpoints/restarts hold physical fields
verbatim (no conversion trap, no double-conversion hazard); io output
names agree with the user surface; `jax.grad` through
`Model.propagator` w.r.t. an initial field is a gradient w.r.t. the
*physical* IC.

**(b) Hydrostatic `w` storage.** The core's continuity diagnosis
keeps building the contravariant volume flux `J\omega` (bottom-up
face-form cumint — the exact-telescoping, exact-bottom-BC working
quantity) but stores it as an **internal component**; the public
state `w` becomes the physical
`w = J\omega + u Z_x + v Z_y` (slope terms interpolated to the `w`
faces; byte-identical on flat columns where `Z = 0`). Flux consumers
(advection's vertical trio member per section 1 wants physical `w`;
anything genuinely wanting the flux reads the internal component)
are repointed per an explicit consumer census. The
`ConstantStratification` terrain slope spelling of `d629a489`
migrates into the core diagnosis and the module simplifies back to
`-N^2\,w.to(b)` — same physics class; the O(h^2) energy-gate
collapse (`design/research/energy_metric_asymmetry.md` §4) remains
the acceptance.

**(c) shallowwater2 flip: its own campaign.** Moving the spherical
prognostics from `dlon/dt` to physical m/s components (the
NEMO/MITgcm curvilinear standard; conversions are pointwise diagonal
metric rescales) reverses a deliberate recorded design and touches
the chart operator plumbing, the Sadourny energy-conserving
spellings, the energy correction and the eigen machinery. It is
scheduled as a standalone campaign (roadmap item) with the
energy-exactness gates re-proven; sw2 is untouched until then (its
`u_physical` / `v_physical` properties remain the interim conversion
points and are retired by the campaign).

**(d) `state.chart`.** The derived, **read-only** expert namespace on
the vocabulary State classes: `state.chart["w"]` / `state.chart.w`
is the chart-native quantity (`J\omega` on a mapped column), the
identity on unmapped grids and for uncoupled components (terrain
horizontal `u`, `v`); `u, v, w = state.chart.velocities` destructures
in grid axis order. `chart` deliberately means "whatever the chart
convention makes this component" — the sphere's horizontal quantity
is a coordinate velocity, the mapped vertical is a J-weighted flux;
each State documents which. No `*_physical` names survive once a
package satisfies the invariant.

## 3. Physical input (ICs)

With the invariant, `Model.set_fields` takes physical fields by
definition — no conversion layer. Package by package:

- **nonhydro2**: a physical `w` IC already works today, mapped grids
  included (the prognostic *is* physical `w`). The first pressure
  projection removes the non-solenoidal part and enforces
  no-normal-flow at the bed — the effective IC is the divergence-free
  projection of what was set (standard, desirable).
- **hydrostatic**: `w` is not an IC degree of freedom (continuity
  slaves it to `u`, `v` every substage); nothing to set. After (b)
  the *reported* `w` is physical automatically.
- **shallowwater2**: the only genuine input conversion (pointwise
  metric rescale); it arrives with campaign (c). No interim shim.

The general fact removing any ill-posedness worry: the
physical-to-chart inverse is **triangular and explicit** on every
supported composition (horizontal components convert pointwise, the
vertical then subtracts slope terms built from them) — no solve, an
exact round trip, and linear (so propagator gradients w.r.t. physical
ICs stay exact).

## 4. Side prediction (cousin-leak audit)

Since nonhydro2's mapped `w` is physical, its `-N^2 w` buoyancy
coupling was correct all along; the open −6.5e-3 mapped energy leak
(`energy_metric_asymmetry.md` §1.6) is then most likely ordinary
interpolation-transpose truncation, not a missing term — the pending
n-scaling probe should show it *converging*, unlike the hydrostatic
case pre-`d629a489`.
