---
status: active
date: 2026-07-23
---

# Traveling wave packets — direction control on bounded axes

**Status: WP-D1..D7 approved as recommended (owner, 2026-07-23);
P1 + P2 shipped 2026-07-23 (entry in `../../roadmap/done.md`; gates:
mirrored tests + ruff green, drift measured at 96–102% of the
continuum group speed both ways). P3 deferred
(`../../roadmap/deferred.md`); P4 rides the docs cycle.** The concrete
construction that shipped (a simplification of §1 discovered during
implementation): because the family projection is a *real* operator,
`Re ∘ P ∘ (E ⊙ C) = P(E ⊙ Re C)`, so no complex field ever enters
the projection — the kernel builds the **real traveling carrier**
`Re C` directly. Per component, the periodic-axes structure
`R = Σ_j (z0 + i z1) T_j / Σ_j T_j²` is extracted from the temporal
quadrature pair (`em.mode` at `phase` and `phase + π/2`) by
contraction against the component's own sampled trig vector `T`
(family read off the coefficient-space type, argument
`m π (x − x_min)/L` per the documented amplitude convention of
`spatial/operators/trig.py`), and the carrier is
`Re[R ⊗ e^{iσθ}]` with `σ = direction · sign(Δω)` resolved from
`em.mode` frequencies at the neighboring carrier indices. The
factory then envelopes and projects exactly as the standing path
does.

Driver (owner, 2026-07-23): while porting the reflecting-wave-package
example, `nh.wave_package` on a walled-vertical grid was found to
produce a packet that splits into an upward and a downward beam. The
investigation (§0) cleared the eigenmodes: the split is the correct
physics of Gaussian-masking a *standing* carrier, which is the only
carrier a bounded axis offers at a single frequency. The owner wants
single-sided (traveling) packets as an upstream feature, and wants it
**general** — not a walled-in-z special case: any bounded axis, any
eigenmode tier, both models. This plan is design-only; implementation
starts after owner sign-off of the WP-D calls in §2.

## 0. Context — what the investigation established (2026-07-23)

All findings from executed CPU probes on the example's grid
(256×1×128, walled z, f0 = 1e-4, N² = 2.5e-5, carrier
`{"x": 15, "z": 15}`, Gaussian mask 150 m at domain center):

- **The bare walled mode is exact.** The Leray-projected linear
  tendency of `em.mode("wave+", ...)` equals
  `omega * mode(phase + pi/2)` to 2.5e-15 max relative error, and
  1080 AB3 steps keep correlation 0.9999997 against the analytically
  phase-advanced mode (the 7e-3 L2 drift is stepper truncation at
  `omega dt = 0.07`).
- **The mask creates the split; the projection is innocent.** The
  masked, unprojected state evolves with its vertical energy centroid
  pinned at the mask center while the spread grows 75 → 420 m (two
  beams, both walls); the factory output (masked + `wave+`
  re-projection, 9.9e-2 L2 apart at t = 0) evolves identically to
  three digits in every centroid/spread number.
- **A traveling carrier gives one packet.** The same factory call on
  the fully periodic twin grid descends as a single compact packet at
  the predicted 3.7 cm/s group speed. The old-stack example got its
  single reflecting packet exactly this way (the triply periodic twin
  `ModelSettings` in the pre-port `examples/nonhydro/wave_package.py`).
- **Mechanism.** A standing carrier is the equal sum of the two
  traveling carriers, so localizing it along the bounded axis hands
  you both packets. A single-sided packet in a bounded domain exists
  regardless — it is a superposition of many bounded-axis modes whose
  complex coefficients carry phases linear in the mode number (the
  one-way pulse on a fixed-end string) — but the current factory
  cannot build it: it masks the *real* mode
  ([`initial_conditions.py`](../../../src/fridom/nonhydro2/initial_conditions.py):584-592),
  and real input yields direction-symmetric coefficients, always.

Current surface, for reference: the factory refuses channels
([`initial_conditions.py`](../../../src/fridom/nonhydro2/initial_conditions.py):310-327),
`sw2` has `single_wave` but no `wave_package`
([`shallowwater2/initial_conditions.py`](../../../src/fridom/shallowwater2/initial_conditions.py):363),
and the walled behavior has no test coverage (the `wave_package`
tests run on the periodic fixture only,
[`test_initial_conditions.py`](../../../tests/nonhydro2/test_initial_conditions.py):260).

## 1. Design — the analytic-signal kernel

One construction covers every tier, because it is expressed against
the eigenmodes interface (complex mode synthesis, labeled families,
per-mode frequencies) and never names an axis, a trig family, or a
grid topology:

> **A single-sided packet is the family projection of the
> analytic-signal packet.** Build the complex carrier state
> `C = mode + i * mode_H`, where `mode_H` is the spatial quadrature
> (Hilbert partner) of the mode along the traveling axis; apply the
> user's envelope (`E ⊙ C`, a coordinate callable sampled per
> component at its own staggered nodes through the existing
> signature-inspected sampler,
> [`_sample`](../../../src/fridom/nonhydro2/initial_conditions.py):342);
> project onto the carrier's mode family; take the real part;
> normalize by the same amplitude convention as `em.mode`.

Why this is the general spelling:

- **Bounded axes.** Against the bounded-axis mode functions the
  complex target produces coefficients
  `c_m ≈ ĝ(m - m_c) e^{±i (m - m_c) x_0}` — the linear-in-m phase
  pattern of a one-way packet. Under the family evolution
  `c_m e^{-i omega_m t}` the envelope drifts one way, reflects at the
  wall (the coefficients rephase into the mirrored packet), and
  returns — the desired demo dynamics.
- **Periodic axes.** The quadrature partner of `e^{i k x}` is itself
  (times `-i`), so the kernel reduces to the current construction and
  the direction stays what it is today: the sign of the integer
  wavenumber. No behavior change.
- **Tier independence.** On the analytic tier `mode_H` is closed-form
  (the per-component sin ↔ cos swap of the walled trig families,
  synthesizable through the existing
  [`eigenstates.py`](../../../src/fridom/model/eigenstates.py)
  helpers — `hermitian_mode_data`, `synthesize_columns` — that
  [`_mode_branch`](../../../src/fridom/nonhydro2/eigenmodes.py):967
  already drives, including its `synth(0)/synth(pi/2)` +
  `envelope_scale` normalization pattern). On the numeric channel
  tier `mode_H` is a discrete Hilbert transform of the column
  profile — no closed form needed. A future tier (spherical charts,
  analytic walled-horizontal at f = 0) plugs in by providing the same
  two ingredients.
- **Purity for free.** The family projection is part of the
  construction, so the packet is family-pure by construction — same
  as today's factory exit.

Direction semantics (see WP-D3): the user-facing sign is the sign of
the **envelope drift** (group velocity) along the axis — what the eye
sees — not the sign of the carrier wavenumber (for internal waves the
two are opposite in the vertical). The kernel resolves the carrier
orientation numerically: evaluate the family frequency at the
neighboring indices along the traveling axis and orient the quadrature
so `sign(d omega / d m)` times the carrier sign matches the requested
drift. This works identically on analytic and numeric tiers and needs
no closed-form dispersion. A vanishing frequency difference (carrier
at a band extremum, or the vortical family's `omega = 0`) is a taught
error, not a silent standing fallback.

The kernel lands in
[`fridom/model/eigenstates.py`](../../../src/fridom/model/eigenstates.py)
next to its ingredients; the per-model `wave_package` factories stay
thin wrappers, like every other factory pair.

### 1.1 The surface, by example (under the §2 recommendations)

The docs example's case — walled z, one packet sinking and
reflecting off the bottom; the envelope is a plain coordinate
callable, the sign is the envelope drift:

```python
def envelope(x, z):
    return jnp.exp(
        -((x - 500.0) ** 2 + (z - 500.0) ** 2) / 150.0 ** 2)

omega, packet = nh.wave_package(
    model,                                # rigid lid and bottom
    {"x": 15, "y": 0, "z": 15}, "wave+",
    envelope=envelope,
    traveling={"z": -1})                  # +1 rises instead
```

The common case keeps a one-liner through the convenience factory,
`envelope=nh.gaussian_envelope(pos={"x": 500.0, "z": 500.0},
width={"x": 150.0, "z": 150.0})`. The callable is sampled per
component at its own staggered nodes; the axes its signature names
are the enveloped axes (a named-but-unused axis is harmless —
constant along an axis means unenveloped there). Omitting
`traveling=` keeps today's standing (two-beam) packet,
value-identical in the Gaussian case. On periodic axes nothing
changes: the signed carrier index
already picks the side (`k={"z": -8}` on a periodic vertical), and a
`traveling=` key naming a periodic axis is a taught error pointing at
the `k` sign. Further taught errors: a traveling axis the envelope's
signature does not name (no localized envelope to move), and a
non-propagating family
(`"vortical"`, `omega = 0`). `family` / `branch` / `phase` /
`at_time` and the `(omega, state)` return compose unchanged; the
`sw2` parity factory (P2) reads identically (doubly periodic, so
direction lives in the `k` signs until P3), and the channel tier
(P3) is the same kwarg on the bounded channel axis
(`traveling={"y": -1}`).

Nuance for the WP-D3 review: this leaves a deliberate asymmetry —
the `k` sign on periodic axes is the *phase* direction (the
`single_wave` convention), `traveling=` on bounded axes the *group*
drift. The alternative (accepting `traveling=` on periodic axes as a
group-direction override that orients the carrier sign) gives one
consistent "which way does it go" knob everywhere, at the cost of two
spellings for the same thing on periodic grids.

## 2. Owner calls — decisions needing sign-off

- **WP-D1 — API surface.** Recommended: one keyword on the factories,
  `traveling: Mapping[str, int] | None = None` (e.g.
  `traveling={"z": -1}`), axis-keyed like `k`, composing
  over several bounded axes if a tier ever offers them. Alternatives
  considered: signed indices on bounded axes in `k` (breaks the
  existing unsigned physical-mode convention and silently changes
  meaning), a bare `direction=` int (not general beyond one bounded
  axis). Name is open: `traveling=` vs `direction=` vs `drift=`.
- **WP-D2 — default.** Recommended: `None` keeps today's standing
  packet (statu quo, no silent behavior change; the docs example opts
  in explicitly). Alternative: default single-sided along every
  bounded masked axis with a nonzero carrier index — arguably the
  physics the name promises, but it needs an arbitrary default sign
  and changes shipped (if untested) behavior.
- **WP-D3 — sign semantics.** Recommended: sign of the envelope
  drift (group velocity), resolved numerically as in §1. Alternative:
  sign of the carrier phase along the axis — cheaper to implement,
  but for internal waves it is the opposite of what the animation
  shows, which is exactly the confusion this feature exists to
  remove.
- **WP-D4 — kernel.** Recommended: the analytic-signal projection of
  §1 (tier-uniform, purity built in). Alternative: closed-form
  quadrature-pair synthesis on the analytic tier only — less code
  today, dead end for channels and future tiers.
- **WP-D5 — tier scope of the first landing.** Recommended: P1 + P2
  (analytic tier in `nh2`, and the `sw2` parity factory on its
  periodic tier) now; the channel tier (P3) deferred with a promotion
  trigger (the first docs example or user request needing a packet in
  a walled channel). The channel tier adds the numeric Hilbert step
  and its distributed-synthesis legalization (the channel synthesis
  is currently single-device-tested,
  [`test_initial_conditions.py`](../../../tests/nonhydro2/test_initial_conditions.py):36-42).
- **WP-D6 — standing path stability.** Recommended: the
  `traveling=None` path keeps the existing enveloped-then-projected
  construction value-identical (`gaussian_envelope` reproduces the
  exact `sample_gaussian_mask` expression), and the new kernel is a
  separate branch. Alternative: route everything through the new kernel — one
  code path, but the periodic-tier output changes at the
  polarization-weighting level (mask-then-project freezes the
  polarization at the carrier; coefficient placement varies it across
  the bump), which would invalidate current expectations for no user
  benefit.
- **WP-D7 — envelope surface (owner prompt, 2026-07-23).** The
  `mask_pos` / `mask_width` pair is replaced ("envelope" is the
  physics term: the profile that moves at the group velocity; "mask"
  was old-stack implementation vocabulary). Recommended: one required
  keyword `envelope=`, a coordinate-named callable sampled per
  component on its own staggered nodes (signature-inspected, the
  same convention as `grid.create_field(init=...)` and the internal
  `_sample`), plus a shared `gaussian_envelope(pos=..., width=...)`
  convenience factory reproducing today's expression — the common
  case stays a one-liner, and anisotropic, rotated, tapered, or
  complex-valued (chirped) envelopes become possible with no factory
  changes, on the standing and the traveling path alike.
  Alternatives: a renamed parameter pair (`envelope_pos=` /
  `envelope_width=` — better name, no generality gained), or
  callable-only without the Gaussian helper (the overwhelmingly
  common case gets wordier and loses the standardized width
  convention). The factory is unreleased, so the old pair is removed
  outright, no deprecation path.

## 3. Phases and gates

- **P0 — this plan.** Owner review of WP-D1..D6. Gate: sign-off in
  chat.
- **P1 — kernel + `nh2` surface.** The eigenstates kernel (complex
  carrier, quadrature synthesis on the analytic tier, direction
  resolution, amplitude normalization), the `envelope=` +
  `traveling=` surface on `nh.wave_package` (the `mask_pos` /
  `mask_width` pair removed outright per WP-D7), the shared
  `gaussian_envelope` helper, taught errors (traveling key on a
  periodic axis — use the sign of `k`; a traveling axis the
  envelope's signature does not name; zero carrier index along the
  traveling axis; degenerate `d omega = 0`), docstrings. Tests (mirrored,
  `tests/nonhydro2/test_initial_conditions.py` or a
  `test_initial_conditions_packets.py` shard, coarse ≤16³ grids):
  drift sign exact and magnitude within ~20% of the
  finite-difference group velocity from neighboring `em.mode`
  frequencies, for both signs; family purity of the output; the
  standing Gaussian case reproduces today's fields bitwise
  (`gaussian_envelope` against `sample_gaussian_mask`); a free-form
  callable envelope on the standing path; the taught errors. Gate: mirrored tests + ruff, 95% patch coverage.
- **P2 — `sw2` parity.** `sw.wave_package` as the same thin wrapper
  (closes the factory parity gap); on `sw2` every analytic axis is
  periodic, so `traveling=` always teaches there until P3 — the value
  is the shared kernel plus surface parity. Same test pattern in
  `tests/shallowwater2/`. Gate: as P1.
- **P3 — channel tier (deferred).** Numeric Hilbert quadrature of the
  channel columns, packets along the bounded channel axis (Poincaré
  packets running into a wall; Kelvin packets along the wall already
  work via the sign of the periodic `k` once the factory accepts
  channels), multi-device legalization of the channel synthesis.
  Promotion trigger per WP-D5.
- **P4 — the consuming example.** `examples/nonhydro/wave_package.py`
  switches to `traveling={"z": -1}`: one packet descending, one
  bottom reflection — the old example's story on the intended
  surface. Runs through the normal docs-review cycle; the two-beam
  investigation figures stay available as review context.

## 4. Sizing

Small campaign. P1 is one focused session (the synthesis ingredients,
projection, and normalization machinery all exist and are already
composed once in `_mode_branch`; the new kernel is a second
composition plus direction logic). P2 is a fraction of a session. P3
is the only piece with real unknowns (numeric quadrature quality near
the walls, distributed synthesis) and is deferred.

## 5. Non-goals

- **No change to the wave-maker module** —
  a forced oscillating source radiating both beam pairs is physical,
  and its `mask_*` vocabulary stays: a forcing-region mask is a
  mask, not a packet envelope (`sample_gaussian_mask` keeps its
  module-internal role there).
- **No change to the random-phase ICs** (`random_state` /
  `random_waves`): direction-symmetric content is their point.
- **No new tier work** beyond the channel Hilbert step of P3; the
  kernel's tier independence is the designed-for hook for spherical
  charts, not an implementation promise here.
- **No `custom_vjp` anywhere** — the kernel is host-side assembly
  feeding `set_state`; the differentiability policy's step-path rules
  are not triggered, and the construction stays plain `jnp` so
  `jax.grad` w.r.t. mask parameters keeps working if a user closes
  over them.

## 6. Risks

- **Edge strata.** Near the ends of the bounded mode lattice
  (`m` near 0 or `n`) the quadrature partner is poorly represented
  and the packet tail truncates; an envelope much wider than the
  distance to the wall degrades single-sidedness. Free-form envelope
  callables widen the reachable pathology space (sharp edges spread
  coefficients into the edge strata). Mitigation: the purity and
  drift tests quantify it at coarse resolution; the docstring states
  the "smooth, well inside the domain, several carrier wavelengths
  wide" validity window (same physics caveat the old example
  obeyed), and nothing is guarded.
- **Direction resolution at band edges.** The numeric
  `d omega` probe needs both neighbor indices representable;
  boundary-of-lattice carriers fall back to one-sided differences or
  the taught degeneracy error. Kept explicit in tests.
- **Normalization convention drift.** The envelope convention must
  match `em.mode` ("largest horizontal-velocity envelope is one") so
  factory amplitudes stay comparable across standing/traveling;
  reuse `envelope_scale` on the packet's own temporal quadrature
  pair.

## 7. Roadmap tie-in

Shipped state (2026-07-23): the P1 + P2 record lives in
[`../../roadmap/done.md`](../../roadmap/done.md), the P3 channel
tier in [`../../roadmap/deferred.md`](../../roadmap/deferred.md)
with its promotion trigger. What remains of this plan is P4 — the
example switch to ``traveling={"z": -1}`` — which rides the docs
review cycle like every example port; when it lands, this plan
moves to `design/plans/done/`. The docs plan's "Open upstream
items" section ([`docs_examples_plan.md`](docs_examples_plan.md))
is under the owner's own edit and is left to the owner.
