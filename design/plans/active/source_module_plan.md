---
status: active
date: 2026-07-24
---

# Generalized source module — separable volumetric forcing

**Status: SRC-D1..D8 approved as recommended (owner, 2026-07-24),
including the SRC-D5 generic-law call. Implementation in flight
(P1–P3); P4 rides the docs cycle.** Driver (owner brainstorm,
2026-07-24): the
two v1-ported wave makers (`nh.GaussianWaveMaker`,
`nh.PolarizedWaveMaker`) are special cases of one machine — a sum of
separable terms `Q(x) · g(t)` added to the tendency, where `Q` is a
state-valued spatial pattern (possibly one component) and `g` a
scalar time law. The owner wants that machine as a single generic
module: any field or the full state, multi-phase forcing (e.g.
`f1 sin + f2 cos`), and packet forcing, without one bespoke module
per use case. All code snippets are illustrative, not normative.

## 0. Context — what exists

- `nonhydro2/modules/gaussian_wave_maker.py`: one variable,
  Gaussian mask (AUXILIARY, sampled on the forced variable's own
  space) × `A sin(2π f t)`; `wavemaker.<variable>.{frequency,
  amplitude}` dynamic-leaf parameters; zero-halo `extra_halo`
  substitute (V-N2).
- `nonhydro2/modules/polarized_wave_maker.py`: four-component
  packet precomputed at bind from the analytic eigenmodes,
  × `A sin(ω t)`; fixed parameter name
  `wavemaker.polarized.amplitude` (silently limits a model to one
  instance); frequency read-only derived structure; re-implements
  the WavePackage construction (envelope, mask-at-own-nodes, branch
  re-projection) inside `bind`, taking `.real` and doubling — the
  complex phase information is discarded.
- The IC side already owns the ingredients: coordinate-named
  envelope callables (`gaussian_envelope`, `envelope_axes`,
  `sample_envelope` in `model/eigenstates.py`), `wave_package`
  (envelope × carrier × projection, `traveling=`), and the scalar
  time-law family (`Ramp`, `TimeFunction`, `TimeDependent`).
- AUXILIARY fields are module-owned **carry**: materialized once at
  assembly, threaded through the jitted step as traced pytree
  arguments (`state[name]`), never baked as XLA constants — so
  patterns are compile-safe and swappable without recompile.

## 1. The shape of the machine

Everything requested is an instance of

    δz += Σ_i Q_i(x) g_i(t)

**Separability is a binding constraint, not a convenience**: it is
what makes the term jit-pure with zero halo (patterns freeze into
AUXILIARY fields; the law is a scalar off the traced clock), keeps
amplitude/frequency as sweepable dynamic leaves
(`model.update_parameters`, no re-assembly), and keeps `jax.grad`
through the term clean. A genuinely non-separable `S(x, t)` is a
different machine (per-step recompute — `ProfileFunction`
territory) and stays out of this module.

## 2. Decision calls (SRC-D1..D8)

### SRC-D1 — one module instance = one separable term

A `Source` instance carries exactly one `(pattern, law)` pair. Sums
(`Σ_i`) are spelled as multiple instances: the tendency-level sum
already exists (`multiple_wave_makers.py` precedent), and the
complex-pattern spelling (SRC-D4) absorbs the common quadrature pair
into a single instance. No `terms=[...]` list on the module.

### SRC-D2 — name, home, labels

`fr.model.modules.Source` — model-agnostic, next to `relaxation.py`
and `ramping.py`. "Source" is the PDE vocabulary the docstrings
already use, is naturally volumetric (unlike "forcing", which reads
as boundary forcing), and is correct for non-momentum targets
(unlike "BodyForce" — a buoyancy or tracer injection is not a
force). Each instance takes a mandatory label (first positional
argument): parameters publish as `source.<label>.{amplitude,
frequency, phase}`, the tendency term is named by the label. Labels
must be unique per model (the parameter table enforces collisions);
this dissolves the one-polarized-maker-per-model limit.

### SRC-D3 — the pattern surface

`pattern` is a mapping `variable -> (coordinate-named callable |
ScalarField)`; a `State` is accepted directly (it is such a
mapping). Callables follow the IC envelope convention — the
signature names the coordinates the pattern varies along — and are
sampled on **the forced variable's own space** (staggered faces for
velocities, centers for scalars) via the generic sampling helper;
the staggering-aware duplicate `sample_gaussian_mask` is deleted.
Patterns materialize as AUXILIARY fields `source_<label>_<var>`
(two per component for complex patterns, SRC-D4).

Shipped mechanism (P2): a new declarations-layer descriptor
`LikeField(name)` — an AUX declaration on `LikeField("u")` adopts
the referenced field's own resolved space during assembly step 1,
so `Source` hardcodes no staggering vocabulary (the generalization
of the nh `_variable_pattern` table).

Bind-time validation (the GaussianWaveMaker precedent): every
pattern coordinate must be a grid coordinate; every forced variable
must exist and be PROGNOSTIC; a field-valued pattern must live on
the forced variable's own (interned) space. TDF-D6 carries over
verbatim: a time-dependent input (Ramp/ProfileFunction-valued
dependency, `time_dependent` field) is refused with the taught
error — ramp the amplitude instead. Zero-halo `extra_halo`
substitute (V-N2) carries over.

### SRC-D4 — one phase convention, complex patterns first-class

The term is

    S(x, t) = A · Re[ Q(x) · e^{-i(2π f t + φ)} ]

(sign convention corrected during P2: the drafted `−φ` spelling
contradicted its own sine parenthetical and the v1 `+A sin` parity —
only `+φ` satisfies both)

- real `Q` ⇒ `A cos(2π f t + φ) Q(x)` (sine = `φ = -π/2`);
- complex `Q` ⇒ the traveling quadrature pair
  `A [cos(·) Re Q + sin(·) Im Q]`, expanded internally into two
  real AUXILIARY fields per component.

One convention covers the single-field case, the owner's two-phase
case, and packet forcing; real patterns are literally the special
case `Im Q = 0`. The engine applies **no hidden factors**: the v1
polarized factor 2 belongs to the packet construction
(`wave_package`), not the forcing module.

### SRC-D5 — the time law

The blessed law is `Harmonic(amplitude, frequency, phase)`,
publishing the three labeled parameters of SRC-D2 as dynamic
leaves:

- `amplitude : float | Ramp | TimeFunction` — the TDF-D6 spin-up
  doctrine ("ramp the amplitude, not the structure") becomes
  uniform across all sources.
- `frequency : float` — a sweepable leaf (`update_parameters`
  between runs; detuning experiments), but **not rampable within a
  run**: `A sin(2π f(t) t)` has instantaneous frequency
  `f + t f'`, not `f(t)` — the naive law is a footgun, and a
  correct chirp needs the explicit phase integral. Chirps are
  spelled through the escape hatch.
- `phase : float`.

Escape hatch: `law=` accepts any `TimeDependent` (e.g. a
`TimeFunction` chirp with the phase law written out; its `params`
stay dynamic leaves, so sweeping them never recompiles). A generic
law publishes **no** named parameters (its leaves are anonymous);
complex patterns require `Harmonic` (the quadrature expansion needs
`f` and `φ` structurally). Resolved (owner, 2026-07-24): a generic
law publishes no named parameters — its leaves stay anonymous
dynamic leaves (recompile-free to sweep, but not addressable by
name through `update_parameters`).

### SRC-D6 — `gaussian_envelope` → `gaussian`; shapes get a home

The principle: **the parameter names the role** (`envelope=` in
`wave_package`, `pattern=` in `Source` — both keep their names),
**the builder names the shape**. Hence:

- `gaussian_envelope` → **`gaussian`** (`gaussian_profile` collides
  with the `ProfileFunction` time-dependence vocabulary;
  `gaussian_mask` reads as windowing/immersed-mask). The kwargs
  disambiguate: `gaussian(pos=..., width=...)`.
- `width : float | Mapping[str, float]` — a bare float broadcasts
  to every axis named in `pos` (owner request 2026-07-24).
- New neutral home `fridom/model/shapes.py`, exported at `fr.model`
  level (where `Source` lives); `nh`/`sw` top-level re-exports
  keep working under the new name. Future siblings (`cosine_bell`,
  `tanh_front`, super-Gaussian plateau) land there when a use case
  shows up — none committed now.
- Internal helpers renamed in the same change so one vocabulary
  survives: `envelope_axes` → `pattern_axes`, `sample_envelope` →
  `sample_pattern` (non-public; call sites in
  `initial_conditions.py` updated).

### SRC-D7 — `wave_package(..., quadrature=True)`

`wave_package` grows a `quadrature=` flag: `True` returns
`(omega, complex State)` — the packet before the real part is
taken — for direct use as a complex `Source` pattern. Packet
forcing then dissolves into IC-machinery reuse:

    omega, packet = nh.initial_conditions.wave_package(..., quadrature=True)
    maker = fr.model.modules.Source("packet", pattern=packet,
        law=fr.model.Harmonic(amplitude=A, frequency=omega / (2 * jnp.pi)))

The forcing frequency becomes an ordinary sweepable parameter
instead of read-only derived structure — deliberate detuning off
resonance becomes a one-line experiment. Normalization/doubling
conventions stay in `wave_package` (SRC-D4: engine factor-free);
parity with the v1 polarized maker is a validation gate (§4).

### SRC-D8 — both v1-ported makers are deleted

`nonhydro2.GaussianWaveMaker` and `nonhydro2.PolarizedWaveMaker`
are removed, not wrapped: the direct `Source` spelling is the same
line count, and the new stack is pre-cutover (no public users). The
old-stack (`fridom/nonhydro`) originals stay frozen as always. The
two consuming examples (`examples/nonhydro/internal_wave_maker.py`,
`examples/nonhydro/multiple_wave_makers.py`) are re-spelled; as
reader-facing content they ride the docs cycle under the private
owner-review workflow (AGENTS.md).

## 3. API sketches (illustrative)

Today's Gaussian wave maker, spelled directly:

    wave_maker = fr.model.modules.Source(
        "wave_maker",                              # -> source.wave_maker.*
        pattern={"u": fr.model.gaussian(
            pos={"x": 400.0, "z": 75.0}, width=4.0)},
        law=fr.model.Harmonic(amplitude=1e-5, frequency=1 / PERIOD))
    model = nh.Model(..., modules_extra=(wave_maker,))
    model.update_parameters({"source.wave_maker.amplitude": 2e-5})

Two-phase forcing (`f1 sin + f2 cos`, shared frequency):

    a = fr.model.modules.Source("phase_a", pattern={"b": f1},
        law=fr.model.Harmonic(amplitude=A, frequency=f, phase=-jnp.pi / 2))
    b = fr.model.modules.Source("phase_b", pattern={"b": f2},
        law=fr.model.Harmonic(amplitude=A, frequency=f))

Chirp through the escape hatch (explicit phase law):

    chirp = fr.model.TimeFunction(
        lambda t, f0, rate: jnp.sin(2 * jnp.pi * (f0 + 0.5 * rate * t) * t),
        params=(1e-4, 1e-9))
    scan = fr.model.modules.Source("scan", pattern={"u": env}, law=chirp)

## 4. Validation gates

- Mirrored tests: `tests/model/modules/test_source.py` (engine:
  pattern tiers, complex expansion, label/parameter publication,
  bind-time refusals) plus an nh parity shard — the new spelling
  reproduces the deleted makers' tendencies (Gaussian: exactly;
  polarized: to tolerance, factor conventions pinned).
- Differentiability policy: one autodiff regression — `jax.grad`
  of a quadratic loss w.r.t. `source.<label>.amplitude` and
  `.frequency`, finite and matching central FD to rtol 1e-4
  (≤16²/8³ grid, ≤10 steps). Shipped via the `_chunk_body` kernel
  surface (policy-sanctioned): `Model.propagator` blanket-refuses
  `wrt=` parameters of modules that own materialized AUX fields —
  a pre-existing limitation the old maker shared. Relaxing the
  per-owner refusal for parameters the materialization does not
  depend on is a candidate follow-up, not in this plan's scope.
- Coverage ≥ 95% patch; `ruff` zero errors; merge gates per
  AGENTS.md.

## 5. Phases

- **P1** (`refactor/shapes-module`): `gaussian` rename + `shapes.py`
  relocation + scalar-width broadcast + internal helper renames.
  Mechanical; keeps `initial_conditions` behavior identical.
- **P2** (`feat/source-module`): `Source` + `Harmonic` engine,
  tests, autodiff shard.
- **P3** (`feat/wave-package-quadrature`): `quadrature=` on
  `wave_package`, polarized parity test, delete both v1-ported
  makers.
- **P4**: example re-spells — rides the docs cycle
  (owner-reviewed, `docs-review` skill).
