# D2.4 — Host-side parameter access and the eigenmode seam

Research report (see [`README.md`](README.md) for status).

## 1. The access surface: three tiers

**Tier 1 — `model.parameters`, primary.** A read-only
`Mapping[str, value]` over the D2-resolved provides namespace;
values **read live from the model's current carry**, not a snapshot.
Mapping access only — no `__getattr__` sugar (same reasons D1.5
rejected it on State). The repr is a name → value → providing-module
table, doubling as the assembly log entry. Missing name →
`MissingParameterError` with hint. Ambiguity is impossible by
construction (one-provider rule at assembly); instance-plural
modules (two `Tracer`s) publish prefixed names or nothing (tier 2).
**Ramp-valued slots return the `Ramp` object itself**, never a
silently-evaluated number; `model.parameters.at_time(t)` evaluates
explicitly — load-bearing for §5.

**Tier 2 — `model.module(...)`, typed lookup, secondary.**
`model.module(nh.FPlaneCoriolis).f0`;
`model.module(nh.Tracer, name="dye").kappa`. Returns the current
instance from the carry. Zero instances → hinted
`MissingModuleError`; several without `name=` →
`AmbiguousModuleError`. For module-private knobs deliberately not
published, and for introspection — not the eigenmode feed.

**Tier 3 — direct references, demoted.**
`cor = nh.FPlaneCoriolis(f0=...)` remains legal wiring, but the
reference is only guaranteed live **up to assembly** (the run
advances the carry functionally; whether the user's object tracks
it depends on D4). Rule: **read post-assembly values through the
model, not your constructor variables.**

### Staleness and mutation rules

1. Pre-assembly: mutation free (plain objects).
2. Assembly snapshots derived data (aux parameter fields, `default=`
   backgrounds, solver precomputes) with assembly-time parameters.
3. **Post-assembly: direct attribute mutation raises**
   (`ImmutableParameterError`, the D1.5 teaching-shim idiom) — it
   would be silently wrong three ways: stale aux fields, stale
   solver precomputes, no effect under an already-traced run.
4. The only in-run parameter dynamics are traced: Ramp-valued slots
   and the self-update hook — carry leaves, visible to
   `model.parameters`.
5. **Parameter sweeps re-assemble** — cheap where it matters: a
   swept leaf gives an identical treedef, so the jit cache is
   shared across sweep members.

## 2. The eigenmode-object seam

**Both constructors**: explicit-scalar for standalone use +
`from_model` extracting via the access surface. One eigenmode class
per model family, in the model package.

```python
em = nh.eigenmodes.Eigenmodes(grid, f0=1e-4, n2=2.5e-5, dsqr=1.0, discrete=True)
em = nh.eigenmodes.from_model(model)
```

- **Scalars, not module objects** — the old
  `omega(s, f0, stratification_n2, dsqr, k)` shows the true
  dependency is three floats. Verified: nonhydro eigenmodes read
  exactly `f0, stratification_n2, dsqr` from mset (**not `Ro`** —
  Ro enters only nonlinear analysis: NNMD/optimal-balance);
  shallowwater reads `f0, csqr`.
- **Grid is required positional** — the object consumes grid
  primitives the model doesn't mediate (`op.eigenvalues(grid,
  coeff_space) -> Symbol`, per-origin coefficient spaces,
  `grid.wavenumbers`).
- **Surface (seam only)**: `em.omega(s) -> Symbol`;
  `em.q(s)/em.p(s) -> State` on per-variable coefficient spaces
  (phase shifts baked in, energy normalization inside);
  `em.projector(s) -> Callable[[State], State]`;
  `discrete=True|False` replaces `use_discrete`.
- **`from_model` validation** — successor of the old
  spectral-analysis checks: required names present (hinted),
  provided values constant scalars (a varying aux field → error,
  as the old `array_is_constant`), factors Fourier-diagonalizable
  (structural), `f0 == n2 == 0` → error.
- Old `SingleWave` applied
  `time_stepper.time_discretization_effect(omega)` — stays a
  **stepper** concern applied by the recipe, never folded into the
  eigenmode object.
- Precedents: Dedalus EVP/IVP as two problems sharing bases with
  explicit params (validates the standalone form); pyqg linear
  stability from the model instance (validates `from_model`);
  Oceananigans users hand-holding `N, f` script constants — the
  shared-binding hazard `from_model` removes. The in-tree
  equatorial-waves example (hand-built Hermite modes) is the
  witness that not all eigenmode families are spectral-diagonal
  (risk 3).

## 3. Projection-family mapping

Found: `framework/projection/{projection, spectral_projections,
geostrophic_time_average, optimal_balance, nnmd}.py`; there is
**no** `nonhydro/projection/` (nonhydro reuses the framework
classes). Plus the in-step pressure machinery.

| Old | New home / seam | Traced? |
|---|---|---|
| `Projection` base (takes mset) | **Deleted** — projections are plain callables | — |
| `GeostrophicSpectral` | `em.projector(0)` | host-side |
| `WaveSpectral` | `em.projector(+1, -1)` | host-side |
| `DivergenceSpectral` | `em.projector("d")` (analysis-side) | host-side |
| `GeostrophicTimeAverage` | consumes an assembled `fr.Model` (a linearized variant the caller builds) | host driver around traced runs |
| `OptimalBalance` | consumes two `fr.Model`s + base projector; the ramp mutation → a **`fr.Ramp`-valued scaling parameter** | host driver |
| `NNMD` | consumes `(Eigenmodes, model)` — needs both linear modes and the nonlinear term/scaling | host driver |
| pressure projection | **a stage owned by `nh.DynamicalCore`** — not in this family | **traced, in-step** |

Sharp line: the pressure stage enforces the constraint inside the
step and is built from `op.eigenvalues` directly (no f0/N²); the
geostrophic/wave/divergence projectors are analysis-side eigenmode
consumers — mathematically adjacent, opposite sides of the trace
boundary, opposite parameter seams (owner-internal vs `from_model`).

## 4. Pressure solvers: no cross-module parameters

Verified: `rfft_pressure_solver.py` and
`spectral_pressure_solver.py` read grid data plus **`mset.dsqr`
only**. Under D1.3 `dsqr` is owned by the same `DynamicalCore` that
owns the solver slot and the stage — the feed is owner-internal
(`(grid, self.dsqr)` at assembly; the anisotropic inverse
`1/(kx² + ky² + kz²/dsqr)` assembled from eigenvalue Symbols).
**Zero D2 machinery** — the degenerate case confirming the
ownership boundaries are drawn correctly. (A Ramp-valued `dsqr`
would force the precompute in-trace — noted, physically unlikely.)

## 5. IC consistency, including Ramp interaction

`from_model` fully solves the static case: the recipe reads the
same resolved leaves the run's modules read, through one surface —
no second binding to drift; the post-assembly mutation ban closes
the remaining hole.

**Ramp-valued parameters: error by default, explicit opt-in:**

```python
em = nh.eigenmodes.from_model(model)
# TimeDependentParameterError: 'stratification.n2' is Ramp-valued; an
# eigenmode set is a fixed-time snapshot. Pass at_time=... to evaluate.
em = nh.eigenmodes.from_model(model, at_time=0.0)
```

Silent `t=0` is rejected (a single_wave on ramping N² would report
a wrong period for most of the run, no signal); warn-and-proceed is
rejected for the same reason warnings failed the old flag
conventions. Deliberately unsolved: stepper discretization of ω
(recipe applies `time_discretization_effect`) and nonlinear
consistency (NNMD/optimal-balance territory).

## 6. Risks / open questions

1. **Provides-name vocabulary is a contract**: eigenmode classes
   hardcode required names — fine intra-package; framework-level
   consumers must consume models, not names (hence
   GeostrophicTimeAverage/OptimalBalance take models). 02_rules
   entry owed.
2. **`model.parameters` liveness depends on D4** — specified as
   "reads the current carry" so either Model-pytree resolution
   satisfies it.
3. **Non-Fourier eigenmode families** (equatorial Hermite; mixed
   grids' banded solves): family classes may implement other bases
   behind the same `omega/q/p/projector` surface — designed-for.
4. **`em.projector` composition semantics** and Nyquist handling
   are Phase-2.7 internals.
5. **Scalar-at-k evaluation of ω** (a Symbol is an operator;
   single_wave wants one number): small accessor
   (`em.omega_at(k, s)`) flagged for the Phase-2.7 surface.

## 7. Sketches

```python
# (a) single_wave via from_model
def single_wave(model, k, s=1, phase=0.0):
    em = nh.eigenmodes.from_model(model)          # f0, n2, dsqr via model.parameters
    q  = em.q(s)                                  # State on coefficient spaces
    state_hat = q.map(lambda f: fr.operators.Hadamard()(f, mode_mask(f, k)))
    t = fr.operators.Fourier(model.grid, axes=("x", "y", "z"))
    state = t.backward(state_hat * jnp.exp(1j * phase))
    return state / state.norm_l2()

# (b) standalone linear theory — no model, no modules
grid = fr.grid.cartesian.Grid(shape=(256, 256, 64), extent=..., periodic=True)
em = nh.eigenmodes.Eigenmodes(grid, f0=1.0, n2=1.0, dsqr=0.2**2)
om = em.omega(s=1)                                # Symbol; dispersion-relation plots

# (c) jet with the geostrophic projector inside
def jet(model, jet_strength=1.0, jet_width=0.16, pert_strength=0.05,
        pert_wavenum=5, geo_proj=True):
    u = model.grid.create_field(model.state_space("u"), init=lambda x, y, z: ...)
    pert = single_wave(model, k=(pert_wavenum, 0, 0), s=0)
    state = model.initial_state().replace(u=u) + pert_strength * pert
    if geo_proj:
        state = nh.eigenmodes.from_model(model).projector(0)(state)
    return state
```
