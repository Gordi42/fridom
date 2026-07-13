---
status: normative
date: 2026-07-13
---

# Model layer redesign — API sketches

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map.

Status: **drafted from the resolved decisions D1–D5** (2026-07-08).
All code is **illustrative, not normative** — names and exact
signatures may shift during class design and implementation; the
numbered design files (§4–§6, §10) are the normative reference.
These sketches double as the acceptance surface for the class-design
phase: every spelling here must remain expressible.

---

## 7.1 Hello model: nonhydro without ModelSettings

```python
import numpy as np
import fridom.framework2 as fr
import fridom.nonhydro as nh

grid = fr.grid.cartesian.Grid(
    shape=(256, 256, 64), extent=((0, 4e3), (0, 4e3), (-1e3, 0)),
    periodic=(True, True, False), names=("x", "y", "z"),
)

model = fr.Model(
    grid=grid,
    modules=(
        nh.DynamicalCore(dsqr=1.0),            # u, v, w; p (DIAGNOSTIC); projection stage
        fr.modules.FPlaneCoriolis(f0=1e-4),    # aux f_coriolis (1-DOF field); provides coriolis.f0
        nh.ConstantStratification(n2=1e-5),    # b; both coupling terms; aux n2; provides stratification.n2
        nh.CenteredAdvection(),                # transports every ADVECTED component
    ),
    time_stepper=fr.time_steppers.AdamBashforth(dt=60.0, order=3),
    name="hello",
)
print(model.report)                            # fields / parameters / schedule / halo tables

model.set_fields(b=lambda x, y, z: 0.01 * np.exp(-((z + 500) / 100) ** 2))
r = model.run(runlen=np.timedelta64(2, "D"))   # chunked scan; RunResult
model.diagnostics.pot_vort().xr.sel(z=-500).plot()
```

The same model without `ConstantStratification` has **no `b`**:
`model.state.b` raises `MissingComponentError` with the
add-a-stratification-module hint (D1.5). The preset form
`nh.Model(grid=..., coriolis=..., stratification=..., ...)` is a
thin factory producing an identical treedef (D4 preset test).

## 7.2 A field-registering module, in full

The complete `ConstantStratification` — every module capability in
~30 lines (D1 declarations/roles, D2 parameters/aux field, D3 terms,
the self-update for the Ramp-valued case):

```python
@partial(fr.utils.jaxify, dynamic=("n2_input",))
class ConstantStratification(fr.Module):
    def __init__(self, n2: float | fr.Ramp = 1e-5):
        self.n2_input = n2

    field_declarations = property(lambda self: (
        fr.FieldDeclaration.tracer(                       # => PROGNOSTIC, {TRACER, ADVECTED}
            "b", space=fr.Collocated(bc={"z": fr.BC.DIRICHLET}),
            long_name="Buoyancy", units="m/s²"),
        fr.FieldDeclaration(
            "n2", space=fr.Profile(),                     # 1-DOF ConstantSpace field (D2-R2)
            lifecycle=fr.Lifecycle.AUXILIARY,
            default=lambda self, grid, space:             # unbound-owner form (D4)
                grid.create_field(space, data=fr.resolve_at(self.n2_input, 0.0)),
            units="1/s²"),
    ))
    parameter_declarations = (
        fr.ParameterDeclaration(fr.params.STRATIFICATION_N2, attr="n2_input",
                                units="1/s²"),)
    field_references = (fr.FieldReference("w",
        hint="buoyancy couples to vertical velocity (a dynamical core)"),)

    def self_update(self, state, ctx):                    # scheduled iff n2_input is a Ramp
        return {"n2": fr.resolve_at(self.n2_input, ctx.clock.time)}

    @fr.term(advances=("w",), linear=True)
    def buoyancy_force(self, state, ctx):
        return {"w": state["b"].to(state["w"].space)
                     / ctx.params["nonhydro.dsqr"]}

    @fr.term(advances=("b",), linear=True)
    def restoring(self, state, ctx):
        return {"b": -(state["w"] * state["n2"].to(state["w"].space))
                      .to(state["b"].space)}
```

## 7.3 User tracers and role-targeted closures

```python
model = nh.Model(grid=grid, ...,
    modules_extra=(fr.modules.Tracer("dye", units="1"),           # one-liner declaring module
                   fr.closures.HarmonicMixing(kv=1e-4),           # targets TRACER by role
                   fr.closures.HarmonicFriction(av=1e-4)))        # targets Velocity family

model.set_fields(dye=grid.random.normal(model.state_space("dye"), seed=7))
# advection transports dye automatically (ADVECTED); mixing acts on {b, dye};
# friction acts on {u, v, w}; sw's p would be advected but never mixed (no TRACER role).
```

## 7.4 A dispatch override (WENO reconstruction)

```python
class MyAdvection(nh.CenteredAdvection):
    dispatch = {"reconstruct": fr.operators.WenoReconstruction(order=5)}
# merged once at assembly step 3 (before bind/dry-run/negotiate, so the
# halo trace sees WENO's wider stencil); two modules overriding one
# resolved key -> DispatchCollisionError naming both (D4 §6.2).
```

## 7.5 IMEX vertical mixing (CNAB2), and the treatment flip

```python
mix = fr.closures.VerticalMixing(kv=1e-2, treatment=fr.IMPLICIT)
#   one module, two terms: explicit horizontal (fn) + vertical with
#   ImplicitOperator (fn=None -> explicit path derived from op.apply)
model = nh.Model(grid=grid, ..., closure=mix,
                 time_stepper=fr.time_steppers.CNAB2(dt=60.0))
# step: rhs = Xⁿ + dt(3/2 Fⁿ − 1/2 Fⁿ⁻¹) + dt/2·L·Xⁿ  ->  solve(rhs, dt/2)
#       -> project the SOLVED state (D3 §5.6)
# two κ-contributing closures on one field merge into ONE tridiagonal solve.

fr.closures.VerticalMixing(kv=1e-6, treatment=fr.EXPLICIT)   # cheap problems
# ... under plain AdamBashforth; IMPLICIT under AdamBashforth is an
# assembly error (never silent demotion).
```

## 7.6 Time-dependent parameters

```python
# spin up the nonlinear term (the old Ramper + rossby-setter, dissolved):
core = nh.DynamicalCore(dsqr=1.0,
                        rossby_number=fr.Ramp(0.0, 0.1, t0=0.0, period=3600.0,
                                              curve="cosine"))
# every consumer of params["scaling.rossby"] sees the ramp, evaluated at
# RK stage times (D2 eval_params); zero mutation, zero extra modules.

# ramped stratification: the AUX field case — self_update rewrites the
# 1-DOF n2 field per substage (sketch 7.2); an eigenmode built from this
# model demands at_time=:  nh.eigenmodes.from_model(model, at_time=0.0)
```

## 7.7 The production run: writers, snapshots, resubmission

```python
# run.py — THE SCRIPT IS THE RECIPE (no pickled models, D4 §6.4)
model = build_model(grid)                       # identical assembly every job
model.set_fields(u=jet_profile, b=perturbation) # harmless on resume

model.run(
    end_time=np.timedelta64(100, "D"),          # absolute target -> resumable
    outputs=(
        fr.io.Writer("out/snap.zarr",           # xarray/xgcm-openable
                     fields=("u", "v", "w", "b"),
                     derived={"pv": model.diagnostics.pot_vort},
                     trigger=fr.every(hours=10)),
        fr.io.TimeSeries("out/energy.csv",
                         columns={"etot": model.diagnostics.etot},
                         trigger=fr.every(hours=1)),
    ),
    snapshots=fr.io.Snapshots("restart/",
                              trigger=fr.every(walltime="7.5h"),
                              keep=2, resume=True,
                              on_walltime=fr.io.resubmit()),
)
# chunk boundaries = union of trigger times; the compiled step is IO-free;
# Ctrl-C completes the chunk (zero steps lost); NaN -> RunResult(NAN_ABORT)
# with first-failure iteration; resubmitted job: same script, fingerprint
# check, leaves overwritten bitwise, writers truncate_after(t_snap).
```

## 7.8 Parameter sweeps (both idioms)

```python
grid = fr.grid.cartesian.Grid(...)                    # ONE grid object (verify path)

for f0 in f0_values:                                  # idiom A: re-assemble
    m = nh.Model(grid=grid, coriolis=fr.modules.FPlaneCoriolis(f0=f0), ...)
    m.set_fields(b=ic); m.run(steps=10_000)           # compiles once, first iteration

m = nh.Model(grid=grid, ...)                          # idiom B: leaf-only
for f0 in f0_values:
    m.update_parameters({fr.params.CORIOLIS_F0: f0})  # re-materializes f_coriolis,
    m.reset(); m.set_fields(b=ic)                     #   re-ramps warm-up (rewarm=True)
    m.run(steps=10_000)
```

## 7.9 The state-transform workflow (D5)

```python
vortical = nh.transforms.VorticalProjection(model)    # Tier 1; from_model validation
average  = nh.transforms.TimeAverage(model, n_ave=4,  # Tier 2; owns a variant twin
    filter=fr.terms.linear & ~fr.terms.owned_by(fr.closures.ClosureBase))

state_ini      = nh.initial_conditions.jet(model)     # builds on grid.create_field
state_vortical = vortical(state_ini)                  # dye -> 0 (rest="zero")
state_residual = state_ini - state_vortical           # carries the full tracer
model.set_state(average(state_ini))

# optimal balance, composed:
ob = nh.transforms.OptimalBalance(model, ramp_period=T, max_it=3, tol=1e-9,
        backward_filter=~fr.terms.owned_by(fr.closures.ClosureBase)
                        & ~fr.terms.implicit)         # backward diffusion is ill-posed
z_bal, info = ob.call_with_info(state_ini)            # info.errors, .stopped_by
ob.ramp_cycle                                         # forward @ vortical @ backward — public

# IC ensembles through Tier 1:
batch = jax.vmap(lambda k: nh.ics.random_field(grid, k))(jax.random.split(key, 64))
balanced = jax.jit(jax.vmap(vortical))(batch)

# restricted tendencies (budget analysis; the variant/tendency surface):
adv_budget = model.tendency(state, filter=fr.terms.named("CenteredAdvection/momentum"))
```

## 7.10 The hydrostatic step (design-freeze sketch, ROADMAP 3.1)

```python
model = fr.Model(grid=grid,
    modules=(hs.DynamicalCore(...),           # u, v; DIAGNOSE stage: p_hyd = ∫ b dz
             fr.modules.BetaPlaneCoriolis(f0=..., beta=...),
             hs.ConstantStratification(...),
             hs.CenteredAdvection(),
             fr.closures.VerticalMixing(..., treatment=fr.IMPLICIT),
             hs.SplitExplicitFreeSurface(substeps=32)),
             # declares eta, U, V (PROGNOSTIC, 2D, no Velocity role); owns
             #   ADVANCE({eta,U,V}): lax.scan subcycle, filtered averages,
             #     slow forcing from ctx per-treatment sums
             #   CONSTRAINT({u,v}): depth-mean correction (Gauss-Seidel read)
    time_stepper=fr.time_steppers.CNAB2(dt=600.0))
# schedule: SELF_UPDATE -> DIAGNOSE(p_hyd) -> terms -> ADVANCE(CNAB2 + solve)
#   -> ADVANCE(barotropic) -> CONSTRAINT(correction) -> [S5 NaN] -> DIAGNOSTIC
```

## 7.11 Standalone linear theory (no model)

```python
em = nh.eigenmodes.Eigenmodes(grid, f0=1.0, n2=1.0, dsqr=0.2**2, discrete=True)
om = em.omega(s=1)                                   # a Symbol; dispersion plots
P  = nh.transforms.VorticalProjection(em)            # explicit-params Tier-1 form
# and the discrete-in-time correction, applied recipe-side (D2.4):
om_dt = fr.time_steppers.AdamBashforth(dt=60.0, order=3) \
          .time_discretization_effect(om_materialized)
```
