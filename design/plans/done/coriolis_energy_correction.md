---
status: done
date: 2026-07-13
---

# Exactly-conserving Coriolis — the correction term and the full nonlinear module

> **LANDED 2026-07-13** (merge `fc0b61c8`, branch `feat/conserving-coriolis`;
> commits `f3ca888c` model layer, `f70fad97` shallowwater2).
> Both routes shipped in **shallowwater2**. Nothing remains open.

## 1. The problem it solved

The shallow-water scheme conserves the thickness-weighted (nonlinear)
energy `sw.diagnostics.etot_full` to machine precision — but only for
**gravity + Sadourny advection**. Adding the *linear* Coriolis module
broke that: the module is exactly skew under the **linearized** metric
`M` (which is what makes it energy-neutral in the linear model and why
the C2 M-skewness gates pass), but not under the thickness-weighted norm
the nonlinear scheme actually conserves. Measured production of
`etot_full` was **O(Ro)**: 1.6e-2 relative on the sphere.

## 2. The conserving discrete term

Continuously, the vector-invariant form carries Coriolis and vorticity in
**one** term, `q (h u)^perp` with `q = (zeta + f) / h`, and the f-part
collapses exactly: `(f/h) (h v) = f v` — linear.

Discretely it does **not** collapse. Sadourny's energy conservation needs
the *same averaged* thickness in numerator and denominator: the PV sits at
corners with `h_bar_corner`, the flux is the corner mass flux, and the two
averages do not cancel. The exactly-conserving f-term is therefore a
**ratio of averages**, nonlinear in `h`:

    du = +avg_y(F^v q_f),   dv = -avg_x(F^u q_f),   q_f = f_bar_xy / h_bar_xy

with the corner mass fluxes `F^u`, `F^v` interpolated exactly as
`SadournyAdvection` interpolates them. It conserves because the
measure-weighted interpolations are adjoints, so the exchange
`<F^u, F^v Q> - <F^v, F^u Q>` vanishes for *any* corner scalar `Q`;
choosing `Q = f_bar/h_bar` is what makes it *consistent* with `f v`.
Conservation itself is free. On a **chart** grid every average carries the
metric as Sadourny places it (sqrt(g)-weighted corner fluxes, covariant
tendencies raised onto the prognostic contravariant components).

**Why the obvious fix was not acceptable.** Folding the f-term into the PV
flux removes Coriolis from every `linear=True` term — and the eigenmode /
state-transform machinery assembles the linear operator `L` from exactly
those declarations
([`specs/model/08_state_transforms.md`](../../specs/model/08_state_transforms.md)).
`L` would then describe a **non-rotating** system. Not a degradation:
wrong. Hence two routes, both shipped.

## 3. What shipped

Shared expressions live in `src/fridom/model/modules/coriolis.py`
(`linear_rotation`, `chart_rotation` — one owner, so the correction
subtracts exactly what the linear module adds). Everything else is in
`src/fridom/shallowwater2/modules/coriolis.py`, gated by
`tests/shallowwater2/test_coriolis.py` (~29 tests).

| route | terms | `L` (eigenmodes) | when to use |
|---|---|---|---|
| **A. linear + correction** | `f v` (`linear=True`) **+** `CoriolisEnergyCorrection` (`linear=False`, optional) | **bit-for-bit unchanged** | the default path; anything using eigenmodes, projections, balance, IMEX-by-linearity |
| **B. full nonlinear Coriolis** | one `linear=False` term replacing the linear module: `NonlinearFPlaneCoriolis` / `NonlinearBetaPlaneCoriolis` / `NonlinearRotationCoriolis` | **loses rotation** | runs that never touch `L` — the cheaper expression (one term, no cancellation) |

Route A carries the **difference** `(f/h_bar)(h v)_bar - f v`; its sum with
the linear module is the conserving form. It is **optional by
construction** (CS-D4 discipline): omitting the module is the off switch,
and it reproduces today's behavior bitwise. Rationale for optional rather
than default: it adds a nonlinear term to the tendency and to the IMEX /
term-filter bookkeeping; the error it removes is O(Ro) and lives in the
energy *diagnostic*, not in stability; and the linear model and every
eigenmode workflow see no difference at all.

Route B modules are refused next to any linear Coriolis module or the
correction (`check_rotation_modules`, called from `sw.Model` assembly and
at bind) — a double-counted rotation is a taught error.

**Measured semi-discrete production of `etot_full`, including Coriolis**
(relative, random state):

| grid | linear Coriolis | route A | route B |
|---|---|---|---|
| periodic | 1.4e-2 | 3.9e-16 | 4.7e-16 |
| channel  | 1.1e-3 | 1.1e-16 | 6.4e-17 |
| sphere   | 1.6e-2 | 5.1e-18 | 4.3e-17 |

A vs B assembled-tendency agreement: 1.5e-16 on all three grids — the test
that pins them as the same physics.

## 4. The `L`-honesty gate is consumer-side

A static, assembly-time check is **impossible**: the eigenmode /
projection / balance machinery binds to a model *after* assembly, so
nothing in the module tuple says the model will ever ask for `L`.

The mechanism instead: a module may declare `Module.linear_operator_gap`
(a sentence naming the physics that is *not* in its `linear=True` terms),
and every consumer of `L` calls
`fr.model.require_linear_operator(model, consumer=...)` first, raising
`LinearOperatorGapError` (`src/fridom/model/errors.py`) rather than handing
out a knowingly wrong operator. Gated consumers: `fr.model.linearize`,
`sw.eigenmodes.from_model`, `sw.eigenbasis` (hence every `sw.transforms`
projection and optimal balance). Route B's modules declare the gap;
route A's do not.

## 5. The one thing route A does not restore (small, benign)

Route A's correction is exactly zero *in value* on a rest state (both parts
are proportional to the velocity) and `L` is bit-for-bit unchanged by
declaration — the plan's requirement holds. But it is not identically zero
as an **operator** where `f` varies: the conserving form averages `f` to the
*corner* while the linear module samples it at the `u` faces, and those two
placements differ at `O(dy^2 f'')` on a beta plane / sphere. The
correction's linearization about the rest state is that residual, which `L`
does not see.

It is harmless: a difference of two exactly M-skew rotations is itself
M-skew, so it does no work, and `L` remains a consistent linearization of
the scheme to the scheme's own order. On the f-plane it is zero to
rounding. There is no way around it — the numerator of the conserving
f-term is *forced* to be the Sadourny corner mass flux by the conservation
proof.

## 6. Scope

**shallowwater2 only, by design.** The defect is specific to the
thickness-weighted energy of the Sadourny shallow-water scheme. nonhydro2's
energy is the quadratic Boussinesq one, under which the linear Coriolis
module is already exactly skew; there is nothing to correct, and the shared
`fr.model.modules` Coriolis family it uses is unchanged. The prerequisite
API change (`refactor/coriolis-default`, merge `371113dd`: no-rotation
default, `NoCoriolis` / `SphericalCoriolis` removed, `RotationCoriolis` the
general module) landed first, as this plan sequenced it.
