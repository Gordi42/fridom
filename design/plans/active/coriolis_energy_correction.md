---
status: implemented
date: 2026-07-12
---

## Outcome (2026-07-13, `feat/conserving-coriolis`)

Both routes shipped in `src/fridom/shallowwater2/modules/coriolis.py`:
`CoriolisEnergyCorrection` (route A) and the conserving
`NonlinearFPlaneCoriolis` / `NonlinearBetaPlaneCoriolis` /
`NonlinearRotationCoriolis` family (route B). Measured semi-discrete
production of `etot_full` **including** Coriolis (relative, random
state):

| grid | linear Coriolis (today) | route A | route B |
|---|---|---|---|
| periodic | 1.4e-2 | 3.9e-16 | 4.7e-16 |
| channel  | 1.1e-3 | 1.1e-16 | 6.4e-17 |
| sphere   | 1.6e-2 | 5.1e-18 | 4.3e-17 |

A vs B assembled-tendency agreement: 1.5e-16 (all three grids).

Two notes worth keeping:

- **The `L`-honesty gate is consumer-side, not assembly-side.** A
  static check is impossible: the eigenmode / projection / balance
  machinery binds to a model *after* assembly, so nothing in the
  module tuple says the model will ever ask for `L`. Route B's
  modules therefore declare `Module.linear_operator_gap` (a sentence),
  and every consumer of `L` calls
  `fr.model.require_linear_operator(model, consumer=...)` first —
  `fr.model.linearize`, `sw.eigenmodes.from_model`, `sw.eigenbasis`
  (hence every `sw.transforms` projection and optimal balance) —
  raising `LinearOperatorGapError`.
- **Route A's correction is not identically zero as an *operator*
  where `f` varies.** Its *value* on a rest state is exactly 0 (both
  parts are proportional to the velocity), and `L` is bit-for-bit
  unchanged by declaration, as the plan requires. But the conserving
  form averages `f` to the *corner* while the linear module samples it
  at the `u` faces, and those two placements differ at
  `O(dy^2 f'')` on a beta plane / sphere: the correction's
  linearization about the rest state is that (small) residual, which
  `L` does not see. It is a difference of two exactly M-skew
  rotations, hence itself M-skew — it does no work, and `L` remains a
  consistent linearization of the scheme to the scheme's own order.
  On the f-plane the residual is zero to rounding. No way around it:
  the numerator of the conserving f-term is *forced* to be the
  Sadourny corner mass flux by the conservation proof.

# Exactly-conserving Coriolis — the correction term and the
# full nonlinear module

Owner decision, 2026-07-12: **build it, as an optional module.**

## 1. The problem

The shallow-water scheme conserves the thickness-weighted (nonlinear)
energy to machine precision — but only for **gravity + Sadourny
advection**. Measured semi-discrete production rates
([`tests/shallowwater2/test_diagnostics.py`](../../../tests/shallowwater2/test_diagnostics.py)):
1.4e-16 (periodic), 1.3e-16 (channel), 6.3e-16 (sphere).

Include the Coriolis term and the sphere rate jumps to **3.7e-2
relative** — an **O(Ro)** error. The Coriolis module is skew under the
*linearized* norm (that is what makes it exactly energy-neutral in the
linear model, and why the C2 M-skewness gates pass), but not under the
thickness-weighted norm the nonlinear scheme actually conserves.

## 2. Why the obvious fix breaks the eigenmodes

Continuously, the vector-invariant form carries Coriolis and vorticity
in **one** term, `q (h u)^perp` with `q = (zeta + f) / h`. The f-part
collapses exactly: `(f/h) (h v) = f v` — linear.

Discretely it does **not** collapse. Sadourny's energy conservation
requires the *same averaged* thickness in the numerator and the
denominator: the PV sits at corners with `h_bar_corner`, the flux is
`(h v)_bar_corner`, and those averages do not cancel. The
exactly-conserving f-term is therefore

    (f / h_bar) * (h v)_bar        <- a ratio of averages: NONLINEAR in h

Folding that into the PV flux would remove Coriolis from every term
declared `linear=True` — and the eigenmode / state-transform machinery
assembles the linear operator `L` from exactly those declarations
([`specs/model/08_state_transforms.md`](../../specs/model/08_state_transforms.md),
[`classes/declarations.md`](../../specs/model/classes/declarations.md)).
`L` would then describe a **non-rotating** system. That is not a
degradation, it is wrong — and it is precisely why the model splits
Coriolis out as its own linear module today.

## 3. Two routes, both shipped (owner, 2026-07-12)

The exactly-conserving discrete f-term is one expression; there are two
honest ways to put it into a model, and which one is right depends on
whether the user needs the **linear operator** `L`.

| route | terms | `L` (eigenmodes) | when to use |
|---|---|---|---|
| **A. linear + correction** (§4) | `f v` (`linear=True`) **+** correction (`linear=False`, optional) | **unchanged** | the default path; anything using eigenmodes, projections, balance, IMEX-by-linearity |
| **B. full nonlinear Coriolis** (§5) | one term, `linear=False`, replacing the linear module | **loses rotation** | runs that do not need `L` at all — the simpler, cheaper expression |

Both produce the **same tendency** (that is the acceptance test between
them: assembled A and assembled B must agree to rounding), and both
conserve the thickness-weighted energy to machine precision. They differ
only in how the tendency is *declared*, and therefore in what the
term-filtering machinery can still do with the model afterwards.

## 4. Route A: keep the linear term, add the correction

Declare **two** terms:

| term | declaration | expression |
|---|---|---|
| Coriolis (unchanged) | `linear=True` | `f v` |
| **energy correction** (new, optional) | `linear=False` | `(f / h_bar) (h v)_bar  -  f v` |

Their **sum** is the exactly-conserving discrete form; the **linear
operator is bit-for-bit unchanged**, because the correction *vanishes
identically at the rest state* (`h = c^2`, where the two averages
collapse and the bracket is zero). So it contributes nothing to the
linearization, and eigenmodes, projections and IMEX splitting keep
working exactly as they do today.

**Optional by construction** (the CS-D4 discipline applied again):
the correction is its own module. Omitting it from the module list is
the off switch, and its docstring must state plainly what running
without it costs (energy conserved only to O(Ro); the *linear* model is
unaffected either way, since the correction is identically zero there).

Rationale for optional rather than default:

- it adds a nonlinear term to the tendency (compute, and one more term
  for the IMEX/term-filter bookkeeping to carry);
- the error it removes is O(Ro) and lives in the energy *diagnostic*,
  not in stability — at small Rossby number it is small by
  construction;
- the linear model and every eigenmode-based workflow see **no
  difference at all**, so a user doing linear/balance work should not
  pay for it.

## 5. Route B: the full nonlinear Coriolis module

A **standalone module** carrying the exactly-conserving f-term whole:

    du = (f / h_bar) * (h v)_bar        (and its v-counterpart)

declared `linear=False`, and used **instead of** the linear Coriolis
module — not alongside it (assembling both would double-count; the
model must reject that, see the gates).

- **Simpler and cheaper than route A**: one term instead of two, and no
  cancellation between a linear term and a correction that mostly
  undoes it.
- **The cost is `L`.** With Coriolis living only in a `linear=False`
  term, the linear operator has no rotation, so **eigenmodes,
  projections, optimal balance, and IMEX-by-linearity are invalid** —
  not degraded, wrong. The module's docstring must say this in those
  words, and the model should refuse to assemble it together with any
  consumer of `L` if that is detectable at assembly time (investigate:
  the eigenmode/state-transform machinery is model-side and may only
  bind later — if a static check is not possible, the transforms must
  raise when they find no rotation in `L` on a model that declares a
  nonlinear Coriolis).
- **When to use it**: forward runs that never touch the linear
  machinery — the majority of plain nonlinear integrations.

Routes A and B are the *same physics*; B is what you reach for when you
are not paying for `L`.

## 6. Gates

- **The invariant closes — for both routes.** The semi-discrete
  production rate of the *thickness-weighted* energy, including
  Coriolis, drops from ~3.7e-2 to **machine zero** (~1e-16, matching
  the gravity+Sadourny gates) — on flat, channel AND sphere. This is
  the whole point and is the primary test of each route.
- **A and B agree.** The assembled tendency of route A (linear +
  correction) equals that of route B (full nonlinear module) to
  rounding. This is what pins them as the same physics, and it is the
  cheapest way to catch an algebra error in either.
- **B refuses to co-exist with A.** Assembling the nonlinear Coriolis
  module together with the linear one (or with the correction) must
  raise a taught error — that combination double-counts rotation.
- **B is honest about `L`.** A model carrying the nonlinear Coriolis
  must not silently hand a rotation-free linear operator to the
  eigenmode / projection / balance machinery: either refuse at
  assembly, or raise from the transform when it finds no rotation in
  `L`. Test whichever mechanism is chosen.
- **The linear operator is untouched.** Assembling the correction does
  not change `L`: the eigenmode spectrum / projection results are
  **bitwise identical** with and without the module. (The correction is
  zero at the rest state; assert that directly too — the term evaluated
  on a rest state returns exactly 0.)
- **Off switch works.** Omitting the module reproduces today's
  behavior bitwise.
- Works on flat, channel and chart (sphere/torus) grids — the averages
  must carry the metric exactly as the Sadourny module places them, or
  the conservation will not be exact on a chart.
- Mirrored tests; ruff; coverage.

## 5. Sequencing

After the Coriolis default/API change (`refactor/coriolis-default`:
no-rotation default, `NoCoriolis` and `SphericalCoriolis` removed,
`RotationCoriolis` the general module) lands — the two touch the same
files.
