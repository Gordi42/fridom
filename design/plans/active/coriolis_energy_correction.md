---
status: draft
date: 2026-07-12
---

# Coriolis energy correction — an optional nonlinear term

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

## 3. The design: keep the linear term, add the correction

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

## 4. Gates

- **The invariant closes.** With the correction module assembled, the
  semi-discrete production rate of the *thickness-weighted* energy,
  including Coriolis, drops from ~3.7e-2 to **machine zero** (~1e-16,
  matching the gravity+Sadourny gates) — on flat, channel AND sphere.
  This is the whole point of the module and is its primary test.
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
