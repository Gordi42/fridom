# A0 — advection at a sloping wall: reproduction and root cause

Scripts backing defect **A0** of
[`../../example_authoring_defects.md`](../../example_authoring_defects.md).
Run them from this directory with the repo `src/` and this directory on
`PYTHONPATH`.

| script | what it shows |
| --- | --- |
| `clean_repro.py` | the blow-up: uniform `u = 0.4` on `zp = z H(x)`, `H = 1 + 0.2 sin x`, `n2 = 0`, no friction |
| `spectrum.py` | ARPACK on the `jax.jvp`-linearized one-step map: the leading eigenvalue, its `dt`-independence, the mode's z-profile and x-spectrum |
| `wall_consistency.py` | the mechanism: row-wise consistency against a manufactured flow whose contravariant column flux vanishes identically |
| `order_probe.py` | row-wise convergence order, admissible (`Omega == 0`) vs. lid-penetrating states |
| `stability_run.py` | forward integration after the fix (walled, periodic column, flat, 40% slope) |

## Reproduction

The notes' grid is `tests/model/modules/test_advection_mapped.py`'s, whose
`z` factor is **walled** (`IntervalMesh(n, (0, 1), periodic=False)`). With
a periodic column the blow-up still happens but peaks at `iz = 0`; with
the walled column it peaks at the sloping lid (`iz = n - 1`), reproducing
the notes exactly:

```
mapped walled n=32 dt=0.02   step 15: |u|max=2.96e1 iz=31; non-finite at step 23
mapped walled n=32 dt=0.005  step 60: |u|max=7.58e40 iz=31
mapped walled n=64 dt=0.01   step 15: |u|max=1.33e2 iz=63; non-finite at step 22
```

`spectrum.py` confirms the reported signature on the linearized map: the
growth rate `log|lambda|/dt` is `dt`-independent to three digits (0.1665
vs 0.1666 at n=8; 0.9102 vs 0.9144 at n=16), grows with resolution, the
mode's x-spectrum decays by six decades before Nyquist (smooth in x), and
the flat grid is neutral (`|lambda| = 1.000052`). **`spectrum.txt` was
recorded on the pre-fix code with a PERIODIC mapped column** (before the
walled default was restored), so its mode peaks at `iz = 0` rather than
at the lid; the `dt`-independence and the smooth-in-x conclusions are
unaffected. Re-running `spectrum.py` today exercises the walled column
and the fixed code.

## Root cause

The wall-normal velocity of a terrain-following column is the
**contravariant** column flux `Omega = w - Z_x u` (`Z_x = dzp/dx`), not
the Cartesian `w`: at a sloping wall `w = Z_x u != 0`. `w` lives on
`Inner(z)` with a Dirichlet tag, so its wall face is a structural zero —
correct for a flat wall, wrong for a sloping one.

The nodal mapped divergence
(`_FluxFormAdvection._flux_divergence`) spelled the coupled-axis term as
the **product rule** `-(Z_i/J) I_b(D_b F_i)`. Its boundary value is an
interpolation of an interior column difference, so it cannot cancel the
`axis == base` term's structural wall zero, and the pair leaves the wall
flux `Z_i F_i|_wall` unbalanced — an `O(1/h)` term in the
boundary-adjacent row.

`wall_consistency.py` measures it against a manufactured flow whose
`Omega` vanishes identically (`psi = sin(pi z)`, both walls exact
streamlines, `b = zp`, tendency `-w`):

```
                   iz=0        interior      iz=n-1
n=16            5.90e-03      3.46e-02      1.03e+01
n=32            7.68e-04      8.82e-03      2.04e+01
n=64            9.72e-05      2.21e-03      4.10e+01
```

The interior converges at 2nd order, the flat bottom (`Z_x = 0` there) at
3rd, and the sloping lid **diverges linearly in `n`**. The measured values
match the predicted missing wall flux `n * pi * max|H'/H| = 0.641 n`
(10.26, 20.51, 41.02) to better than 1%. Because the advected quantity is
the momentum itself, the term is a linear feedback of rate
`~ Z_x U / (J dz)` over the half of the domain where the slope has the
destabilising sign: `dt`-independent, `∝ U/dz`, `∝` slope, smooth in x,
zero on a flat grid — every symptom in the notes.

## Fix

Give the nodal path the **J-weighted flux form** the FV path
(`_mapped_fv_divergence`) and the mapped pressure operator already use:

```
(1/J) [ D_i(J F_i) - D_b(Z_i I_b(F_i)) ]
```

Both column terms now close on the same structural wall zero, so the
column flux the pair imposes at the wall is `(Omega q)|_wall = 0` —
impermeability, exactly, for any state. After the fix
`wall_consistency.py` gives 2.83e-1, 1.49e-1, 7.58e-2 at the lid (1st
order) with the interior unchanged at 2nd, and `stability_run.py` holds
600 steps at every configuration, relaxing to the steady potential flow
over the bump (`|u|max -> 0.506`, the continuity speed-up `0.4/0.8`).

The price is formal: the lid row drops from 2nd to 1st order for
admissible states, the standard cost of a structurally exact wall
closure and the same closure the FV divergence already carries.
`order_probe.py` records both regimes.

## What is left (`steep_probe.py`)

A **40% bump** (`H = 1 + 0.4 sin x`) still fails after the fix — but as
an ordinary time-step limit, not as A0's spatial instability. At n=32
the two time steps disagree in sign, at matched physical time:

```
             t=1     t=2     t=3     t=4     t=5     t=6     t=7
dt=0.02     0.665   0.677   0.715   0.803   0.918   0.942   non-finite
dt=0.005    0.644   0.640   0.636   0.631   0.631   0.635   0.632
```

`dt=0.02` grows monotonically to a blow-up; `dt=0.005` decays
monotonically toward the steady speed-up. A0's signature was the
opposite — the growth rate was `dt`-independent to three digits. The
residual is therefore a CFL constraint that tightens with slope (the
40% bump squeezes the channel to `H = 0.6`, accelerating the flow and
halving the physical `dz` at the crest), not a leftover of the
inconsistency. It has **not** been characterized further.

## Not covered: the immersed half

`immersed_probe.py` targets the other geometry A0 named. The fix here
touches only the `CoordinateMapping` column path; an immersed grid
carries no mapping and rides the open-area / volume fractions instead.
The notes' own lead there — a uniform `u = 1` projecting to
`|u|max = 5.70` in cut cells against 1.64 on the mapped grid — points at
an unguarded small-cell (sliver) problem, whose standard remedies (cell
merging, flux redistribution, h-boxes) are a redesign of the cut-cell FV
stack rather than a contained change.

Recorded runs are kept beside the scripts as `.txt` (`spectrum.txt`,
`stability_run.txt`, `steep_probe.txt`); `steep_probe.txt` is truncated
after the two n=32 cases, which are the ones the conclusion rests on.

## Adjudication: why the nodal path had to adopt the flux form (2026-08-14)

Giving the nodal path the J-weighted flux form made its mapped tendency
coincide with the FV one to round-off (6e-17 relative), which broke
`test_fv_mapped_tracer_conserves_and_nodal_does_not` — a test asserting
that terrain conservation is an FV-only property. `projection_compatibility.py`
(output in `projection_compatibility.txt`) settles whether the two
spellings were ever interchangeable. They were not.

A tracer's flux divergence has to be the **same discrete operator the
pressure projection drives to zero**, or a discretely non-divergent
velocity injects a spurious source into every constant field.
`tau(b == 1)` is exactly `-Div_adv(v)`, so it compares directly against
`MappedPressureSolver.divergence`:

| measurement | product rule (pre-fix) | flux form (post-fix) |
|---|---|---|
| `\|Div_adv - Div_proj\|` interior, n=16/32/64 | 1.20e-2, 3.09e-3, 7.95e-4 (`O(h^2)`) | 4.0e-15, 6.7e-15, 1.7e-14 |
| same, sloping-lid row | 10.4, 21.5, 43.4 (`O(1/h)`, **diverging**) | 1.3e-14, 7.6e-14, 2.5e-13 |
| `\|tau(b == 1)\|`, projected velocity, n=16/32 | **13.0, 30.8** | 1.2e-7, 3.7e-7 (the CG residual) |
| `\|tau(b == 1)\|`, uniform `u`, interior | 1.0e-16 | 5.1e-16 |

So the product-rule spelling was not a second consistent discretization
of the same operator; it was a *different* operator from the model's own
continuity — `O(h^2)` different in the interior and inconsistent at the
wall — and the constant-tracer source it injected under the projection
grew with resolution.

The last row is what the change did **not** cost: free-stream
preservation for a constant flux stays exact. The coupled-axis term of
the flux form is `(q/J)[D_i(J) - D_b(Z_i)]`, zero only under the discrete
metric identity, and the identity holds here because `grid.metric`
derives a parameter field's slope through the registry `diff` rows rather
than analytically. (Only the parameter-declared form
`maps={"zp": lambda z, H: z * H}, params={"H": ...}` produces a mapped
column at all — an analytic `maps={"zp": lambda x, z: ...}` yields empty
`column_corrections` and never reaches this path.)

The sloping-lid entry of the last row is deliberately *not* small: with
`w = 0` a uniform `u` drives `Omega = -Z_x u != 0` through the rigid lid,
and `Div_proj` reports the same 1.22, 2.53, 5.11 there. That is also why
`test_mapped_transport_converges_at_second_order` now measures the
interior rows: its manufactured state is that same inadmissible one, and
before the fix its full-domain assertion passed **because** the scheme had
no wall closure (top-row error 1.43e-2, 3.75e-3, 9.54e-4 against a
reference that flows through the lid). The interior numbers are unchanged
by the fix to three digits (1.3362e-2 -> 1.3426e-2, 3.6337e-3 ->
3.6383e-3, 9.3861e-4 -> 9.3891e-4).

Conclusion: the collapse onto the FV numbers is the *design* — "at 2nd
order the FV and nodal C-grid stencils are the same numbers ... the
switch is, numerically, a retag"
([`fv_nonhydro_scoping.md`](../../../plans/active/fv_nonhydro_scoping.md)
§1, extended to mapped grids in §13) — and the mapped tracer divergence
was the one place that property failed. Recorded as §13 addendum 3.
