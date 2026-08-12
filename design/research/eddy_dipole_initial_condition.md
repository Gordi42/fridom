---
status: implemented
date: 2026-08-12
---

# `nh.eddy_dipole`, on the redesigned eddy foundation

A self-advecting pair of counter-rotating Gaussian eddies, placed by
position, by the distance between the two centres and by a compass
heading, with an optional target translation speed and depth-varying
profiles.

This record supersedes the prototype note of the same name written on
`feat/eddy-ic-dipole`, which forked from `dev` before the
`coherent_eddy` redesign landed and was therefore built against the
old curl convention, the old eigenmodes-sourced factory, and a
periodic-only vorticity inversion. The parameterization, the speed law
and the depth rule survived that rebase; the construction, the sign
convention and the balance did not, and are re-derived and re-measured
here. Every number below was measured on the new foundation.

Related records: the single-eddy
[redesign](eddy_initial_condition_redesign.md) (geostrophy,
staggering, sign convention) and the
[streamfunction inversion](eddy_streamfunction_inversion.md)
(the BC-aware elliptic solve).

## 1. Verdict

Four things to take away.

1. **One builder, two factories.** `coherent_eddy` grew per-knob
   callables (`pos_x`, `pos_y`, `width`, `amplitude` each take a float
   **or** a callable of the vertical coordinate) and both factories now
   route through one private `_eddy_state`. A private construction path
   for the dipole was rejected; section 2.
2. **The dipole is exactly balanced, thermal wind included.** The
   prototype sampled an analytic `b = -f d_z psi` at `b`'s own nodes
   and left 1.6e-8 of the energy in wave modes. Sampling `psi` on the
   vertical **faces** instead — the redesign's staggering, generalized
   off the separable case — makes the projected linear tendency
   machine zero (3.6e-16 to 1.7e-15 relative) on **all six**
   topologies, for the barotropic dipole and for both depth-varying
   ones. The buoyancy is no longer an add-on with a `balance=` switch;
   it falls out of the same lattice as the velocities, so the switch is
   gone.
3. **The self-similar depth default is confirmed on the new default
   route.** With `R(z)` varying by a factor of two, top-to-bottom
   travel spread is 11.0 percent for `separation=None` against 43.3
   percent for `match="separation"` and 112.2 percent for a naive fixed
   pair, on the vorticity route; 29.2 / 68.0 / 60.8 percent on the
   streamfunction route. `match="separation"` is worse than doing
   nothing on the streamfunction route, as the prototype found.
4. **A real gap in `spectral_sibling`, worked around here.** A
   depth-varying vorticity dipole needs the elliptic inverse level by
   level. Handing `invert_negative_laplacian` a three-dimensional
   operand **raises** on a rigid lid whose horizontal has a wall
   (2 of 6 topologies), because the passive `Outer(z)` face factor
   cannot take the Dirichlet tag the walled horizontal commits the
   product to. The factory maps the two-dimensional solve over the
   vertical instead, which is both cheaper and unconditional.
   Section 6.

## 2. The builder decision

**Adopted: `coherent_eddy` grows per-knob callables, and both
factories call one private builder.**

### 2.1 What forced the question

The prototype's own finding, which stands: a depth-varying radius
breaks the separable form `psi = G(x, y) F(z)` **before** any tilt of
the dipole axis enters, because `A(z) exp(-r^2 / R(z)^2)` is not a
product of a horizontal shape and a vertical profile for any
non-constant `R(z)`. A `vertical_structure` *multiplier* therefore
cannot express a depth-varying dipole at all. The dipole needs a
construction that accepts z-dependent centre, width and amplitude, and
that does not renormalize.

The redesigned `coherent_eddy` offered a multiplier. So either it grows
per-knob callables, or the dipole keeps a private path.

### 2.2 Why unification was possible

The obstacle was thought to be the staggering. The redesign's exact
balance rests on sampling `F` on the vertical faces and handing the
velocities `I_z^{f->c} F` and the buoyancy `delta_z^{f->c} F`, and
section 3.3 of that record says the implementation "never materializes
a 3-D psi" because `psi` is separable.

That framing was one step short. The identity the balance rests on,

```
I_z^{c->f} delta_z^{f->c} = delta_z^{c->f} I_z^{f->c}
```

is a property of the **staggered lattice**, not of separability. Write
`Phi` for the full three-dimensional `psi` sampled on the corner
crossed with the vertical faces. Then

```
psi_at_centres = I_z^{f->c} Phi ,      b / f = delta_z^{f->c} Phi
```

satisfies `I_z^{c->f} b = delta_z^{c->f} p` exactly, for **any** `Phi`,
separable or not. The separable construction is the special case
`Phi = G ⊗ F`, in which both operators can be pushed onto the
one-dimensional `F`.

So there is one formula with two spellings, and the shared builder
carries both:

- `depth_varying=False` — sample the horizontal shape once on the
  corner with a one-DOF vertical, and let a separable
  `vertical_structure` carry the depth. No three-dimensional array is
  built before the state itself, and the elliptic inversion is a single
  two-dimensional solve. **This is bit-for-bit the code the redesign
  shipped**; the common case did not get slower or harder to read.
- `depth_varying=True` — sample on the corner crossed with the vertical
  faces, and apply the same two operators to that.

Measured agreement where the two overlap: `amplitude=F` reproduces
`vertical_structure=F` to 1.1e-15 (streamfunction) and 2.7e-15 to
7.6e-15 (vorticity, the extra digits being the inversion) on every
topology, in `u`, `v` and `b`. That equivalence is a test, which is
what guards the two spellings from drifting apart.

### 2.3 Why the private path was rejected

The brief's reason is the decisive one and is worth restating: two
divergent construction paths for the same physics is exactly how the
sw2/nh2 vorticity sign discrepancy happened
([redesign](eddy_initial_condition_redesign.md) section 4.1). A
private dipole path would have had to re-derive the curl, the
inversion sign, the geostrophic pressure and the thermal wind, and
nothing but review would have kept the four in step with
`coherent_eddy`'s.

The measurable cost of *not* unifying would have been the prototype's:
its own thermal wind was a second analytic sample at `b`'s nodes,
correct to truncation order and no better, leaving 1.6e-8 of the
energy in wave modes. Reaching machine zero from a private path would
have meant re-deriving the face-lattice argument in a second place.

The unification also buys a property the prototype asserted and could
now be checked cheaply: a dipole **is** two `coherent_eddy` lobes
summed, measured to 3.5e-15 relative (vorticity) and 1.7e-15
(streamfunction).

### 2.4 Rejected variants of the unification

- **Fold `vertical_structure` into the general path and delete the
  separable branch.** One tail instead of three lines of branch.
  Rejected on cost: it would make every baroclinic eddy sample an
  `nx * ny * nz` Gaussian and invert an `nx * ny * nz` operand where
  `nx * ny + nz` and `nx * ny` do today (the redesign measured the
  two-dimensional operand at 184 times cheaper at `512^2 x 64`). The
  separable spelling is not sugar; it is the fast path, and the
  docstring now says so.
- **Have `eddy_dipole` call `coherent_eddy` twice and add.** The
  cleanest thing to write, and it is exactly what the superposition
  test asserts. Rejected as the *implementation* because it doubles the
  sampling and, on the vorticity route, doubles the elliptic solve, for
  a state that is one Gaussian pair. Keeping it as an assertion rather
  than an implementation gets the guarantee without the cost.
- **Per-knob callables on the dipole only.** Would have left
  `coherent_eddy` unable to express a widening eddy at all, for no
  saving: the builder needed the general path either way.

### 2.5 `_SWIRL` is gone

The prototype parameterized the curl convention through a module
constant so a flip would be one edit. That indirection no longer earns
its keep and has been **inlined**.

- The convention is settled and recorded
  ([redesign](eddy_initial_condition_redesign.md) section 4.1). The
  redesign already inlined it in `coherent_eddy` and documented where
  the sign lives; a second, differently-spelled copy of the same
  decision in the same module is the drift risk the unification is
  meant to remove.
- It was not doing the job it claimed. `_SWIRL` carried three signs at
  once — curl, inversion, thermal wind — and the redesign's measured
  finding is that those three are **not** independent: fixing the curl
  fixes `zeta`, which fixes the inversion, which with the momentum
  equations fixes `p = +f psi` and hence `b`. A single scalar that
  multiplies all three is a coincidence of the standard pairing, not a
  parameterization of the choice.
- What replaces it in the dipole is not a constant but a statement of
  physics: the `+A` lobe is counter-clockwise under
  `gauss_field="vorticity"` and clockwise under `"streamfunction"`,
  because `zeta = laplacian_h psi`. That is the same sentence
  `coherent_eddy`'s docstring already carries, so the two factories now
  read their sign from one *explanation* rather than one variable.

## 3. The signature

```python
def eddy_dipole(
    model: Model,
    *,
    pos_x: float = 0.5,          # dipole centre, relative to Lx
    pos_y: float = 0.5,          # dipole centre, relative to Ly
    angle: float = 0.0,          # heading, degrees clockwise from north
    separation: float | Callable | None = None,   # d, relative to Lx
    width: float | Callable = 0.1,                # R, relative to Lx
    amplitude: float | Callable = 1.0,
    speed: float | None = None,
    match: str = "amplitude",    # or "separation"
    gauss_field: str = "vorticity",   # or "streamfunction"
    at_time: float = 0.0,
) -> State
```

Four differences from the prototype's recommendation.

- **`model`, not `source`.** The dipole builds no eigenbasis, exactly
  like the redesigned `coherent_eddy`, and an eigenmodes object gets
  the same taught error.
- **`gauss_field="vorticity"`** by owner decision (2026-08-12), which
  also brings nh2 into agreement with sw2. Section 5 records the
  measurements behind it. `coherent_eddy`'s default moved with it.
- **No `balance=`.** On this foundation the buoyancy is not an optional
  extra sample; it is the vertical difference of the same face-lattice
  `psi` the velocities interpolate. There is no cheaper construction
  with it off, and `balance=False` would ship a state that radiates at
  the percent level. Section 1, point 2.
- **No `vertical_structure=`.** `coherent_eddy` keeps it as the
  separable fast path, but on the dipole it would silently rescale the
  one thing the caller asked for: with `speed=` and
  `match="amplitude"` the amplitude is solved so the *closed-form
  speed* is the target, and a multiplier `F(z)` on top would make the
  realized nominal speed `U F(z)`. The dipole's depth profile is
  `amplitude=`, which is the same multiplier and is the one the speed
  solve reads.

## 4. The physics, re-measured on the new foundation

### 4.1 Heading and sign

**`angle` is a compass bearing in degrees, clockwise from north, and
the dipole travels toward it.** 0 is north (+y), 90 east (+x), 180
south, 270 west. The reasoning for the convention is the prototype's
and is unchanged: the owner anchored at north, north is a natural zero
only for a bearing, and the surrounding vocabulary is geophysical.

> The counter-clockwise eddy (positive relative vorticity) sits on the
> **left of the heading**, the clockwise one on the right. The jet
> between the two cores then points along the heading and carries the
> pair with it.

That statement is about relative vorticity and about a velocity, so it
survived the curl flip untouched. What flipped is which *lobe* is
counter-clockwise: under the standard pairing a streamfunction high is
an anticyclone, so the `+A` lobe sits on the **right** of the heading
under `gauss_field="streamfunction"` and on the left under
`"vorticity"`. The factory places them accordingly, which is why the
heading means the same thing on both routes.

Measured, by least-squares fitting the velocity of the tracked dipole
centre over `t = 0` to `0.4` on a `128^2` unit box, `R = 0.05`,
`speed=0.2`:

| route | heading | measured | error |
| --- | --- | --- | --- |
| vorticity | 0 | 360.00 | -0.00 |
| vorticity | 90 | 90.00 | -0.00 |
| vorticity | 217 | 216.87 | -0.13 |
| vorticity | 305 | 305.28 | +0.28 |
| streamfunction | 0 | 360.00 | -0.00 |
| streamfunction | 90 | 90.00 | 0.00 |
| streamfunction | 217 | 216.26 | -0.74 |
| streamfunction | 305 | 304.32 | -0.68 |

Cardinals are exact by symmetry; the oblique residual is grid
anisotropy and the image lattice, and it is four to six times smaller
on the vorticity route.

### 4.2 The translation speed

Unchanged from the prototype, and re-confirmed. Two Gaussian eddies of
amplitude `+-A`, width `R`, centre separation `d`:

```
streamfunction:  U = (2 A d / R^2) exp(-d^2 / R^2)
                 peak 0.857764 A / R at d = 0.707107 R
vorticity:       U = (A R^2 / 2 d) (1 - exp(-d^2 / R^2))
                 peak 0.319086 A R at d = 1.120906 R
```

In both routes `U_max` equals the peak azimuthal speed of a **single**
eddy exactly, which gives the reachability constraint in one line:

> A dipole can never translate faster than one of its eddies swirls.

**The point-vortex limit is never recovered on the streamfunction
route.** A Gaussian streamfunction eddy has `zeta = laplacian_h psi`,
whose integral is zero, so it carries no net circulation and has no
`1/r` far field at all. This is the single most consequential
difference between the routes and it drives section 5.

Measured against the constructed discrete field at `t = 0`, sampled
bilinearly at the prescribed core, `128^2` unit box, `R = 0.05`,
`A = 1`:

| route | `d/R` | `d/L` | measured | formula | ratio |
| --- | --- | --- | --- | --- | --- |
| vorticity | 1.50 | 0.075 | 0.01462 | 0.01491 | 0.980 |
| vorticity | 2.50 | 0.125 | 0.00949 | 0.00998 | 0.951 |
| vorticity | 3.00 | 0.150 | 0.00774 | 0.00833 | 0.929 |
| vorticity | 3.50 | 0.175 | 0.00644 | 0.00714 | 0.901 |
| streamfunction | 1.00 | 0.050 | 14.589 | 14.715 | 0.991 |
| streamfunction | 1.41 | 0.070 | 7.7437 | 7.7243 | 1.003 |
| streamfunction | 2.00 | 0.100 | 1.4867 | 1.4653 | 1.015 |

The vorticity ratios track the prototype's periodic-image table (which
depends only on `d/L`: 0.950 at 0.125, 0.898 at 0.175) to three digits,
so the closed form is exact to better than half a percent once the
lattice factor is applied. The streamfunction ratios show **no** `d/L`
trend, which is the zero-circulation property showing up as a
measurement: those eddies have no far field for the images to act on.

`speed` is therefore a *nominal* speed, exact at `t = 0` up to the
image factor. Calibrating it to the realized long-time value stays
rejected: that calibration is a two-dimensional table in route and
`d/R` that would silently change meaning whenever the advection scheme
or the resolution changed.

### 4.3 Balance

The headline change. Projected linear tendency relative to the raw
Coriolis tendency, `24^2 x 12`, `f0 = 1.5`, `N^2 = 3`, `delta^2 = 2`:

| case | vorticity | streamfunction |
| --- | --- | --- |
| barotropic | 1.1e-15 to 1.7e-15 | 5.1e-16 to 7.9e-16 |
| `width(z)`, R varying by two | 8.3e-16 to 9.9e-16 | 3.6e-16 to 8.6e-16 |
| `amplitude(z)` | 1.3e-15 to 1.5e-15 | 5.3e-16 to 7.6e-16 |

across all six topologies (periodic, channel-x, box-xy, each with and
without a rigid lid). Discrete three-dimensional divergence is 1.9e-15
to 5.6e-15 relative to `max |u_h|`, and `w` is bit zero. On the
`96^2 x 16` runs of section 4.4 the baroclinic dipole's balance is
9.7e-16 to 5.9e-15 in every construction.

So the dipole is a discrete steady solution of the linear model and
`VorticalProjection` is a no-op on it, tested. The prototype's
`b = 0` imbalance (2.4e-3 to 2.5e-2 wave-energy fraction) and its
analytic-thermal-wind residual (1.6e-8) are both gone, not reduced.

### 4.4 The depth rule

The owner's ruling (self-similar by default, `match="separation"`
supported) re-measured on the new foundation and on the new default
route. `R(z) = 0.05 + 0.0167 cos(2 pi z)`, so `R` varies by a factor of
two; `96^2 x 16`, fully periodic, run to `t = 0.8`, nominal speed 0.2.
The naive control uses the amplitude that would give the target speed
at mid-depth, which is what a caller who has not thought about depth
would pick.

Vorticity route, reference `d/R = 3`:

| construction | travel spread | travel min..max | mean/nominal |
| --- | --- | --- | --- |
| naive (`d`, `A` fixed) | 112.2 % | 0.0713..0.2109 | 0.777 |
| fixed `d`, `match="amplitude"` | 50.9 % | 0.1149..0.1823 | 0.827 |
| **self-similar (default)** | **11.0 %** | 0.1242..0.1388 | 0.833 |
| `match="separation"` | 43.3 % | 0.1035..0.1610 | 0.830 |

Streamfunction route, reference `d/R = 1.4`:

| construction | travel spread | travel min..max | mean/nominal |
| --- | --- | --- | --- |
| naive (`d`, `A` fixed) | 60.8 % | 0.1038..0.2041 | 1.032 |
| fixed `d`, `match="amplitude"` | 424.8 % | 0.0476..0.6467 | 0.881 |
| **self-similar (default)** | **29.2 %** | 0.1444..0.1926 | 1.030 |
| `match="separation"` | 68.0 % | 0.1050..0.2210 | 1.065 |

Reading this.

- **Self-similar wins on both routes**, by a factor of four on the
  vorticity route and two on the streamfunction one. The owner's
  default is confirmed where it now matters most, since the vorticity
  route is the one most callers will hit.
- **`match="separation"` is worse than doing nothing on the
  streamfunction route** (68.0 against 60.8 percent), reproducing the
  prototype's central finding on a different vertical topology and with
  a better-calibrated control. Equalizing the *closed-form* speed
  leaves `d/R` free, and `d/R` is what actually sets the realized
  speed.
- **The streamfunction route's fixed-`d` column is a warning.** At
  fixed `d` and matched amplitude, `d/R` runs from 1.05 to 2.10 over
  the column, across which the streamfunction pair's
  realized-over-nominal ratio spans nearly an order of magnitude; the
  column shears itself apart (424.8 percent, one level travelling
  0.0476 and another 0.6467). The vorticity route's ratio moves only
  from about 0.5 to 0.85 across its usable range, which is why the same
  column costs 50.9 percent there. This is the same mechanism as the
  prototype's finding 2, now visible as the difference between the two
  routes rather than between two constructions.

**Rigid-lid caveat**, unchanged and worth repeating in user-facing
docs: on a walled vertical the vortical mode's buoyancy has sine parity
and must vanish at the lids, so depth profiles need zero slope at top
and bottom. `cos(pi z / H)` satisfies this; a linear `R(z)` does not.

## 5. The `gauss_field` default

**Owner decision, 2026-08-12: `"vorticity"`**, for both `eddy_dipole`
and `coherent_eddy`, which also brings nh2 into agreement with sw2.
The measurements are recorded here because they are what justifies the
decision in the record, and because they tell a reader when to pick the
other route.

### 5.1 Coherence over a translation

`160^2` unit box, `R = 0.05`, nominal speed 0.2, each route at its own
default separation, `t = 0` to 2.4:

| route | `d/R` | travel / L | separation drift | peak `zeta` | realized / nominal |
| --- | --- | --- | --- | --- | --- |
| vorticity | 3.0 | 0.382 | 13.5 % | +1.1 % | 0.794 |
| streamfunction | 1.4 | 0.622 | 60.1 % | +8.4 % | 1.354 |

The two dipoles fail in opposite directions. The vorticity pair spreads
gently (separation 0.1500 -> 0.1714) and holds its peak to one percent.
The streamfunction pair **contracts** (0.0705 -> 0.0504, minus 28
percent) and therefore accelerates: its realized speed is 1.35 times
nominal against the vorticity route's 0.79, most of which is the `d/L`
image factor rather than adjustment. A contracting pair is a
self-reinforcing error, since a smaller `d/R` on that route means a
faster pair.

The prototype's figures showed the streamfunction dipole shedding a
visible trailing wake while the vorticity one stayed compact. That is
measurable as the enstrophy fraction inside a disc of radius `2R`
around each tracked core, which is 0.9997 for an undisturbed Gaussian
pair:

| route | `t = 0` | `t = 1.2` | `t = 2.4` | `d/R` at 2.4 |
| --- | --- | --- | --- | --- |
| vorticity | 0.9997 | 0.9981 | 0.9923 | 3.43 |
| streamfunction | 0.9964 | 0.9310 | 0.9222 | 1.01 |

**Ten times the shed material**: the streamfunction pair leaves 7.4
percent of its enstrophy behind, the vorticity pair 0.7 percent. The
wake is real and it is the streamfunction route's.

### 5.2 The wall

`128^2` closed box (walled x and y), dipole launched due north from
`pos_y = 0.3`, otherwise identical. Gap is the distance from the dipole
centre to the wall, in core radii; both trajectories stay exactly on
`x = 0.5` throughout, which is the symmetry check on the head-on
geometry.

| route | `t` | gap / R | `d / R` | speed / nominal |
| --- | --- | --- | --- | --- |
| vorticity | 1.0 | 10.85 | 3.15 | 0.94 |
| vorticity | 1.9 | 7.97 | 3.16 | 0.65 |
| vorticity | 2.9 | 5.01 | 3.17 | 0.69 |
| vorticity | 3.8 | 2.44 | 3.80 | 0.59 |
| vorticity | 4.8 | 1.68 | 7.26 | 0.04 |
| streamfunction | 1.0 | 9.89 | 0.97 | 1.42 |
| streamfunction | 1.9 | 4.49 | 0.98 | 1.42 |
| streamfunction | 2.9 | 0.75 | 3.00 | 0.02 |
| streamfunction | 3.8 | 0.78 | 12.21 | -0.04 |
| streamfunction | 4.8 | 1.36 | 18.50 | -0.66 |

Read the `d/R` column. A dipole approaching a wall should splay: the
wall is an image pair, and the image pushes the two cores apart. The
vorticity dipole starts splaying at a gap of about **2.5 R** and is
already at 3.8 R separation by 2.44 R of gap, decelerating smoothly
from 0.94 to 0.04 of nominal. The streamfunction dipole holds a
constant 0.98 R separation and a constant 1.42 of nominal speed until a
gap of **0.75 R** — that is, until its cores physically touch the wall
— and then splays violently (3.0 to 12.2 to 18.5 R) and reverses.

Closest approach: 1.68 R for the vorticity dipole, 0.47 R for the
streamfunction one.

That is the `1/r` far field doing exactly what it is claimed to do. A
wall is an image dipole and the image induces `Gamma / (2 pi s)` at
distance `s`, which is identically zero on a route whose eddies carry
`Gamma = 0`. A streamfunction dipole does not feel a wall; it collides
with one.

### 5.3 When to pick `"streamfunction"`

Two cases, both real.

- **When the peak vorticity has to be exact on a fully periodic
  horizontal.** The vorticity route's zero-mean gauge subtracts the
  bump's own area fraction `pi sigma_r^2` — 4.5 percent of the peak at
  `width=0.12`. The streamfunction route has no gauge at all.
- **When the eddies must not interact at range.** The exponentially
  local field is a feature if the experiment is about one dipole and
  the domain is small: the streamfunction route ignores the periodic
  images (measured: no `d/L` trend at all in section 4.2), where the
  vorticity route loses 5 percent of its speed at `d = 0.125 L`.

Against that, the streamfunction route's usable band is narrow
(`d/R` roughly 1.2 to 1.6), it drifts in separation by tens of percent
inside it, and depth-varying columns at fixed `d` are not viable there
at all (section 4.4).

## 6. A gap in `spectral_sibling`

Found while implementing the general path; not fixed here, worked
around, and worth its own follow-up.

A depth-varying vorticity dipole prescribes `zeta(x, y, z)` on the
horizontal corner crossed with the vertical **faces**, and needs
`-laplacian_h^{-1}` applied level by level. The obvious spelling is to
hand `invert_negative_laplacian` the three-dimensional operand and let
the vertical ride along as a passive axis, which is what its docstring
invites ("the remaining axes ride along as passive broadcast
dimensions").

That works on four of the six topologies and **raises** on the other
two:

| topology | passive vertical factor | sibling tags it | result |
| --- | --- | --- | --- |
| periodic | `Right(z)` | (periodic, no tag) | ok |
| channel-x | `Right(z)` | (periodic, no tag) | ok |
| box-xy | `Right(z)` | (periodic, no tag) | ok |
| periodic + lid | `Outer(z)` | Neumann | ok |
| channel-x + lid | `Outer(z)` | **Dirichlet** | raises |
| box-xy + lid | `Outer(z)` | **Dirichlet** | raises |

The mechanism: `spectral_sibling` keeps one trig family across all
bounded axes (a per-family transform instance is grid-bound, so a mixed
`Sine(x) x Cosine(z)` product is not resolvable — verified, it raises
`no DST signature on Outer(z, bc=(NEUMANN, NEUMANN))`). A walled
horizontal axis commits that family to Dirichlet. The passive vertical
face factor is then tagged Dirichlet, and

```
SpaceMismatchError: retag changes BC structure only: at 'z' the
factors Outer(z) and Outer(z, bc=(DIRICHLET, DIRICHLET)) differ
beyond their BC tags
```

because `Outer` carries `n + 1` nodes and its Dirichlet reading carries
`n - 1`. There is no Dirichlet trig origin on the outer node set at
all, so no tag choice fixes it. (A cell-centred passive vertical is
shape-safe under both families — `CellAvg(z, bc=DIRICHLET)` resolves —
which is why the pressure solver and the separable eddy never met
this.)

**Worked around** by `_invert_horizontal`, which maps the
two-dimensional solve over the vertical with `jax.vmap` instead of
handing the transform a three-dimensional operand. Verified bitwise
identical to a per-level Python loop on all six topologies. The
workaround is not a consolation prize: it is also what the redesign's
cost argument asks for (a three-dimensional operand pays a vertical
transform pair the horizontal Laplacian symbol never reads, measured at
184 times the two-dimensional cost at `512^2 x 64`), and the existing
test that pins the two-dimensional operand now covers both depth paths.

Not a distributed-mode regression: the vorticity branch already refuses
a grid that shards a horizontal transform axis (`Fourier.forward cannot
run on this grid that shards the transform axis`), verified under
forced-4 devices on the *unmodified* code, so mapping adds no new
restriction.

**Follow-up for whoever owns the elliptic layer.** Either
`invert_negative_laplacian` should map its passive bounded axes itself
(so every caller inherits the fix), or `spectral_sibling` should refuse
loudly with a taught error naming the mapping workaround, instead of
producing an illegal retag whose message is about BC structure and
mentions neither the family commitment nor the caller's options.

## 7. Errors found in the earlier records

Three were already known and are **already corrected** in
`eddy_streamfunction_inversion.md` on this branch (the sign section
carries a "Superseded 2026-08-12" banner naming the required minus
sign, the gauge table carries a "re-measured 2026-08-12" note giving
`pi w^2` = 3.1416e-2 at `w = 0.1` and 4.5239e-2 at `w = 0.12`, and the
divergence section carries "Read that as luck, not as structure"). A
fourth was not:

- **`eddy_streamfunction_inversion.md`, the summary bullet list**
  still asserted "Divergence of the resulting velocities is exactly
  zero on a walled grid and 2e-15 on a periodic one" — the same
  over-claim the body already retracts eighty lines later. A reader
  who stops at the summary, which is what a summary is for, inherits
  the retracted claim. **Fixed in this change**, with a pointer to the
  body.

One error in the **prototype** worth recording, since it is the kind
that survives a rebase:

- The prototype's separability finding says a depth-varying radius
  makes a separable *vertical structure* insufficient, and concludes
  that the single-eddy builder must therefore accept "arrays
  broadcastable over the vertical, or a shape callable of all three
  coordinates". True as far as it goes, but it framed the requirement
  as being about the **interface**, and section 2.2 above shows the
  interesting part is about the **staggering**: the naive reading — let
  the caller supply a three-dimensional shape and sample each field at
  its own nodes — is exactly the construction the redesign measured at
  4.6e-3 balance residual and rejected. The requirement on the builder
  is not "accept a 3-D shape"; it is "accept a 3-D shape *on the
  vertical face lattice*". A dipole built to the prototype's stated
  requirement would have been unbalanced.

## 8. Rejected alternatives (carried over, still rejected)

- **`U = Gamma / (2 pi d)` alone.** Wrong by construction on the
  streamfunction route, where `Gamma = 0`.
- **The centroid formula as the API's speed**, `U = Gamma (1 -
  exp(-d^2 / 2 R^2)) / (2 pi d)`, peaking at `1.585201 R`. Exact for a
  different measurable, agrees with the peak formula to 5 percent
  beyond `d = 2.5 R`, and has no streamfunction counterpart since the
  weight there integrates to zero. Recorded for anyone tracking
  centroids.
- **Lamb-Oseen `psi` in closed form**, which would give the vorticity
  route with no transform at all. Still rejected: it ignores the
  periodic images, about `4 (d/L)^2` in spurious velocity at the seam.
  Worth revisiting for walled domains, where images are the wrong model
  anyway — and now more attractive than before, since section 6 shows
  the transform route has a topology-dependent seam of its own.
- **The Lamb-Chaplygin dipole**, an exactly steadily translating
  solution with no adjustment transient. Rejected because it has no
  separation parameter and a separation parameter is what was asked
  for. Worth offering later as a separate `shape=`.
- **Calibrating `speed` to the realized long-time speed.** A table
  that would silently rot.
- **Counter-clockwise-from-east angles**, and **silently clamping an
  unreachable target speed**. Both rejected in the prototype for
  reasons that did not change.

## 9. Reproducing

Throwaway scripts; every load-bearing number is inlined above. The
measurements come from: construction-only probes of the balance,
divergence and superposition; bilinear sampling of the discrete
velocity at the prescribed cores for the speed table; sub-cell
parabolic tracking of the `rel_vort_z` extrema at the cell centres for
every direction, coherence and travel measurement; and the linear
model's own `tendency(state, constraints=True)` against
`constraints=False` for the balance residual. Advection was
second-order centred with no closure, so nothing in the tables is a
dissipation artefact.
