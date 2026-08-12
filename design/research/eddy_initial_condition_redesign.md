# Redesigning `nh.initial_conditions.coherent_eddy`

Status: implemented on `feat/eddy-ic-geostrophy` (2026-08-12).
Scope: the API, the geostrophic physics and the staggering. The
walled-horizontal vorticity inversion is a separate work item and is
left behind a marked seam.

## 1. Summary

The eddy is now a separable geostrophic streamfunction

$$\psi(x, y, z) = G(x, y)\,F(z)$$

turned into `u`, `v`, `b` by geostrophic and hydrostatic balance. Three
things changed.

1. **Source.** The factory takes the assembled `Model`, not an
   eigenmodes object. It builds no eigenbasis, so it serves every grid
   topology, the horizontally walled channel included (which the old
   spelling refused, after paying for the channel eigensolve).
2. **Vertical structure.** A caller-supplied `vertical_structure(z)`
   multiplies the streamfunction, and `b` follows from hydrostatic
   balance.
   The default is the constant 1, the barotropic eddy of the old
   factory.
3. **Staggering.** `G` is sampled on the horizontal corner and `F` on
   the vertical **faces**. The velocities take the corner curl of `G`
   times the face-to-centre *interpolant* of `F`, and the buoyancy takes
   the corner-to-centre interpolant of `G` times the face-to-centre
   *difference* of `F`. Those two vertical operators are the adjacent
   pair of one staggered lattice, which makes the state **exactly**
   balanced rather than balanced to truncation order.

The measured consequence is that the spectral projection the old
factory relied on is not merely optional, it is a no-op: the wave-mode
energy of the constructed state is $4\times 10^{-32}$ of the total, and
the model's own projected tendency is machine zero. Removing the
projection costs nothing on an f-plane.

## 2. Recommended signature

```python
def coherent_eddy(
    model: Model,
    *,
    pos_x: float = 0.5,
    pos_y: float = 0.5,
    width: float = 0.1,
    amplitude: float = 1.0,
    gauss_field: str = "streamfunction",
    vertical_structure: Callable[[jax.Array], jax.Array] | None = None,
    at_time: float = 0.0,
) -> State
```

`pos_x`, `pos_y`, `width`, `amplitude` and `gauss_field` keep their old
meaning and defaults. `at_time` freezes a ramped rotation. The two new
facts are the `model` source and `vertical_structure`.

### 2.1 Why `Model` and not `Grid` plus parameters

The factory needs exactly three things: the grid (to sample on), the
component function spaces of `u`, `v`, `w`, `b` (to know the staggering
and whether the assembly even declares a buoyancy), and the constant
Coriolis parameter. A `Grid` supplies the first, nothing else. The
model supplies all three:

- spaces: `model.state[name].function_space.bare`, which is what
  `Eigenmodes.physical_space` was being used as a proxy for;
- rotation: `model.parameters`, read exactly as
  `eigenmodes.eigenbasis` reads it, so the dimensional `coriolis.f0`
  and the nondimensional $\varepsilon/\mathrm{Ro}$ both work
  (`_coriolis_parameter`);
- buoyancy presence: `"b" in model.state`.

Neither $N^2$ nor the aspect ratio $\delta$ is needed. See section 4.

Passing an eigenmodes object now raises a taught `ValueError` naming
the model spelling. The other factories in the module keep their
`Model | Eigenmodes | ChannelEigenmodes` union because they genuinely
synthesize eigenmodes. This one does not, and a union that accepts an
object it never uses would be dishonest.

### 2.2 The vertical structure callable

`vertical_structure(z)` is called with the **physical** vertical
coordinate array, positionally, and its result is not renormalized.

This matches every other coordinate callable in the module and in the
model layer: `wave_package`'s `envelope`, the `jet` profiles,
`MeridionalStratification`'s `n2(y)` and `_sample`'s whole convention
all receive physical coordinates. A nondimensional depth in $[0, 1]$
would be more portable across domains, but it would be the only such
convention in the package and it hides an orientation choice (is 0 the
bottom or the surface?) that the reader cannot see at the call site. A
caller who wants portability writes
`lambda z: f((z - z0) / (z1 - z0))` and keeps the orientation in view.

Coordinate-**named** kwargs, the `pattern_axes` convention of
`wave_package`, buy nothing here: there is exactly one axis, so
positional is unambiguous, and the named form would let a caller write
`lambda x: ...` and silently get the vertical coordinate.

No normalization. $\psi = A\,G\,F$, so the peak streamfunction is
`amplitude` times $\max G$ times $\max F$. A caller who wants
`amplitude` to be the peak normalizes $F$ themselves. Normalizing
inside would be a silent rewrite of the caller's function, and it is
not even well defined for a sign-changing $F$.

The default `None` means the constant 1 and is not just sugar: on that
path `b` is identically zero (bit zero, not roundoff), the Coriolis
parameter is never read, and the factory therefore works on a
beta-plane and on a model with no rotation module at all.

## 3. The staggering decision

### 3.1 The constraint

On the nonhydro C-grid the four fields sit at

| field | x | y | z |
| --- | --- | --- | --- |
| `u` | face | centre | centre |
| `v` | centre | face | centre |
| `w` | centre | centre | face |
| `b`, `p` | centre | centre | centre |

$\psi$ wants to live on the horizontal corner (x face, y face) because
the discrete curl $u = -\delta_y\psi$, $v = \delta_x\psi$ then lands
exactly on the `u` and `v` spaces and is divergence-free to machine
precision. But $b = f\,\partial_z\psi$ needs $\psi$ at the cell centre
horizontally and, since differencing in $z$ moves centre to face and
face to centre, at the **face** vertically. The corner space that
serves the velocities is at the vertical centre. The two demands
conflict.

### 3.2 What is exactly true on the discrete grid

Write $I$ for a two-point staggered interpolation and $\delta$ for the
matching staggered difference, with superscripts naming the direction
(`c` centre, `f` face). The nonhydro momentum equations in the model's
own nondimensional form (section 4) put the balance condition as: the
linear tendency

$$\big(f\,I_x^{c\to f}I_y^{f\to c}v,\;
   -f\,I_x^{f\to c}I_y^{c\to f}u,\;
   I_z^{c\to f}b\,/\,\delta^2\big)$$

must be the discrete gradient $(\delta_x p, \delta_y p, \delta_z p /
\delta^2)$ of some cell-centred $p$, since the CONSTRAINT stage
subtracts exactly that.

Two elementary stencil identities settle it. On any uniform axis,

$$I^{c\to f}\,\delta^{f\to c} = \delta^{c\to f}\,I^{f\to c}$$

(both equal $(g_{j+3/2} - g_{j-1/2}) / 2\Delta$ at the face $j+1/2$),
and interpolation along one axis commutes with differencing along
another by the tensor-product structure. Applying them:

- the geostrophic potential of the horizontal Coriolis tendency is
  exactly $p = f\,I_x^{f\to c}I_y^{f\to c}\psi$, the corner-to-centre
  interpolant of $\psi$, **not** a fresh centre sample of the same
  analytic function;
- and $I_z^{c\to f}b = \delta_z^{c\to f}p$ holds exactly if and only if
  $b$ carries the *same* horizontal interpolant of $\psi$ and the two
  vertical operators are the adjacent staggered pair.

Which fixes the construction: sample $F$ on the vertical faces, hand
the velocities $I_z^{f\to c}F$ and the buoyancy $\delta_z^{f\to c}F$.

### 3.3 The decision

**Sample $G$ once on the horizontal corner and $F$ once on the vertical
faces, and give the velocities the vertical interpolant of $F$ and the
buoyancy the horizontal interpolant of $G$ times the vertical
difference of $F$.** The alternative that reads more natural, sampling
each field at its own nodes so that every value is the exact analytic
one, buys pointwise fidelity and loses exact discrete balance. The
choice here loses $O(\Delta z^2)$ pointwise fidelity in the velocity's
vertical profile and gains a state that is a discrete steady solution
of the linear model to machine precision.

Because $\psi$ is separable the implementation never materializes a
3-D $\psi$: $G$ lives on `corner_h` $\otimes$ `Constant(z)` and $F$ on
`Constant(x,y)` $\otimes$ `face_z`, and the field algebra broadcasts
the products. That is measured bit-equivalent to the 3-D route
($8\times10^{-15}$ on fields of size 4) and makes the vorticity
inversion a purely two-dimensional problem, which is the seam the
walled-domain solver plugs into.

### 3.4 Exact versus truncation-order properties

Holds **exactly** (machine precision, every topology and both C-grid
families):

- the discrete 3-D divergence $\delta_x u + \delta_y v + \delta_z w$ is
  zero, with $w \equiv 0$;
- the horizontal Coriolis tendency is curl-free, so it is exactly the
  discrete gradient of $p = f\,I_h\psi$ (measured $1.3\times10^{-15}$
  relative);
- $I_z^{c\to f}b = \delta_z^{c\to f}p$, hydrostatic balance in the
  discrete form the `w`-equation actually uses;
- consequently the model's projected tendency is zero, i.e. the state
  is in the null space of the linear propagator and therefore exactly
  in the vortical ($\omega = 0$) subspace.

Holds only to **second order**:

- the continuum correspondence of the velocity. The sampled `u` carries
  $\tfrac12(F(z_{j-1/2}) + F(z_{j+1/2}))$ where the analytic eddy has
  $F(z_j)$; measured $2.4\times10^{-2}$, $6.3\times10^{-3}$,
  $1.6\times10^{-3}$, $4.0\times10^{-4}$ at $n_z = 8, 16, 32, 64$ for
  $F = 1 + \tfrac12\cos 2\pi z$, a clean factor 4 per refinement;
- thermal wind in the "interpolate both sides onto one space" form,
  $f\,\delta_z u = -\delta_y b$. It closes only up to the C-grid factor
  $I_z^{c\to f}I_z^{f\to c}$, because `b` sits at the cell centre while
  $\delta_z p$ sits at the `w` face. Measured $1.46\times10^{-1}$,
  $3.81\times10^{-2}$, $9.61\times10^{-3}$, $2.41\times10^{-3}$ at
  $n = 16, 32, 64, 128$ (ratios 3.85, 3.96, 3.99). This is a property
  of the grid, not of the construction. No local $b$ can do better,
  since exactness would need the operator $\delta_z / I_z$, which is
  nonlocal and singular at the vertical Nyquist.

### 3.5 Alternatives rejected

Balance residual below is
$\max|\text{projected tendency}| / \max|\text{Coriolis tendency}|$ on a
$32^2\times16$ periodic grid, $f_0 = 1.5$, $N^2 = 3$, $\delta^2 = 2$,
$F = 1 + \tfrac12\cos 2\pi z$, `width=0.15`.

| construction | residual |
| --- | --- |
| shipped (face-sampled $F$, interpolate for `u`, difference for `b`) | $2.3\times10^{-15}$ |
| sample every field at its own nodes, $F'$ by autodiff | $4.6\times10^{-3}$ |
| as above but `b`'s horizontal from the corner interpolant | $3.3\times10^{-3}$ |
| centre-sampled $\psi$, wide $2\Delta z$ centred vertical difference | $9.8\times10^{-3}$ |

- **Sample twice, the sw2 precedent.** `shallowwater2.coherent_eddy`
  samples the Gaussian on the corner and again on the centre rather
  than interpolating. In two dimensions that costs only the
  $O(\Delta h^2)$ difference between $p$ and $f\,I_h\psi$, and sw2
  carries `p` as a prognostic so the difference is visible but
  harmless. In three dimensions the same choice leaks into `b` and the
  state stops being balanced (row 2 above). Rejected on measurement.
  The horizontal interpolant is the discretely correct object.
- **Analytic $F'$, supplied or by `jax.grad`.** Requires the caller's
  callable to be jax-differentiable (or requires a second callable),
  and is still second-order wrong against the discrete operator (rows
  2 and 3). The face-difference needs neither.
- **Differentiate the centre-vertical $\psi$ and interpolate back**
  (`psi.diff("z").to(b)`). The cheapest thing to write, and the worst
  of the local options: the round trip applies the interpolation twice
  (row 4, residual scaling $3.6\times10^{-2}$, $9.8\times10^{-3}$,
  $2.5\times10^{-3}$ at $n_z = 8, 16, 32$). It also silently imposes
  $\partial_z\psi = 0$ at a rigid lid through the Dirichlet tag on the
  `w` face space, which is wrong for a surface-intensified eddy.
- **An exact nonlocal `b`.** Solving $I_z^{c\to f}b = \delta_z^{c\to
  f}p$ for `b` means inverting $I_z$, which is singular at the vertical
  Nyquist and needs a periodic vertical. Rejected: it would trade a
  second-order truncation for a global, conditionally defined operator.
- **Reading the vertical face positions off the `w` space.** On a
  rigid lid `w` lives on `Inner(z, bc=DIRICHLET)`, which drops the two
  boundary faces; interpolating from it would force $\psi = 0$ at the
  lid. The factory uses the mesh's own face node set instead
  (`mesh.right` when periodic, `mesh.outer` when bounded, the latter
  carrying all $n+1$ faces). The boundary faces never enter the balance
  condition, because the model's own `w` and $\delta_z p$ are defined
  only at the interior faces there, which is why the identity of
  section 3.2 survives the rigid lid (measured $2.6\times10^{-15}$).

## 4. The geostrophic relations, with their factors

Established from the code, not from memory:

- `fridom/model/modules/coriolis.py:348`, the rotation term is
  $\partial_t u = f\,v|_u$, $\partial_t v = -f\,u|_v$, scaled by the
  live $\varepsilon/\mathrm{Ro}$ on the nondimensional variant;
- `fridom/nonhydro2/modules/stratification.py:203`, the buoyancy force
  is $\partial_t w = b\,/\,\delta^2$ with $\delta$ the
  `nonhydro.aspect_ratio`, and the restoring is $\partial_t b = -N^2 w$
  (dimensional) or $-(\varepsilon/\mathrm{Fr})^2 w$;
- `fridom/nonhydro2/modules/core.py:736`, the CONSTRAINT stage
  subtracts $\delta_x\phi$, $\delta_y\phi$ and $\delta_z\phi/\delta^2$
  from `u`, `v`, `w`. There is no pressure-gradient tendency term, and
  the gradient enters through the projection.

So the momentum equations are

$$\partial_t u = f v - \partial_x p, \qquad
  \partial_t v = -f u - \partial_y p, \qquad
  \delta^2\,\partial_t w = b - \partial_z p,$$

with $f$ the effective Coriolis parameter: the dimensional
`coriolis.f0`, or $\varepsilon/\mathrm{Ro}$ read from
`scaling.nonlinearity` and `coriolis.rossby` (identical to the re-key
in `eigenbasis`). Balance therefore gives

$$p = f\,\psi, \qquad
  u = -\partial_y\psi, \qquad
  v = \partial_x\psi, \qquad
  w = 0, \qquad
  b = \partial_z p = f\,\partial_z\psi .$$

Three factors that are easy to get wrong and are **not** there:

- **no $\delta$ in `b`.** The aspect ratio multiplies $\partial_t w$
  and divides both $b$ and $\partial_z p$ in the `w`-equation, so it
  cancels out of hydrostatic balance exactly. Verified: the balance
  residual is machine zero for $\delta^2 = 2$ without any $\delta$ in
  the construction.
- **no $N^2$ or $\mathrm{Fr}$ anywhere.** The stratification sets how
  the eddy *evolves* (the Burger number, the deformation radius) and it
  weights the energy metric, but the diagnosis of `b` from $\psi$ is
  purely hydrostatic. Verified: the same construction is exactly
  balanced at $\mathrm{Fr} = 0.4$ and $N^2 = 3$.
- **no Rossby number beyond $f$ itself.** On a nondimensional assembly
  the whole rotation is scaled by $\varepsilon/\mathrm{Ro}$, so the
  single substitution $f \to \varepsilon/\mathrm{Ro}$ covers it.
  Verified at $\mathrm{Ro} = 0.1$ and $0.5$ under
  `fr.scaling.Advective()`, residuals $4.0\times10^{-15}$ and
  $3.5\times10^{-15}$.

### 4.1 Signs, and a disagreement between the two packages

With $u = -\partial_y\psi$, $v = \partial_x\psi$ the model's own
`rel_vort_z` $= \delta_x v - \delta_y u$ equals $\nabla_h^2\psi$, and
$p = +f\psi$. Plainly:

- a **positive** `amplitude` with `gauss_field="streamfunction"` is a
  pressure high, hence an **anticyclone**: clockwise for $f > 0$,
  negative relative vorticity at the centre. Measured
  $\zeta_{\text{centre}} = -4.41$ for `width=0.15`, `amplitude=1`;
- a **positive** `amplitude` with `gauss_field="vorticity"` prescribes
  positive `rel_vort_z`, hence a **cyclone**, which is a pressure low.

The two branches turn opposite ways for the same sign of `amplitude`.
That is the physics ($\zeta = \nabla_h^2\psi$, so a $\psi$ bump has
negative $\zeta$ at its peak), not a convention that can be chosen
away, and the docstring says so.

#### The convention, stated once

**Adopted: the standard geostrophic pairing.**

$$u = -\partial_y\psi,\quad v = +\partial_x\psi,\quad
  \zeta = \nabla_h^2\psi,\quad p = +f\psi,\quad
  \hat\psi = -\hat\zeta / k_h^2 .$$

The four are not independent. Fixing the curl fixes $\zeta$, fixing
$\zeta$ fixes the inversion sign, and the momentum equations of
section 4 then fix $p = +f\psi$ with no freedom left. Only the curl is
a choice, and the standard one is the one under which a positive
streamfunction is a positive pressure anomaly.

This was cosmetic while the eddy carried only `u` and `v`. It is
load-bearing now, because `b` is derived through $p$: under the other
curl the same $\psi$ would give $b = -f\,\partial_z\psi$, and a
surface-intensified anticyclone would come out cold-core.

#### What each package did

Measured on matched setups, `amplitude=+1`:

| | `rel_vort_z` at the centre | verdict |
| --- | --- | --- |
| sw2 `streamfunction` | $-4.41$, $p = +1.47$ | correct |
| sw2 `vorticity` | $-0.93$ | **wrong sign** |
| nh2 `streamfunction` (old) | $+4.41$ | non-standard (implies $p = -f\psi$) |
| nh2 `vorticity` (old) | $+0.93$ | self-consistent |

nh2 used $u = +\delta_y\psi$, $v = -\delta_x\psi$, for which
$\zeta = -\nabla_h^2\psi$, paired with the inversion
$\hat\psi = +\hat\zeta / k_h^2$, i.e. $(-\nabla_h^2)^{-1}$. The pair is
**internally consistent**, so the old nh2 vorticity branch delivered
exactly the vorticity the caller asked for. What it was not is
standard: the same curl forces $p = -f\psi$, which is why the
streamfunction branch turned a positive $\psi$ into a cyclone.

sw2 has the standard curl, so its $\zeta = +\nabla_h^2\psi$ and its
inversion should be $(+\nabla_h^2)^{-1}$. It calls the same
`_invert_laplacian`, which is $(-\nabla_h^2)^{-1}$, so its vorticity
branch returns minus the vorticity requested. That one is a plain bug,
independently reproduced on a $64^2$ periodic grid, `width=0.12`,
`amplitude=+1`: `rel_vort` runs from $-0.955$ to $+0.045$, right
magnitude, wrong sign, backwards rotation.

#### Consequences and migration

For **nh2**, this change adopts the standard pairing on both branches.

- `gauss_field="streamfunction"` flips rotation sense. A caller
  reproducing an old field passes `amplitude=-1` times the old value,
  which gives the old field bit for bit (the construction is linear in
  `amplitude`).
- `gauss_field="vorticity"` is **unchanged**, because both signs
  flipped together. Old vorticity-branch callers need no edit.
- No existing nh2 test encoded the sign. The pre-change tests asserted
  divergence-freedom and the taught errors only, so nothing in the
  suite had to be relaxed to make room. The new
  `test_eddy_signs` and `test_eddy_vorticity_branch_reproduces_its_gaussian`
  pin the convention going forward, which is what makes a future
  reversal loud instead of silent.
- Reversing the decision is two characters plus one: the signs on the
  `u`/`v` lines of `coherent_eddy` and the sign inside
  `_streamfunction_from_vorticity`. Reversing all three together keeps
  the vorticity branch correct and puts `p` back to $-f\psi$, which
  would then have to be threaded into the buoyancy as well. The seam
  docstring records where the sign lives.

For **sw2**, deliberately untouched here.

- The one-line fix is the sign of `_invert_laplacian`'s output at its
  two call sites in `coherent_eddy` (or negating the helper and
  renaming it, since sw2 has no other caller).
- It breaks **no** existing test. `test_eddy_is_divergence_free_and_balanced`
  is sign-blind (divergence and a residual bound),
  `test_eddy_streamfunction_centers_the_pressure` locates the pressure
  peak with `np.abs(p).argmax()` and checks only that a negative
  amplitude flips the state exactly, which survives any global sign
  choice. So the current behaviour is unpinned in sw2 as well, and a
  fix there should ship with a sign test of its own.
- Until it is fixed, the two packages disagree on what
  `gauss_field="vorticity"` means. That is worth closing.

## 5. Removing the projection

The old factory relied on the caller to project, or on the neighbouring
factories' `geo_proj=True`. The owner asked what removing the
projection costs. Measured, on a $32^2\times16$ periodic grid and on
the rigid-lid twin:

| state | wave-mode energy / total |
| --- | --- |
| shipped construction, barotropic | $0$ (bit zero) |
| shipped construction, baroclinic, periodic | $3.8\times10^{-32}$ |
| shipped construction, baroclinic, rigid lid | $3.5\times10^{-32}$ |
| sample-at-own-nodes construction, baroclinic | $7.5\times10^{-6}$ |

$10^{-32}$ is squared roundoff: the projection is a no-op, and
`VorticalProjection` leaves the state unchanged to $4\times10^{-16}$
relative. Run under the linear model for 40 steps the state does not
move ($1.8\times10^{-16}$ relative drift, $\max|w| = 1.8\times10^{-16}$
where it started at zero). So on an f-plane, removing the projection
costs nothing measurable, and the docstring says so instead of telling
users to project by reflex.

The naive alternative would have radiated: $7.5\times10^{-6}$ of the
energy in wave modes is a $2.7\times10^{-3}$ relative state
perturbation, which is what the projection would have removed. That is
the honest cost of the *other* staggering choice, and the reason the
staggering decision is load-bearing rather than cosmetic.

The docstring names the cases where a caller should still project: a
beta-plane or chart rotation (whose varying $f$ admits no exactly
balanced $\psi$ at all), an immersed or mapped grid, a state a
perturbation has been added to, and a `vertical_structure` so steep it
is not resolved.

Nonlinear runs are a separate matter and are not a defect. With
advection on, the eddy self-advects. For a circular vortex the
nonlinear terms are a radial gradient absorbed by the pressure
(gradient-wind balance), so the residual is genuine ageostrophic
physics that scales with the Rossby number. Measured over 40 steps on
$64^2\times16$: $|w|/|u| = 1.9\times10^{-2}$ at
$\mathrm{Ro} = |\zeta|_{\max}/f = 1.33$ and $6.5\times10^{-3}$ at
$\mathrm{Ro} = 0.44$, energy conserved to $2\times10^{-4}$, everything
finite.

## 6. The vorticity branch and its seam

`_streamfunction_from_vorticity(grid, zeta)` is the only
topology-dependent piece. It inverts the **horizontal** Laplacian,
$\nabla_h^2\psi = \zeta$, on a two-dimensional operand (the corner
field with a one-DOF `Constant` vertical factor), because the vertical
structure multiplies the streamfunction after the inversion. Three
consequences:

- the vertical topology no longer restricts the branch. The old
  three-dimensional transform refused a walled vertical, while the
  two-dimensional one serves the rigid lid exactly like the periodic
  box (tested);
- the operand is $n_x n_y$ values rather than $n_x n_y n_z$. The
  parallel inversion work measured the two-dimensional operand at 184
  times cheaper than the full three-dimensional field at
  $512^2\times 64$, a second reason to keep the vertical structure
  outside the inversion rather than folding it in;
- the general walled-horizontal solver replaces one function body, with
  nothing above or below it to change. Today that body raises a taught
  `ValueError` on a horizontally walled grid, naming the
  `gauss_field="streamfunction"` fallback.

The seam is written against the prototype
`invert_negative_laplacian(grid, field, axes=...)` in
`src/fridom/model/streamfunction.py` (branch `feat/eddy-ic-inversion`),
which solves the positive-definite $-\nabla_h^2$. This factory needs
$+\nabla_h^2$, so the swap is

```python
psi = -invert_negative_laplacian(grid, zeta, axes=(x, y))
```

with the minus sign staying on this side, next to the curl convention
that forces it. The `axes` parameter is redundant for the operand used
here (whose vertical factor is already one-DOF) and harmless. That work
also found that the walled-horizontal case needs no new BC tag: the
C-grid's `Inner(x, bc=(DIRICHLET, DIRICHLET))` is already the DST-I
origin and `resolve_transform` returns the sine product, which is
exactly the wall gauge $\psi = 0$ on a solid boundary. Only a walled
vertical would need a tag, and this construction never transforms the
vertical.

The zero-mean gauge is unchanged and worth stating: on a doubly
periodic domain $\int\zeta = \int\nabla_h^2\psi = 0$, so the recovered
vorticity is the prescribed Gaussian minus its domain average. For
`width=0.1` on a $(2\pi)^2$ box that average is
$\pi\sigma^2 / L_xL_y \approx 0.031$, and the recovered peak is
$0.969$ rather than $1$. Tested.

## 7. Call sites and migration

`grep -rn coherent_eddy` over the repository:

| site | action |
| --- | --- |
| `tests/nonhydro2/test_initial_conditions.py` | rewritten in this change |
| `src/fridom/nonhydro2/__init__.py` | re-export only, name unchanged, no edit |
| `src/fridom/nonhydro2/initial_conditions.py` module docstring | updated |
| `design/plans/active/cutover_parity_plan.md:67` | prose parity list, mentions the name only |
| `src/fridom/shallowwater2/initial_conditions.py` | the sibling factory, untouched, see 4.1 |
| `src/fridom/nonhydro/initial_conditions/coherent_eddy.py` | old stack, untouched |
| `docs/`, `examples/` | no uses |

No reader-facing documentation and no example script calls it, so the
`docs-review` gate does not apply to this change.

The migration is one line at each call site:

```python
# before
em = nh.eigenmodes.from_model(model)
z = nh.coherent_eddy(em, width=0.15)

# after
z = nh.coherent_eddy(model, width=0.15)
```

plus two behaviour notes for anyone with a stored figure or a
regression baseline:

- `gauss_field="streamfunction"` now rotates the other way. Flip the
  sign of `amplitude` to reproduce the old field exactly.
- `gauss_field="vorticity"` is unchanged in sign, and the branch now
  also runs on a walled vertical.

## 8. Open follow-ups

- The walled-horizontal inversion behind the seam (owned elsewhere,
  branch `feat/eddy-ic-inversion`).
- The sw2 vorticity sign (section 4.1). An owner decision because it is
  a behaviour change in a second package, though it breaks no test
  there.
- A baroclinic eddy on a beta-plane is refused with a taught error
  rather than approximated with the centre-latitude $f$. A varying $f$
  admits no exactly balanced $\psi$, and a silent local-$f$ buoyancy
  would look balanced without being so. The barotropic eddy still works
  there, which is the case the old-stack example used.
- A grid whose horizontal is one coupled two-dimensional factor (a
  chart) is not served, since the construction replaces per-axis space
  factors. Same limitation as the old factory.
