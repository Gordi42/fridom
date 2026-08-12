---
status: proposed
date: 2026-08-12
---

# Streamfunction from vorticity on walled domains

Research report (see [`README.md`](README.md) for status). Question:
how does `nh.initial_conditions.coherent_eddy` recover a
streamfunction from a prescribed relative vorticity on every grid
topology fridom supports, including the horizontally walled channel
and the closed box, with or without a rigid lid? Method: a prototype
inversion plus manufactured-solution, divergence, wall, cost and
autodiff experiments on a worktree branched from `dev`. Every number
below is reproducible from
[`streamfunction.py`](../../src/fridom/model/streamfunction.py) and
[`test_streamfunction.py`](../../tests/model/test_streamfunction.py);
the throwaway sweep scripts were not kept.

## Verdict

**The C-grid already carries the correct wall condition. The
inversion on the vorticity corner needs no new boundary machinery on
a walled horizontal axis, only a tag for the passive vertical, and it
is exact to round-off on all six topologies and both C-grid
families.**

- The wall condition on the streamfunction is **Dirichlet**, and the
  corner space is **already** `Inner(x, bc=(DIRICHLET, DIRICHLET))`
  on every walled horizontal axis. The walled transform (DST-I)
  resolves on the corner space with no retag.
- The recommended route is the grid's own operator symbols,
  `sum_i |D_i|^2` inverted through `Symbol.inverse`. That is the shape
  sw2's `_invert_laplacian` already has, generalized with a BC-aware
  sibling. Both `SpectralSolve` routes the task proposed fail on
  every walled topology, for reasons given below.
- Discrete round-trip error 3.3e-16 to 1.1e-15 at every resolution
  tested, on all six topologies, both families. Divergence of the
  resulting velocities is exactly zero on a walled grid and 2e-15 on
  a periodic one.
- No iterative fallback is needed. Every topology is spectral.

## What the boundary condition is, and why

At a solid wall the normal velocity vanishes. With nonhydro2's curl
`u = d_y psi`, `v = -d_x psi`, the condition at an x-wall is
`u = d_y psi = 0` for all y, i.e. `psi` is **constant along each
wall**. That is a Dirichlet condition on `psi`, one constant per
boundary component.

The constant is the gauge. On a simply connected closed box the whole
boundary is one component, so the single constant is pure gauge and
zero is as good as any value. On a channel walled in x and periodic
in y the boundary has two components, and their difference is
physical: integrating `v = -d_x psi` across the channel gives
`integral v dx = psi(0) - psi(L_x)`, the net meridional transport
through the channel. Setting both constants to zero is the statement
that a coherent eddy carries no net through-flow, which is what an
eddy initial condition wants. A caller who wants a mean transport
should add a background flow, not perturb the gauge.

The tangential condition does not enter. Free slip and no slip
constrain `d_x psi` at an x-wall, a second condition on a
second-order elliptic problem, so imposing both would overdetermine
it. Which one holds is a property of the model's viscous closure, not
of the initial state; an inviscid run imposes neither. So the
free-slip / no-slip question is simply not a question for this
initial condition.

Discretely none of this needs implementing, because the C-grid
already spells it. Probing the declared spaces on a grid with
`periodic=(False, False, True)`:

```
u : Inner(x, bc=(DIRICHLET, DIRICHLET)) x CellAvg(y) x CellAvg(z)
v : CellAvg(x) x Inner(y, bc=(DIRICHLET, DIRICHLET)) x CellAvg(z)
corner (u_x x v_y) :
    Inner(x, bc=(DIRICHLET, DIRICHLET))
  x Inner(y, bc=(DIRICHLET, DIRICHLET)) x CellAvg(z)
```

`Inner` is the n - 1 interior faces: the two wall faces are not
degrees of freedom at all. Dirichlet `Inner` is exactly the DST-I
origin, whose basis `sin(k pi (x - x_min) / L)` vanishes at both
walls by construction. So `psi = 0` at the wall is structural, not
enforced, and `resolve_transform` returns
`Sine(x) x Sine(y) x Fourier(z)` for the corner space directly. The
claim in the task framing that a walled corner streamfunction needs a
new BC tag or a new sibling space is **false for the horizontal
axes**; it is true only for a passive walled vertical.

A second, quieter benefit of the corner space: DST-I carries modes
`k = 1 .. n-1`, and the staggered difference magnitude
`k_hat = 2 sin(k pi dx / (2 L)) / dx` is strictly positive on every
one of them. The walled operator therefore has **no nullspace**, so
the gauge is fixed and the recovery is exact mode by mode. A
cell-centred operand would take DST-II instead, whose top mode
`k = n` the staggered difference annihilates, a genuine nullspace
the pseudo inverse silently drops. Prescribing vorticity on the
corner is the right API for accuracy as well as for physics.

## The sign convention

`nh.State.rel_vort_z` computes `v.diff("x") - u.diff("y").to(...)`.
Both derivatives land on the corner, so the interpolation is the
identity, and with nonhydro2's curl

```
zeta = d_x(-d_x psi) - d_y(d_y psi) = -laplacian_h(psi)
```

so the inversion the caller needs is
`psi = (-laplacian_h)^{-1} zeta`, the inverse of the **negative**
Laplacian, whose symbol `k_hat_x^2 + k_hat_y^2` is positive. That is what
`invert_negative_laplacian` returns, and it is what nonhydro2's
existing periodic branch already does, so nonhydro2's sign is
correct: a positive prescribed Gaussian gives a counterclockwise
eddy whose diagnosed `rel_vort_z` is that positive Gaussian
(measured: prescribed peak +1.0000, diagnosed peak +0.9921 on a
periodic 64^2 grid, the 0.8 percent shortfall being the domain-mean
gauge; +1.0000 to 1e-13 in a channel).

**shallowwater2 has the opposite sign and a resulting sign bug.**
sw2 uses `u = -psi.diff(y)`, `v = +psi.diff(x)`, so
`zeta = +laplacian(psi)` and the correct inversion is
`psi = -(-laplacian)^{-1} zeta`. sw2's `coherent_eddy` calls
`_invert_laplacian` (which computes `(-laplacian)^{-1} zeta`) without
the sign flip. Measured on a periodic 64^2 grid,
`sw.initial_conditions.coherent_eddy(model, gauss_field="vorticity",
amplitude=1.0)` gives a diagnosed `rel_vort` with min -0.9686 and max
+0.0314: the eddy rotates the wrong way, and the +0.0314 is the
mean-gauge compensation. This is independent of the present work and
should be fixed on its own, either by flipping sw2's curl to match
nonhydro2 or by negating the inversion at the sw2 call site. I did
not touch it.

A second sw2 issue found in passing: `coherent_eddy` inverts a
*second*, independently sampled copy of the Gaussian on the cell
centres to build `p = f0 * psi_centre`. On a walled grid that copy
takes the DST-II parity with its annihilated top mode, and it is not
the interpolation of the corner solution, so the pressure is not
exactly geostrophic with the velocities. `p = f0 * psi_corner.to(p_space)`
would be both cheaper and consistent.

## Routes tried

### Recommended: grid symbols on a BC-aware sibling

```python
solve_space = spectral_sibling(space, axes=axes)
operand = field.retag(solve_space)
kit = GridSymbols(grid, {"f": solve_space})
symbol = sum(kit.diff(axis, on="f").magnitude ** 2 for axis in axes)
coeff = kit.forward("f")(operand)
inverse = jnp.broadcast_to(symbol.inverse().data, coeff.data.shape)
solution = kit.backward("f")(
    coeff.with_data(coeff.data * inverse)).real
```

This is sw2's `_invert_laplacian` with three changes: the axes are a
parameter (nonhydro2 inverts x and y of a three-axis grid), the
operand is retagged onto a sibling before the transform, and the
sibling rule is BC-aware. It subsumes sw2's helper exactly, since on
a periodic two-axis grid `spectral_sibling` is the interned identity
and the program is the same, so the two packages can share one
helper. The prototype lives in
[`fridom/model/streamfunction.py`](../../src/fridom/model/streamfunction.py),
next to the other cross-package initial-condition helpers
(`shapes.py`, `eigenstates.py`).

`.magnitude ** 2` is not an approximation of the composed operator.
`Symbol.magnitude` documents `D.magnitude ** 2 == (D.conj() @ D).real`,
and the C-grid backward difference is the adjoint of the forward one
under the Dirichlet closure, so the symbol is the exact eigenvalue of
the second-difference chain the caller applies afterwards. That is
why the round trip is exact rather than second-order accurate.

The sibling rule, worked out from the failures below:

1. A bounded factor that already carries a BC tag is left alone. The
   corner is Dirichlet-`Inner` on walled horizontal axes; retagging
   it Neumann both misstates the physics and resolves nothing,
   because no Neumann `Inner` transform exists.
2. An untagged bounded factor on an *inverted* axis is tagged
   Dirichlet, the same wall condition, now reached for an operand the
   C-grid did not tag (a cell-centred one, whose Dirichlet origin is
   the DST-II).
3. Untagged bounded factors are all given the *same* trig family,
   inherited from whatever the committed factors use, Neumann when
   nothing commits. This is forced by a transform-machinery
   constraint, not by physics, as the failure below shows.
4. Periodic and constant factors pass through, so a fully periodic
   grid takes the no-retag fast path.

### Failure: `SpectralSolve` on an assembled `Div @ Grad`

`SpectralSolve(Laplacian(metric={"z": 0.0}).expand(sibling, grid), ...)`
works on a fully periodic grid and fails on every walled one:

```
channel-x  : SpaceMismatchError: cannot combine spaces (+):
             x: Inner(x) vs Inner(x, bc=(DIRICHLET, DIRICHLET))
box-xy+lid : SpaceMismatchError: Symbol.inverse() needs equal
             (mode_offset, shape) slot layouts
```

Two independent blockers. First, the divergence leg cannot sum the
per-axis gradient outputs, because the gradient of a Dirichlet-tagged
factor emits a BC-free bounded space on its own axis and keeps the
tag on the others. The pressure solver solves the analogous problem
with `_dirichlet_mid`, which declares the mid parity of the
`Div @ Diag @ Grad` chain. That helper is the wrong parity here: the
wall-normal gradient of an *even* (Neumann) pressure is odd, hence
Dirichlet, whereas the wall-normal gradient of an *odd* (Dirichlet)
streamfunction is even, hence Neumann. A `_neumann_mid` twin would be
needed. Second, even with the right mid tag the composed symbol has a
nonzero net mode shift (sine origins carry `mode_offset = 1`, cosine
origins `0`) and `Symbol.inverse` refuses a diagonal whose domain and
codomain slot layouts differ. The magnitude route sidesteps both,
because `|D|^2` collapses to an endo diagonal on the domain tags by
construction.

### Failure: `build_flat_spectral_solve`

```
channel-x : DispatchError: no operator registered for kind 'diff'
            on Inner(x, bc=(NEUMANN, NEUMANN))
```

`_neumann_sibling` retags *every* bounded factor Neumann, including
the already-Dirichlet `Inner` ones, which resolves nothing. Reusing
the pressure builder would require it to skip already-tagged factors.
That guard is a no-op in the pressure path (the pressure space is
always the BC-free cell scalar), so `_neumann_sibling` could be
replaced by `spectral_sibling(space, axes=())` if the owner wants one
sibling helper. I did not make that change; it is a separate,
pressure-affecting edit.

### Blocker found: no mixed `Sine x Cosine` product

The reason the sibling insists on one trig family. Transform rows are
seeded per family with **all** the axes that family's meshes ground
(`_seed_transform_rows` in `grid.py`), so on a grid with walled x and
walled z the sine instance is `Sine(grid, axes=("x", "z"))`. A
`Dirichlet(x) x Neumann(z)` product then asks it for a DST signature
on the Neumann origin:

```
SpaceMismatchError: no DST signature on CellAvg(z, bc=(NEUMANN, NEUMANN)):
sine origins must be Dirichlet-structured at every boundary component
```

The pressure solver never meets this because it tags every bounded
axis Neumann. This inversion is the first caller that wants different
parities on different bounded axes. It gets away with it because the
vertical is passive here. The inverted symbol never reads the
vertical mode index, so any exact transform pair serves, and DST-II
on the vertical is exact for a barotropic field. A future caller that
genuinely needs `DST(x) x DCT(z)` will have to make the trig
transforms axis-restrictable. Recorded as a real limitation, not
worked around beyond this case.

### Not needed: an iterative fallback

`ConjugateGradient` exists under `spatial/operators/krylov.py` and
the pressure module ships a multigrid hierarchy, but no topology in
scope requires them. Every combination of periodic and walled axes
diagonalizes in a Fourier/DST/DCT product. An iterative solver would
be needed only for an immersed or mapped grid, which the coherent
eddy does not serve today.

## Measurements

All numbers are cpu, float64, single device, on a unit cube.

### Manufactured discrete round trip

A mode `psi = sin(k_x x) f(k_y y)` sampled on the corner (choosing
`f = sin` on a walled axis so the mode satisfies the wall condition
exactly), pushed through the same C-grid chain the initial condition
uses (`psi -> u, v -> rel_vort_z`), then inverted. Maximum relative
error over the domain after removing the gauge constant:

| topology       | family | n = 16  | n = 32  | n = 64  |
| -------------- | ------ | ------- | ------- | ------- |
| periodic       | nodal  | 3.3e-16 | 6.7e-16 | 5.6e-16 |
| periodic       | fv     | 3.3e-16 | 6.7e-16 | 5.6e-16 |
| channel-x      | nodal  | 4.4e-16 | 5.6e-16 | 6.7e-16 |
| channel-x      | fv     | 4.4e-16 | 5.6e-16 | 6.7e-16 |
| box-xy         | nodal  | 5.6e-16 | 3.3e-16 | 4.4e-16 |
| box-xy         | fv     | 5.6e-16 | 3.3e-16 | 4.4e-16 |
| periodic + lid | nodal  | 3.3e-16 | 6.7e-16 | 5.6e-16 |
| periodic + lid | fv     | 3.3e-16 | 6.7e-16 | 5.6e-16 |
| channel + lid  | nodal  | 6.7e-16 | 1.1e-15 | 8.9e-16 |
| channel + lid  | fv     | 6.7e-16 | 1.1e-15 | 8.9e-16 |
| box-xy + lid   | nodal  | 1.1e-15 | 6.7e-16 | 1.0e-15 |
| box-xy + lid   | fv     | 1.1e-15 | 6.7e-16 | 1.0e-15 |

The error does not grow with resolution because the inversion has no
truncation error of its own: it inverts the exact eigenvalue of the
discrete operator.

### Prescribed vorticity and divergence

A Gaussian `zeta` of width 0.12 at the domain centre, 64^2 x 64, FV
family. The table compares the diagnosed `rel_vort_z` of the
resulting state against the prescribed field, and the discrete
three-dimensional divergence of the velocities normalized by
`max |u|`.

| topology       | rel err zeta | rel err, mean removed | max div / \|u\| |
| -------------- | ------------ | --------------------- | --------------- |
| periodic       | 3.1e-02      | 2.3e-14               | 1.8e-15         |
| channel-x      | 6.6e-14      | 6.8e-14               | 0.0             |
| box-xy         | 5.4e-14      | 5.5e-14               | 0.0             |
| periodic + lid | 3.1e-02      | 2.3e-14               | 1.8e-15         |
| channel + lid  | 6.2e-14      | 6.4e-14               | 0.0             |
| box-xy + lid   | 6.2e-14      | 6.4e-14               | 0.0             |

The 3.1e-2 on the periodic rows is not an error, it is the gauge:
the Gaussian's domain mean is 3.1e-2 of its peak, and a periodic
domain cannot support a net vorticity, so the recovered eddy carries
the prescribed Gaussian minus its mean. A walled axis removes the
nullspace and reproduces the prescribed field to 1e-13. This is a
user-visible behavioural difference between topologies and belongs
in the `coherent_eddy` docstring.

The divergence is exactly zero, not merely small, on walled grids.
The C-grid curl guarantees it identically, and the walled case has no
half-spectrum round-off from the Hermitian Fourier axis.

### Walls

On a 32^2 channel-x grid with an eddy at `x = 0.15 L_x`:

- `psi` and `u` carry 31 x-degrees of freedom on a 32-cell mesh: the
  two wall faces are not in the space, so the wall-normal velocity is
  structurally absent and the Dirichlet closure supplies zero.
- The divergence in the two wall-adjacent columns is exactly `0.0`.
  Those cells close on `u_wall = 0`, so this is the numerical
  statement that nothing leaks through the wall.

### Truncation near a wall, and the image vortex

An eddy the wall truncates behaves correctly and acquires the
expected image correction. With a Gaussian of width `0.05 L_x`
centred at `x = 0.02 L_x` (so the wall cuts most of it) on a 128^2
grid, the channel reproduces the prescribed vorticity to 1.4e-14 with
exactly zero divergence, while the periodic grid, where the same
blob wraps around, is off by the mean, 5.3e-3. Peak speed drops from
0.0122 (periodic) to 0.0080 (channel): the wall suppresses the
circulation on the truncated side.

The image-vortex signature is quantitative. A compact eddy at
distance `d` from the wall in a channel should self-advect along the
wall at the speed its opposite-signed image induces,
`-Gamma / (4 pi d)`. Measuring the `zeta`-weighted centroid velocity
(which cancels the eddy's own symmetric field) on a 256^2 channel:

| d / L_x | v centroid | -Gamma / (4 pi d) | ratio |
| ------- | ---------- | ----------------- | ----- |
| 0.08    | -1.253e-3  | -1.250e-3         | 1.003 |
| 0.12    | -8.35e-4   | -8.33e-4          | 1.002 |
| 0.16    | -6.22e-4   | -6.25e-4          | 0.994 |
| 0.20    | -4.88e-4   | -5.00e-4          | 0.975 |

The agreement degrades slowly as `d` grows because the far wall and
the periodic images in y start to contribute, which is the physically
correct behaviour, not an error in the inversion.

### Cost

Warm, best of five, cpu float64. Two callable shapes are compared:
the general three-dimensional route (the operand on the full corner
space) and the barotropic route (the operand on a corner space whose
vertical factor is `mesh.constant`, so `resolve_transform` skips the
vertical stage entirely).

| topology     | n   | n_z | 3-D route | barotropic |
| ------------ | --- | --- | --------- | ---------- |
| periodic     | 128 | 1   | 0.77 ms   | 0.56 ms    |
| periodic     | 128 | 64  | 44.1 ms   | 0.48 ms    |
| periodic     | 512 | 1   | 11.0 ms   | 7.1 ms     |
| periodic     | 512 | 64  | 1148 ms   | 9.3 ms     |
| channel-x    | 128 | 1   | 1.14 ms   | 0.93 ms    |
| channel-x    | 512 | 1   | 13.5 ms   | 17.6 ms    |
| channel-x    | 512 | 64  | 1445 ms   | 13.1 ms    |
| box-xy       | 128 | 1   | 1.75 ms   | 1.32 ms    |
| box-xy       | 512 | 1   | 32.4 ms   | 26.4 ms    |
| box-xy       | 512 | 64  | 1214 ms   | 26.9 ms    |
| box-xy + lid | 128 | 1   | 1.95 ms   | 1.20 ms    |
| box-xy + lid | 512 | 1   | 43.8 ms   | 19.1 ms    |
| box-xy + lid | 512 | 64  | 4615 ms   | 25.1 ms    |

The 128^2 and 512^2 numbers asked for are the `n_z = 1` rows: under a
millisecond and about 10 ms periodic, 2 ms and 20 to 44 ms walled.
The DST costs roughly two to four times the FFT, because the kernels
evaluate through a length-2n complex FFT of the odd extension.

The `n_z = 64` column is the interesting one. The three-dimensional
route pays a full transform pair along a vertical the symbol never
reads, and on a walled vertical that pair is a DST-II, so
`box-xy + lid` at 512^2 x 64 costs 4.6 s against 25 ms barotropic, a
factor of 184. **A barotropic eddy should be inverted on a
`mesh.constant` vertical factor and broadcast afterwards.** That
route also needs no vertical tag at all, since `resolve_transform`
skips constant factors, so `spectral_sibling` is the identity on it
for every topology. Verified end to end on `box-xy + lid` at
64^2 x 16: the diagnosed vorticity matches to 1.5e-14, the divergence
is exactly zero, and the recovered `u` has zero vertical variation.

### Differentiability

`jax.grad` of `0.5 * sum(psi^2)` with respect to the amplitude of the
prescribed vorticity, against a central finite difference at
`h = 1e-6`:

| topology       | grad          | rel err vs FD |
| -------------- | ------------- | ------------- |
| periodic       | 4.5107335e-02 | 2.0e-11       |
| channel-x      | 4.1977952e-01 | 3.1e-10       |
| box-xy         | 1.7447767e-01 | 2.4e-09       |
| periodic + lid | 4.5107335e-02 | 2.0e-11       |
| channel + lid  | 4.1977952e-01 | 1.8e-10       |
| box-xy + lid   | 1.7447767e-01 | 2.2e-09       |

Finite and correct everywhere. The residual spread is the finite
difference's own truncation, not the gradient's. This is initial
condition code, so differentiability is not required by the
differentiability policy, but it costs nothing here: the pipeline is
a transform pair around a multiplication by a constant diagonal, and
`Symbol.inverse`'s exact `== 0` structural test keeps the nullspace
regularization out of the differentiated path.

## Consequences for the caller

- **Prescribe the vorticity on the corner space** (`u`'s normal
  factor tensored with `v`'s). That is what `coherent_eddy` already
  samples. A cell-centred prescription also inverts, but on a walled
  axis it takes the DST-II parity, whose top mode the staggered
  difference annihilates, so that mode of the prescribed field is
  silently dropped.
- **The corner space omits the walls.** A Gaussian centred exactly on
  a wall is sampled only at the interior faces; the wall value never
  enters. This is correct, not lossy: the vorticity at a free-slip
  wall of a Dirichlet streamfunction is zero by construction, which
  is also what sw2's `rel_vort` asserts by retagging its output onto
  the Dirichlet corner.
- **The gauge differs by topology.** With at least one walled
  inversion axis the prescribed vorticity is reproduced exactly. On a
  fully periodic horizontal it is reproduced minus its domain mean,
  because a periodic domain admits no net vorticity. Document this on
  `coherent_eddy`.
- **Invert on a constant vertical for a barotropic eddy**, then
  broadcast with `.to(u_space)` after taking the curl. Orders of
  magnitude cheaper on a deep grid, and it needs no vertical BC tag.
- `coherent_eddy` no longer needs `_analytic` or an eigenmodes object
  at all. It needs the grid, the declared `u` and `v` spaces, and the
  axis extents. Dropping `_analytic` removes the channel refusal.

## Open items

- **Mixed trig products.** `Sine(x) x Cosine(z)` is unresolvable
  because a family transform instance is bound over every bounded
  axis of the grid. Worked around here by keeping one family; a
  caller that genuinely needs two will need axis-restrictable trig
  transforms.
- **nonhydro2's `rel_vort_z` is half-tagged on a walled grid.** It
  returns `Inner(x) x Inner(y, bc=(DIRICHLET, DIRICHLET))`: the `x`
  factor loses its tag because `v.diff("x")` emits a BC-free bounded
  output, while `y` keeps `v`'s own tag. sw2's `rel_vort` retags onto
  the Dirichlet corner and nonhydro2's does not. The consequence is
  family-dependent: under `family="fv"` the untagged result still
  differentiates, under `family="nodal"` `state.rel_vort_z.diff("x")`
  raises `DispatchError: no operator registered for kind 'diff' on
  Inner(x)`. The inversion handles the untagged input (its sibling
  rule retags it), but the asymmetry between the two packages looks
  unintended.
- **sw2's vorticity eddy sign** and its independent cell-centred
  pressure inversion, both described above.
- **`_neumann_sibling` could become `spectral_sibling(space, axes=())`**
  if one sibling helper is wanted. The `is_free` guard is a no-op in
  the pressure path.
