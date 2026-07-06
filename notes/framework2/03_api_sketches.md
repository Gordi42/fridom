# Grid abstraction redesign — API sketches

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map. Concepts are in
[`01_concepts.md`](01_concepts.md), rules in
[`02_rules.md`](02_rules.md).

---

## 4. API sketches (non-normative)

### 4.1 Uniform tensor grid, staggered derivative

```python
import jax.numpy as jnp

import fridom.framework2 as fr

# convenience subclass: builds two IntervalMeshes internally. The
# general assembly root is fr.Grid(meshes=...) (sketch 4.4; coordinate
# names live on the meshes); fr.grid.cartesian.Grid is the cartesian
# convenience form.
grid = fr.grid.cartesian.Grid(
    shape=(256, 256), extent=((0, 1), (0, 1)),
    periodic=True, names=("x", "y"),
)
mx, my = grid.factors      # the Mesh factor objects, in names order

# space defaults to the all-Center nodal space: Center(x) ⊗ Center(y)
f = grid.create_field(
    init=lambda x, y: x * x + jnp.sin(2 * jnp.pi * y),
)

print(f.function_space)
# Center(x) ⊗ Center(y)

g = f.diff("x")                 # dispatch: (diff, Center(x)) -> FD(order=2)
print(g.function_space)
# Right(x) ⊗ Center(y)

f + g                           # -> SpaceMismatchError

# .to(target) is the generic per-axis conversion: it dispatches to the
# registered kind (interpolate here; reconstruct/phase-shift elsewhere) and
# accepts a field or a space
h = g.to(f)                     # explicit, dispatch: (interpolate, Right(x))
f + h                           # ok
```

### 4.2 Custom operator, module-local override

```python
weno = fr.operators.WenoReconstruction(order=5)   # free-standing

class MyAdvection(fr.modules.advection.AdvectionBase):
    def __init__(self):
        super().__init__()
        # override the approximate reconstruct step only (CellAvg ->
        # Outer); the exact conservative flux-difference is left
        # untouched (section 3.9). `reconstruct` (average -> point value)
        # is a distinct kind from `interpolate` (nodal -> nodal); `.to` is the
        # generic sugar that dispatches to either.
        self.dispatch["reconstruct"] = weno       # local override
```

### 4.3 Transform round trip with per-origin coefficient spaces

```python
mx, my = grid.factors
u = grid.create_field(mx.right * my.center)
w = grid.create_field(mx.center * my.center)

# transforms are grid-bound operators with forward / backward; the
# origin of each coefficient space follows from the input space
t = fr.operators.Fourier(grid, axes=("x", "y"))
u_hat = t.forward(u)       # -> Fourier(x, origin=Right)  ⊗ Fourier(y, ...)
w_hat = t.forward(w)       # -> Fourier(x, origin=Center) ⊗ Fourier(y, ...)

u_hat + w_hat                          # -> SpaceMismatchError

# .to dispatches ("interpolate", Fourier(origin=Right)) -> exact PhaseShift
u_hat = u_hat.to(w_hat)
u_hat + w_hat                          # ok

u2 = t.backward(u_hat)                 # unambiguous: the coefficient
                                       # space knows its origin
```

### 4.4 Mixed grid: uniform FV x Chebyshev-Galerkin

```python
mx = fr.meshes.IntervalMesh(shape=256, extent=(0, Lx), periodic=True,
                            name="x")
mz = fr.meshes.ChebyshevMesh(shape=64, extent=(-H, 0), name="z")

# fr.Grid is the model-agnostic assembly root: it takes pre-built
# meshes (any mesh type, incl. sphere / unstructured); coordinate
# names are mesh-constructor arguments, the grid only validates
# uniqueness. The cartesian convenience form fr.grid.cartesian.Grid
# (sketch 4.1) is a subclass that builds the uniform IntervalMesh
# factors from shape=/extent=/names=.
grid = fr.Grid(
    meshes=(mx, mz),
    defaults={
        ("diff", mx.center): fr.operators.FiniteDifference(order=2),
        # mz coefficient spaces only admit the spectral derivative;
        # no entry needed
    },
)

b = grid.create_field(mx.center * mz.galerkin(bc=fr.BC.DIRICHLET))
db_dz = b.diff("z")   # Chebyshev recurrence, BC baked into the basis
db_dx = b.diff("x")   # FD stencil with halo exchange along x only
```

### 4.5 Constant broadcast (topo replacement)

```python
# depth-independent Coriolis parameter on a 3D grid
mx, my, mz = grid.factors
f_cor = grid.create_field(mx.center * my.center * mz.constant)
u3d = grid.create_field(mx.right * my.center * mz.center)

tend = f_cor * u3d    # automatic exact broadcast along z
```

### 4.6 Operator eigenvalues for exact spectral solvers

```python
fd = fr.operators.FiniteDifference(order=2)

# eigenvalues are queried per coefficient *factor* space (section 2.5),
# grid-mediated: the grid supplies the sharded wavenumbers and dx, so
# the 1D operator needs no axis tag. A *factor* is the per-mesh 1D space
# of the flat tensor product; `.factor("x")` selects it by name and
# `.factors` iterates them. (grid is reachable as u_hat.grid.)
grid = u_hat.grid
fourier_x = u_hat.function_space.factor("x")
kx = fd.eigenvalues(grid, fourier_x)  # a Symbol (diagonal operator),
                                      # replaces discrete_spectral_operators

# a Symbol is a diagonal operator in coefficient space; `*`, `**`, `+`,
# `1 / .` are its diagonal (elementwise) algebra — never the physical
# product/convolution a ScalarField's `*` would mean. The Laplacian
# supplies its own eigenvalue symbol directly (-|k|^2):
lap = fr.operators.Laplacian(order=2).eigenvalues(grid, u_hat.function_space)
# equivalently, composed from the 1D symbols (what the operator does
# internally); the per-factor symbols broadcast across the product as
# diagonal-operator extensions (Identity ⊗ D — not the field lift of
# section 3.3; see class doc 03). Shorthand — the FD first-derivative
# symbol retags origins, so the honest composition is `bwd @ fwd`
# (class doc 03), not `** 2`:
#   lap = sum(fd.eigenvalues(grid, s) ** 2
#             for s in u_hat.function_space.factors)

# spectral pressure solve: 1 / lap is the inverse diagonal operator,
# applied (symbols are callable) as a Hadamard multiply (section 3.11);
# the k = 0 mode is regularized by the solver
# (`Symbol.inverse(where_zero=...)`, class doc 03; bare 1 / lap keeps
# jax inf semantics)
p_hat = (1 / lap)(-div_hat)
```

Eigenvalues exist only where the basis diagonalizes the operator
(section 2.5), so this exact-inverse route needs *every* factor to be
Fourier (a fully periodic grid). On a mixed grid — e.g. Fourier in x,
Chebyshev in z — the operator is not diagonal, and the vertical is
solved as a banded per-column system instead.

### 4.7 Worked example: finite volume on a C-grid

Space assignment for the velocity components (flux form):

```
u : Outer(x) ⊗ CellAvg(y) ⊗ CellAvg(z)   # point value in x at the
v : CellAvg(x) ⊗ Outer(y) ⊗ CellAvg(z)   # face, averaged over the
w : CellAvg(x) ⊗ CellAvg(y) ⊗ Outer(z)   # face in the other axes
```

These are the 2D face-average DOFs the divergence theorem wants: with
this staggering,

```
div : (u, v, w) -> CellAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)
```

is **exact** — the deep reason C-grids and FV fit together, now
visible in the operator's type. (Periodic axes use `Right`; bounded
axes with no-normal-flow BCs use `Inner`, sections
[3.5](02_rules.md#35-shape-is-a-property-of-the-space) and
[3.6](02_rules.md#36-boundaries-i-conforming-bc-structure-vs-boundary-data).)

The alternative **momentum-control-volume** formulation integrates
the u-equation over the staggered u-cell:

```
u_cv : FaceAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)   # 3D average over the
                                              # staggered box
```

`u` and `u_cv` are *different functionals of the same continuous
velocity*, equal only to O(dx^2). Second-order codes (including
current FRIDOM) silently identify them; here the identification is an
explicit approximate conversion operator — declared, order-controlled,
swappable. Both formulations are designed-for; iteration 1 implements
the flux-form spaces only (matching the current second-order schemes).

Reconstructing u in v's space (Coriolis term) factors per axis
([section 2.3](01_concepts.md#23-tensorproductspace-and-named-coordinates));
the three 1D operators commute; `.to` dispatches per axis to the
reconstruction operators below:

| Axis | 1D signature         | Operation             | 2nd order      |
|------|----------------------|-----------------------|----------------|
| x    | `Outer -> CellAvg`   | evaluate-to-average   | trapezoid mean |
| y    | `CellAvg -> Outer`   | Shu average-to-point  | two-point mean |
| z    | `CellAvg -> CellAvg` | identity              | —              |

```python
u_at_v = u.to(v)                          # per-axis dispatch (reconstruct)
dv_dt -= f_cor * u_at_v                   # f_cor broadcasts (section 3.3)
```

At second order the composition collapses to the classic four-point
C-grid average — the correctness anchor against today's
`LinearInterpolation`. At higher order, evaluate-to-average and
average-to-evaluate are genuinely different stencils: the place where
the nodal-vs-average type distinction earns its keep.

### 4.8 Fields from functions of physical coordinates

```python
def u_ini(x, y):
    return x * y

mx, my = grid.factors

# nodal space: sample at the (staggered) nodes
u = grid.create_field(mx.right * my.center, init=u_ini)

# average space: per-cell quadrature (midpoint at 2nd order)
u_bar = grid.create_field(mx.cell_avg * my.cell_avg, init=u_ini)

# coefficient space: sample at the origin's nodes, then transform
# (discretize = transform o discretize_origin, section 3.10)
fourier = mx.fourier(origin=mx.center) * my.fourier(origin=my.center)
u_hat = grid.create_field(fourier, init=u_ini)

# arguments are matched to coordinate names by keyword; a z-only
# profile on a 3D grid (ConstantSpace factors in x and y):
b = grid.create_field(
    mx.constant * my.constant * mz.center,
    init=lambda z: -N2 * z,
)
```

### 4.9 Random-spectra initial condition (spectral-space construction)

The successor of `RandomGeostrophicSpectra`: an initial condition
that is *constructed in coefficient space* — an assignment of
coefficients, not the discretization of any physical-coordinate
function. It composes the
[section 3.10](02_rules.md#310-discretizing-continuous-functions)
companions (`init_coeff=`,
`grid.random`) with the model-side eigenmode object
([section 2.5](01_concepts.md#25-operator--typed-maps-between-spaces),
model-side and out of scope for the grid redesign):

```python
import jax.numpy as jnp

import fridom.framework2 as fr
import fridom.nonhydro as nh

# model-side eigenmode object; replaces grid.vec_q. Components live
# on *different* coefficient spaces (unlike today, where the whole
# state shares one spectral grid):
#   q.u on Fourier(x, origin=Right)  ⊗ Fourier(y, origin=Center) ⊗ ...
#   q.b on Fourier(x, origin=Center) ⊗ Fourier(y, origin=Center) ⊗ ...
# The eigenmode object owns the discrete-exact assembly, including
# the phase-shift factors between origins (eigenvalues, section 2.5).
q = nh.eigenmodes.geostrophic(grid, params, s=0)

def amp(kx, ky, kz):
    return jnp.sqrt(spectral_energy_density(kx, ky, kz))

# per-component scaling: the *same* functions, evaluated on each
# component's own coefficient space. |k| is origin-independent, so
# the values agree; only the space tags differ. Strict algebra
# (section 3.1) holds with no new exceptions.
def scale(f):
    a = grid.create_field(f.function_space, init_coeff=amp)
    r = grid.random.phase(f.function_space, seed=12345)
    # coefficient-wise (Hadamard) product, *not* the physical product
    # `*`: scaling a spectrum by amplitude and phase is elementwise in
    # coefficients (section 3.11).
    return fr.operators.Hadamard()(f, a, r)

z_hat = q.map(scale)               # State-wide functional map

# the transform operator applied to a state maps over components;
# backward is unambiguous per component: the origins are constitutive
t = fr.operators.Fourier(grid, axes=("x", "y", "z"))
z = t.backward(z_hat)              # (section 3.2)
```

Notes:

- the scaling is `Hadamard`, the coefficient-wise product
  ([section 3.11](02_rules.md#311-field-operations-linear-ops-and-the-product-problem)),
  not the physical product `*`: multiplying a spectrum by an amplitude
  and a phase is elementwise in coefficients, and its physical-space
  meaning (a convolution) is not intended here.
- `init_coeff=` evaluates `amp` at `grid.wavenumbers(space)`
  ([section 3.10](02_rules.md#310-discretizing-continuous-functions));
  no physical-space detour, no hand-built shapes.
- `grid.random.phase(space, seed)` returns a unit-modulus complex
  field on the coefficient space — seeded, sharding-consistent via
  per-shard keying by global true-DOF index
  ([section 3.10](02_rules.md#310-discretizing-continuous-functions)).
- Reality of the resulting physical fields is structural: the
  Fourier spaces of real origins have rfft-layout shapes
  ([section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces)),
  so the assigned coefficients cannot break the
  Hermitian constraint.
- Eigenvector normalization (energy 1 per mode) happens inside the
  eigenmode object, where the parameters (`dsqr`, `N^2`) live —
  physics stays out of the grid
  ([section 2.6](01_concepts.md#26-grid--the-assembly-object)).

### 4.10 Dealiased pseudospectral product (transform-once)

The successor of `SpectralAdvection`: dealiasing is carried by padded
transforms ([section 3.12](02_rules.md#312-dealiasing)), and the
nonlinear term is assembled by transforming to the finer physical space
once, multiplying there with plain products, then transforming back.
(The old `z.fft`/`z.ifft` methods are going away; the transform is a
grid-bound operator with `forward`/`backward`, as in sketch 4.3.)

```python
# pad factor from the nonlinearity degree: (p + 1)/2  -> 3/2 for p = 2
t = fr.operators.Fourier(grid, axes=("x", "y"), pad=fr.dealias.degree(2))

# transform the whole state to the finer nodal space *once*
up = t.backward(u_hat)       # -> Center((3/2) N) ⊗ ...   (a real space)
vp = t.backward(v_hat)

# many plain CollocationProducts on the finer grid (no aliasing), then
# one forward transform + trim per flux back to Fourier(N)
uu_hat = t.forward(up * up)
uv_hat = t.forward(up * vp)
```

Notes:

- `up * vp` is an ordinary physical product (`CollocationProduct`,
  [section 3.11](02_rules.md#311-field-operations-linear-ops-and-the-product-problem));
  the padded transforms do the dealiasing. The finer space is
  first-class ([section 3.5](02_rules.md#35-shape-is-a-property-of-the-space)),
  so the strict algebra applies there unchanged.
- `Convolution(t)(u_hat, v_hat)`
  ([section 3.12](02_rules.md#312-dealiasing)) would give the same
  result in one call, but re-transforms shared operands; the
  transform-once form above is the performant idiom for a multi-term
  right-hand side, provided as a reusable **transform-once combinator**
  ([section 3.12](02_rules.md#312-dealiasing)) — automatic scheduling
  over a lazy expression graph is rejected (it conflicts with the
  no-lazy-field rule of
  [section 3.10](02_rules.md#310-discretizing-continuous-functions)).

### 4.11 Complex-valued fields (choice of scalars)

The scalars a space is defined over — its Körper, `fr.Real` or
`fr.Complex` ([section 3.1](02_rules.md#31-strict-space-algebra)) — are
part of the space, so a complex nodal field (a wave envelope, a
complex-Ginzburg-Landau state) is just a field on the complex variant
of a space:

```python
mx, my = grid.factors

# real is the default; opt into complex with .as_complex()
env = grid.create_field(
    (mx.center * my.center).as_complex(),
    init=lambda x, y: jnp.exp(1j * k0 * x) * jnp.exp(-(y**2)),
)
print(env.function_space.scalars)     # fr.Complex

f = grid.create_field(init=lambda x, y: y)   # fr.Real by default

# real -> complex promotion is implicit and exact (section 3.1); the
# result lives on the complex space
mix = f + env                         # ok -> Complex(x) ⊗ Complex(y)

# extract the two real parts
re, im = env.real, env.imag           # each fr.Real

# scalars drive transform dispatch with no special-casing: the same
# operator resolves ("transform", space) on the input's scalars
t = fr.operators.Fourier(grid, axes=("x", "y"))
f_hat = t.forward(f)                   # fr.Real   input -> rfft
env_hat = t.forward(env)              # fr.Complex input -> full fft
```

The dtype is derived, never a flag: `f` stores a real array, `env` a
complex one, and the Fourier coefficients of the real `f` are still a
complex Hermitian half-spectrum
([section 3.1](02_rules.md#31-strict-space-algebra),
[section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces)).
