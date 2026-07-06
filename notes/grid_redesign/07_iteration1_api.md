# Grid abstraction redesign — Iteration-1 public API

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map. Concepts are in
[`01_concepts.md`](01_concepts.md), rules in
[`02_rules.md`](02_rules.md).

---

## 10. Iteration-1 public API (cheat sheet)

The design notes are deep because they *design for* future grid types
(spheres, unstructured meshes, terrain-following coordinates,
Galerkin-BC spaces, dealiasing combinators, `Symbol` algebra) — most
of which are **design-for, not implement**
([section 6](05_validation.md#6-paper-validation-against-future-grid-types)).

This page lists the **small surface a day-one user actually types** in
iteration 1, which implements the **uniform FD/FV tensor grid** at
feature parity with the current cartesian grid plus the full
coefficient-space layer (Fourier, DST/DCT, Chebyshev collocation
including `ChebyshevMesh`). Everything not listed here or tagged
iteration-1 in the class-design docs
([`classes/`](classes/README.md)) is deferred; the numbered sections
remain the normative reference. Until the final rename
([section 8](00_overview.md#8-migration-strategy)) the spellings below
live under the transitional `fr.grid2.*` package.

### 10.1 Building a grid

```python
import fridom.framework as fr

# cartesian convenience subclass (section 2.6): builds the uniform
# IntervalMesh factors from shape=/extent=/periodic=/names=
grid = fr.grid.cartesian.Grid(
    shape=(256, 256), extent=((0, 1), (0, 1)),
    periodic=True, names=("x", "y"),
)

mx, my = grid.factors        # the Mesh factor objects (property), in names order
```

The model-agnostic assembly root `fr.Grid(meshes=...)` is also
iteration 1 (section 2.6; coordinate names are mesh-constructor
arguments — `IntervalMesh(..., name="x")` — and the grid only
validates them), as is `fr.meshes.ChebyshevMesh` for mixed grids.

### 10.2 Spaces (per-mesh factory attributes)

Produced by the mesh factors, combined with `*` (section 2.3):

| Spelling        | Space                                    |
|-----------------|-------------------------------------------|
| `mx.center`     | nodal cell centers (`Center`)             |
| `mx.right`      | nodal periodic faces (`Right`)            |
| `mx.outer` / `mx.inner` | all / interior faces (bounded axes) |
| `mx.cell_avg`   | primal-cell averages (`CellAvg`)          |
| `mx.face_avg`   | dual-cell averages (`FaceAvg`)            |
| `mx.constant`   | one-DOF broadcast axis (`ConstantSpace`, section 3.3) |
| `mx.fourier(origin=...)` | Fourier coefficient space (section 3.2; `origin` is required) |
| `mx.sine(origin=...)` / `mx.cosine(origin=...)` | DST/DCT coefficient spaces of bounded axes (section 3.2) |
| `mz.chebyshev(origin=...)` | Chebyshev coefficient space (on a `ChebyshevMesh`) |
| `mx.center * my.center` | tensor-product space (section 2.3) |
| `space.as_complex()` | complex-scalars variant (section 3.1) |

Galerkin/Shen BC bases (`mz.galerkin(bc=...)`) are **designed-for,
not in iteration 1** (section 6.2); the `fr.BC` enum itself ships day
one (Dirichlet/Neumann structure is carried by the sine/cosine
spaces).

### 10.3 Making fields

```python
# from a function of physical coordinates (args matched by coordinate
# name); `space` is a positional, optional first argument that defaults
# to the all-Center nodal space (section 3.10)
f = grid.create_field(init=lambda x, y: x * x)
u = grid.create_field(mx.right * my.center, init=lambda x, y: x * y)

# from raw data
g = grid.create_field(mx.center * my.center, data=arr)

# seeded, sharding-consistent random field (section 3.10)
r = grid.random.normal(mx.center * my.center, seed=0)
```

`create_field` is the **single field factory**; `init=`/`init_coeff=`
are mutually exclusive, `data=` is the direct-array companion.

### 10.4 Operating on fields

```python
g = f.diff("x")          # dispatch (kind="diff", space) -> FD (section 3.4)
h = g.to(f)              # convert g to f's space (interpolate/reconstruct/phase-shift)
f + h                    # strict algebra: same space required (section 3.1)
f * h                    # physical product, dispatched per space (section 3.11)
m = f.integrate("x")     # reduce x to ConstantSpace, broadcasts back (section 3.13)
```

Cross-space arithmetic (`f + g` on different spaces) raises
`SpaceMismatchError`; the only implicit exceptions are the
`ConstantSpace` broadcast (section 3.3) and `fr.Real -> fr.Complex`
promotion (section 3.1). Operands must also share the grid object
(`GridMismatchError` otherwise).

Free-standing operators compose
([operator design](../operator_design/00_overview.md)):

```python
fd = fr.operators.FiniteDifference(order=2)
d2 = fd @ fd             # composition: (A @ B)(f) == A(B(f))
dxdy = fd["x"] @ fd["y"] # op["x"] binds the axis (mixed-axis chains)
lap = fd["x"] @ fd["x"] + fd["y"] @ fd["y"]   # sums, scalar/field coeffs
```

### 10.5 Transforms and coordinates

```python
# transforms bind the grid at construction (deliberate exception,
# section 2.5) and expose forward/backward
t = fr.operators.Fourier(grid, axes=("x", "y"))
u_hat = t.forward(u)
u2 = t.backward(u_hat)

# bounded axes: DST/DCT transforms (fr.operators.Sine / .Cosine) and
# the Chebyshev transform are iteration 1 too, same shape

# coordinate/wavenumber access is grid-mediated and requires an explicit
# space (section 3.10) — no no-argument default form
x = grid.evaluation_nodes(mx.center)   # a ScalarField; .data for the raw array
kx = grid.wavenumbers(u_hat.function_space.factor("x"))
```

### 10.6 Deferred to later iterations (not typed by a day-one user)

Galerkin/Shen BC spaces, immersed/masked domains
(`grid.immersed`; iteration 1 implements the boolean-mask subset per
section 3.7, but a day-one user does not type it — the full
volume-fraction surface is deferred), terrain-following coordinates
(`grid.metric`, `CoordinateMapping`), the `Symbol` eigenvalue algebra,
`Convolution` / dealiasing combinators, binary product operators beyond
the `*` default, `VectorField.map`-based eigenmode construction, and
the model-side eigenmode objects. These are specified in sections 2–3 and validated in
[section 6](05_validation.md#6-paper-validation-against-future-grid-types)
so iteration 1 does not preclude them.
