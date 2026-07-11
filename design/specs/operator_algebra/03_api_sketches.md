---
status: normative
date: 2026-07-06
---

# Operator algebra — API sketches

Part of the operator design notes; see
[`00_overview.md`](00_overview.md) for the document map. Rules are in
[`02_algebra.md`](02_algebra.md). All snippets are **illustrative,
not normative**.

---

## 4. API sketches (non-normative)

### 4.1 Same-axis chains and axis binding

```python
import fridom.framework2 as fr

fd = fr.operators.FiniteDifference(order=2)
li = fr.operators.LinearInterp()

# unbound composition of 1D kernels is still a 1D separable kernel
d_center = li @ fd            # Center -> Right -> Center (per axis)
g = d_center(f, axis="x")     # apply with the call-site keyword ...
g = d_center["x"](f)          # ... or bind first (section 2.3)

# second derivative along one axis; halo = 1 + 1 (section 3.6)
d2 = fd @ fd                  # typing checked at application: the
                              # codomain of the inner fd must equal
                              # the domain of the outer one

# mixed derivative needs bound factors (different axes)
dxdy = fd["x"] @ fd["y"]
h = dxdy(f)                   # no axis keyword: fully bound

fd["x"]["y"]                  # -> error: already bound
```

### 4.2 The FV derivative as a registered composite

```python
# the exact discrete Gauss theorem and a default reconstruction
flux_diff = fr.operators.FluxDifference()          # Outer -> CellAvg, exact
recon = fr.operators.PolynomialReconstruction(2)   # CellAvg -> Outer

# grid default (grid-redesign section 3.9): diff = flux_diff o reconstruct.
# The reconstruct factor is a kind placeholder resolved against the
# *merged* registry at assembly (section 3.10), so module overrides of
# ("reconstruct", ...) propagate into the composite:
grid = fr.Grid(
    meshes=(mx, my), names=("x", "y"),
    defaults={
        ("reconstruct", mx.cellavg): recon,
        ("diff", mx.cellavg):
            flux_diff @ fr.operators.Dispatched("reconstruct"),
    },
)

# a module swaps the approximate factor only (grid-redesign sketch 4.2);
# the exact flux difference — and conservation — is untouched:
class MyAdvection(fr.modules.advection.AdvectionBase):
    def __init__(self):
        super().__init__()
        self.dispatch["reconstruct"] = fr.operators.WenoReconstruction(5)
```

### 4.3 grad / div / curl / laplacian as block operators

```python
# C-grid, 2D. Signatures are space *tuples* (section 3.1); linear
# operators between tuples are block matrices (section 3.5).
fd = fr.operators.FiniteDifference(order=2)

grad = fr.operators.Block([[fd["x"]],          # (Center⊗Center,) ->
                           [fd["y"]]])         # (Right⊗Center, Center⊗Right)

div = fr.operators.Block([[fd["x"], fd["y"]]]) # the transpose layout

# @ is block matmul: (1 x 2) @ (2 x 1) = 1 x 1; the single block is
# fd["x"] @ fd["x"] + fd["y"] @ fd["y"] — computed on demand for halo
# and symbol queries, never eagerly rewritten (section 3.5)
lap = div @ grad

vel = fr.VectorField(u=u, v=v)   # component-space tuple = div.domain
zeta = fr.operators.Block([[-fd["y"], fd["x"]]])(vel)   # curl in 2D

ke = fr.operators.KineticEnergy()   # nonlinear, tuple-signature,
ke(vel)                             # opaque: composes with @, but has
                                    # no blocks and no symbol
```

### 4.4 Terrain-following derivative: a sum with a field coefficient

```python
# d/dx|_z = d/dx|_sigma - (sigma H_x / H) d/dsigma
# (grid-redesign section 3.8), now literal code (section 3.4):
fd = fr.operators.FiniteDifference(order=2)

c = grid.metric("sigma_hx_over_h", u.function_space)  # dynamic leaf;
                                                      # time-dependent
                                                      # metrics trace
                                                      # through

ddx_z = fd["x"] - c * fd["sigma"]

# the coefficient lives on the codomain of fd["sigma"]; ConstantSpace
# broadcast makes plain scalars the constant special case. Halo of the
# sum is the per-axis max of the terms (section 3.6). The sum has no
# symbol unless c is constant (section 3.7).
du_dx = ddx_z(u)
```

### 4.5 Dealiased products: binary pre-composition

```python
# Convolution as the literal composite of grid-redesign section 3.12,
# using both binary rules of section 3.8: `@ B` on a binary operator
# pre-composes B onto *each* operand; `A @` post-composes the result.
t = fr.operators.Fourier(grid, axes=("x", "y"))
tp = fr.operators.Fourier(grid, axes=("x", "y"), pad=fr.dealias.degree(2))

conv = t.forward @ fr.operators.CollocationProduct() @ tp.backward
w_hat = conv(u_hat, v_hat)

# per-operand pre-composition takes a tuple (one unary op per operand):
duv_hat = (t.forward
           @ fr.operators.CollocationProduct()
           @ (fd["x"] @ tp.backward, tp.backward))(u_hat, v_hat)
```

`t.forward`/`tp.backward` are ordinary unary operators with concrete
signatures (section 3.9), so these chains are eagerly type-checked at
composition time. For multi-term right-hand sides the transform-once
combinator remains the scheduling tool (grid-redesign section 3.12);
`conv` re-transforms shared operands.

### 4.6 Verifying FV exactness through composed symbols

```python
# symbol of a chain = mode-wise product of factor symbols, each
# relative to its own domain's coefficient space (section 3.7)
fv_diff = flux_diff @ recon          # CellAvg -> Outer -> CellAvg

k = grid.wavenumbers(coeff_space)    # coefficient space of CellAvg
sym = fv_diff.eigenvalues(grid, coeff_space)     # a Symbol

# grid-redesign section 3.9: reconstruct and flux-difference symbols
# multiply to i k sinc(k dx / 2) — the eigenvalue of "average of d/dx".
# The exactness statement is now a test against the composed operator:
expected = 1j * k.data * jnp.sinc(k.data * dx / (2 * jnp.pi))
assert jnp.allclose(sym.data, expected)

# and the spectral solve of grid-redesign sketch 4.6 works with
# composed symbols unchanged: block operators yield symbol matrices,
# chains yield products
lap_sym = (div @ grad).eigenvalues(grid, u_hat.function_space)
p_hat = (1 / lap_sym)(-div_hat)      # Hadamard application
```
