---
status: normative
date: 2026-07-06
---

# Grid abstraction redesign — Class designs: stencil, FV and spectral operators

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. Base hierarchy, the operator algebra, and `Symbol` are in [`operators_base.md`](operators_base.md).

---

## Stencil / nodal operators

All are `SeparableOperator`s: grid-free, axis-agnostic 1D kernels.
Per-factor signatures are written `Domain -> Codomain`; on bounded
meshes the resolver picks the bounded variant as listed.

Per owner directive (G2), the FV/average family is **model-complete
in iteration 1**: every conversion, derivative, product, and
transform a flux-form C-grid model needs resolves from the default
table — see the flux-form closure walk after `FaceDifference` below.

### FiniteDifference

Staggered finite-difference derivative of configurable order; the
default `"diff"` entry on nodal spaces.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.5, §2.7, §3.4, §3.5, sketch 4.6 |
| Module | `framework2.grid.operators.finite_difference` |

```python
@final
class FiniteDifference(SeparableOperator):
    """Staggered finite-difference derivative (order 2, 4, ...)."""

    dispatch_kind: ClassVar[str | None] = "diff"

    def __init__(self, order: int = 2) -> None:
        """Create an FD kernel of the given even order."""
        ...

    @property
    def order(self) -> int:
        """Order of accuracy of the stencil."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """diff: Center -> Right | Inner; Right/Outer/Inner -> Center."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = order // 2, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """i k_hat symbol on a Fourier factor space (retagging)."""
        ...
```

Notes:

- Per-factor signatures: periodic mesh `Center -> Right`,
  `Right -> Center`; bounded mesh `Center -> Inner`
  (n -> n - 1), `Outer -> Center` (n + 1 -> n), `Inner -> Center`.
  Nodal only — the FV derivative on average spaces is `FVDerivative`.
- Stencil *pattern* (from `order`) is static identity; the spacing
  denominators are the **dual center-to-center / primal cell-width
  measure fields** read from the field's grid at trace time
  (`grid.measure(space, name=...)`, doc 04; §2.7) — a uniform mesh
  constant-folds them. This replaces today's
  `_dx1 = 1 / grid.dx` module state
  (`grid/cartesian/finite_differences.py`).
- `eigenvalues(grid, fourier_x)` exists for Fourier factors
  (constant-coefficient FD on a periodic mesh) and returns the exact
  discrete symbol (order 2: `i * 2 sin(k dx / 2) / dx` times the
  inter-origin phase — a retagging symbol). On non-diagonalizing
  factors (Chebyshev): `EigenbasisError`. This is the
  `discrete_spectral_operators.k_hat` successor.

### LinearInterp

Two-point staggering interpolation; the default `"interpolate"` entry on
nodal spaces.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.1, §3.4, sketch 4.1 |
| Module | `framework2.grid.operators.interp` |

```python
@final
class LinearInterp(SeparableOperator):
    """Second-order two-point interpolation between nodal node sets."""

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, target: NodeSet | None = None) -> None:
        """Create the kernel; ``target`` overrides the default codomain."""
        ...

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """interpolate: Center <-> Right (periodic); Center -> Inner,
        Outer/Inner -> Center (bounded); target= selects Outer."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """one_hat averaging symbol on a Fourier factor (retagging)."""
        ...
```

Notes:

- Fixed codomain per §3.4: the registered operator fixes its own
  codomain; a `.to(target)` whose target disagrees is a space error.
  A *call-time* target argument was rejected; alternative codomains
  are **per-instance**, selected by the `target=` constructor knob
  and registered explicitly by whoever needs them.
- Per-factor defaults (`target=None`): periodic `Center <-> Right`;
  bounded `Center -> Inner` (interior faces) and
  `Outer/Inner -> Center`. The periodic `Center <-> Right` rows are
  **periodic-only**: on a bounded mesh `Right` lacks the left
  boundary face, and accepting it would silently impose a one-sided
  boundary treatment.
- **`Center -> Outer` variant** (`target=NodeSet.OUTER`, iteration 1):
  interior faces by the two-point mean; the two boundary faces are
  filled by one-sided linear extrapolation from the two nearest
  centers (the BC-free `Outer` boundary DOFs receive extrapolated
  values; boundary-data-owning modules overwrite them, §3.6). Not a
  default row — same key as the `Inner` default — so modules register
  it explicitly.
- Higher-order centered interpolation (today's
  `PolynomialInterpolation(order=...)`) is a designed-for sibling
  `PolynomialInterp(order)` in the same module; not spelled out here
  because it adds only the `order` parameter to this exact surface.
- `eigenvalues` is the `one_hat` successor: order 2 gives
  `cos(k dx / 2)` times the inter-origin phase.

### LinearReconstruction

Second-order conversions inside the average family
(nodal-at-face <-> dual/primal averages); the default `"reconstruct"`
entry.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.9, sketch 4.7 |
| Module | `framework2.grid.operators.reconstruct` |

```python
@final
class LinearReconstruction(SeparableOperator):
    """2nd-order average <-> point-value conversion (FV family)."""

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(self, target: NodeSet | None = None) -> None:
        """Create the kernel; ``target`` overrides the default codomain."""
        ...

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """CellAvg -> Right (periodic) | Inner (bounded);
        Right/Outer/Inner -> CellAvg; FaceAvg <-> Center dual."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """sinc-corrected averaging symbol on a Fourier factor."""
        ...
```

Notes:

- Covers both directions of sketch 4.7's table at second order:
  average-to-point (`CellAvg -> Right/Inner`, Shu two-point mean) and
  evaluate-to-average (`Outer -> CellAvg`, trapezoid mean). At second
  order both collapse to two-point means — the classic C-grid
  average — but they are *distinct signatures*, and higher-order
  members of the family (`ShuReconstruction(order=...)`,
  designed-for) genuinely differ per direction.
- Bounded default is `CellAvg -> Inner` (interior faces, the
  no-normal-flow C-grid staggering of sketch 4.7); the
  `CellAvg -> Outer` variant is `target=NodeSet.OUTER` with the same
  boundary-extrapolation treatment as `LinearInterp` — an explicit
  instance, not a default row.
- `"reconstruct"` (average <-> point value) is a distinct kind from
  `"interpolate"` (nodal -> nodal), per sketch 4.2; `.to` picks the kind
  from the source/target family (§3.4).

### WenoReconstruction

Nonlinear upwind-biased reconstruction; the module-override
archetype (sketch 4.2).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 on periodic axes (parity with today's `weno_interpolation.py`); bounded-axis boundary biasing designed-for |
| Concept refs | §3.9, sketch 4.2 |
| Module | `framework2.grid.operators.reconstruct` |

```python
@final
class WenoReconstruction(SeparableOperator):
    """WENO average-to-point reconstruction (CellAvg -> face values)."""

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(
        self,
        order: int = 5,
        bias: Literal["left", "right"] = "left",
    ) -> None:
        """Create a WENO kernel of the given odd order and bias."""
        ...

    @property
    def order(self) -> int:
        """Formal order of the WENO reconstruction."""
        ...

    @property
    def bias(self) -> Literal["left", "right"]:
        """Upwind bias side of the reconstruction."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """reconstruct: CellAvg -> Right | Outer (biased)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = order // 2 + 1, layout "any"."""
        ...
```

Notes:

- Nonlinear, so no `eigenvalues` (base raises `EigenbasisError`) —
  correct and automatic.
- Upwinding is a *pair* of biased instances; flux-splitting advection
  modules hold both and select by velocity sign (the sign selection
  itself is the `("select", ...)` kind, `Where` below). The
  alternative — a velocity-consuming ternary operator — was rejected
  as a module-level (physics) concern, consistent with §3.9
  ("advection schemes override reconstruction, not diff").
- **Iteration 1 is periodic-only**: the wide stencil is valid on
  wrap-around halos. Bounded axes need reduced-stencil boundary
  biasing (one-sided smoothness indicators near the wall) —
  designed-for; a bounded-axis registration of the iteration-1
  instance is a space error, not a silent fallback.

### FluxDifference

The exact FV flux difference — the discrete Gauss theorem
([§3.9](../02_rules.md#39-finite-volume-semantics-the-average-family-and-the-fv-derivative)).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.7, §3.9, sketch 4.7 |
| Module | `framework2.grid.operators.flux_diff` |

```python
@final
class FluxDifference(SeparableOperator):
    """Exact flux difference: Outer -> CellAvg (discrete Gauss)."""

    dispatch_kind: ClassVar[str | None] = "flux_diff"

    def __init__(self) -> None:
        """Create the exact face-difference kernel."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """flux_diff: Outer(n+1) | Inner(n-1) -> CellAvg(n) (bounded);
        Right(n) -> CellAvg(n) (periodic only)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """2 i sin(k dx / 2) / dx = i k sinc(k dx / 2) on Fourier."""
        ...
```

Notes:

- Signature `Outer -> CellAvg` with the **primal cell-width measure as
  denominator**: `(u_{i+1/2} - u_{i-1/2}) / w_i`, where `w` is the
  cell-width measure field *on `CellAvg`/`Center`*
  (`grid.measure(space, name=...)`, doc 04; §2.7, §3.9) — never a
  scalar `dx`. `Inner -> CellAvg` is the homogeneous
  no-normal-flow variant (zero boundary fluxes); inhomogeneous
  boundary fluxes occupy the boundary DOFs of an `Outer`-space flux
  field (§3.6) — no extra parameter here.
- The `Right(n) -> CellAvg(n)` row is **explicitly periodic-only**:
  on a bounded mesh `Right` lacks the left boundary face, so
  accepting it would silently impose a one-sided no-flux boundary.
  Bounded domains must present fluxes on `Outer` (explicit boundary
  fluxes) or `Inner` (homogeneous).
- **Exactness is the contract**: summing the output against the cell
  measure telescopes to boundary fluxes; the eigenvalue identity
  `i k sinc(k dx / 2)` = "average of d/dx" is the §3.9 validation
  hook tested against `../05_validation.md`'s FV claims.

### DualFluxDifference

The discrete Gauss theorem on the **dual cells**: fluxes known at
cell centers, differenced into dual-cell averages — the
momentum-control-volume derivative of sketch 4.7.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (G2: momentum advection on the staggered box) |
| Concept refs | §2.7, §3.9, sketch 4.7 |
| Module | `framework2.grid.operators.flux_diff` |

```python
@final
class DualFluxDifference(SeparableOperator):
    """Exact dual-cell flux difference: Center -> FaceAvg."""

    dispatch_kind: ClassVar[str | None] = "flux_diff"

    def __init__(self) -> None:
        """Create the exact dual-cell face-difference kernel."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """flux_diff: Center -> FaceAvg (exact FTC);
        CellAvg -> FaceAvg (O(dx^2) identification)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """i k sinc(k w / 2) on the dual mesh (Fourier factors)."""
        ...
```

Notes:

- **Exact-FTC semantics on the dual mesh**: the dual cell around face
  `i + 1/2` spans `[x_i, x_{i+1}]`, so
  `(1 / w_{i+1/2}) int_dual du/dx = (u(x_{i+1}) - u(x_i)) / w_{i+1/2}`
  — exact when the domain holds **point values at centers**
  (`Center -> FaceAvg`), with `w` the dual cell-width measure field
  on `FaceAvg` (`grid.measure`, §2.7). Summing against the dual
  measure telescopes to the outermost centers — the mimetic property,
  exactly parallel to `FluxDifference`.
- The `CellAvg -> FaceAvg` row carries the **declared O(dx^2)
  identification** of cell averages with midpoint values (§3.9,
  sketch 4.7's "explicit approximate conversion") — fluxes computed
  on average spaces difference into the momentum control volume
  without a silent retag.
- Registered under the same `"flux_diff"` kind, keyed by the center
  domains — no collision with `FluxDifference`'s face-domain rows.

### FaceDifference

The FV pressure gradient: two-point difference of cell values landing
on face point values — signature `diff: CellAvg -> face space` in the
review's notation, registered under its own kind.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (G2: the C-grid pressure-gradient term) |
| Concept refs | §2.7, §3.9, sketch 4.7 |
| Module | `framework2.grid.operators.flux_diff` |

```python
@final
class FaceDifference(SeparableOperator):
    """FV pressure gradient: CellAvg -> face point values."""

    dispatch_kind: ClassVar[str | None] = "face_diff"

    def __init__(self) -> None:
        """Create the two-point dual-spacing difference kernel."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """face_diff: CellAvg -> Right (periodic) | Inner (bounded)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """2 i sin(k dx / 2) / dx times the sinc/phase retag (Fourier)."""
        ...
```

Notes:

- `(p_{i+1} - p_i) / d_{i+1/2}` with `d` the **dual center-to-center
  spacing measure** on `Right`/`Inner` (`grid.measure`, §2.7). The
  codomain is a *point-value* face space (where the C-grid momentum
  DOFs live, sketch 4.7), so the operator carries the **documented
  O(dx^2) identification** on both ends: cell averages read as
  midpoint values, and the two-point difference lands as the face
  point value. This is the declared-conversion discipline of §3.9,
  not a silent equivalence.
- A dedicated kind (`"face_diff"`) rather than `("diff", CellAvg)`:
  a registry entry fixes one codomain, and `("diff", CellAvg)` is
  §3.9's normative `FVDerivative` composition (`CellAvg -> CellAvg`).
  The pressure-gradient term resolves `("face_diff", CellAvg)`.

**Flux-form closure (G2).** The sketch-4.7 C-grid step closes under
the default table: the advecting velocity reaches the flux point via
per-axis `("reconstruct", ...)` rows (`v.to(u_space)`); the flux
itself is `u * q` via the `("multiply", CellAvg/FaceAvg)`
second-order shortcut; the tracer flux divergence is per-axis
`("flux_diff", Outer/Right/Inner) -> CellAvg`; momentum advection
differences center fluxes into the staggered box via
`("flux_diff", Center/CellAvg) -> FaceAvg` (`DualFluxDifference`);
the Coriolis reconstruction `u.to(v)` is the `reconstruct` rows in
both directions; and the pressure gradient is
`("face_diff", CellAvg) -> Right/Inner` (`FaceDifference`).

### FVDerivative

The FV derivative `flux_diff ∘ reconstruct` — the default `"diff"`
entry on average spaces. It is not a class but a factory that builds
the algebra chain; the result is an ordinary `SeparableComposite`.

| | |
|---|---|
| Kind | factory function -> `SeparableComposite` |
| Pytree | static (the composite) |
| Iteration | 1 |
| Concept refs | §3.4, §3.9; merge D1/D4/D5 |
| Module | `framework2.grid.operators.flux_diff` |

```python
def FVDerivative(
    reconstruct: SeparableOperator | None = None,
) -> SeparableComposite:
    """flux_diff @ (reconstruct or Dispatched("reconstruct")).

    With ``reconstruct=None`` the reconstruction is a ``Dispatched``
    placeholder resolved once at model assembly (D4); an explicit
    kernel pins the composition. Both are same-axis separable factors,
    so the chain is a ``SeparableComposite`` (CellAvg -> CellAvg): it
    binds with ``["x"]``, its halo is the sum of the factor halos
    (§3.6), and ``f.diff("x")`` on an average space resolves to it.
    """
    return FluxDifference() @ (reconstruct or Dispatched("reconstruct"))
```

Notes:

- The reconstruction is a **`Dispatched("reconstruct")` placeholder
  resolved once at model assembly** (D4/D5) — so a module override of
  `"reconstruct"` (sketch 4.2) still propagates into what `f.diff("x")`
  does on average spaces (the reason advection schemes override
  reconstruction, not diff), while the baked chain is fully concrete and
  static, with no per-application registry lookup.
- Halo composes additively as an un-synced chain because
  `SeparableComposite.requirements` sums its factor halos (§3.6). The
  registered default is the composite object itself, so the
  halo-accounting trace sees it with no special casing.

---

## Spectral (coefficient-space) operators

### SpectralDerivative

Exact derivative on coefficient spaces; the only `"diff"` choice
there.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.2, §3.4, §6.1 |
| Module | `framework2.grid.operators.spectral` |

```python
@final
class SpectralDerivative(SeparableOperator):
    """Exact derivative in coefficient space (i k multiply / recurrence)."""

    dispatch_kind: ClassVar[str | None] = "diff"

    def __init__(self) -> None:
        """Create the spectral derivative."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Fourier -> Fourier (same origin); Sine -> Cosine; Cheb recurrence."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "local" (whole-axis coefficient access)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """i k on a Fourier factor space (non-retagging)."""
        ...
```

Notes:

- All three paths are **iteration 1** (U4: all coefficient spaces and
  transforms ship day one): the Fourier diagonal `i k` multiply —
  origin preserved (spectral differentiation does not stagger) — the
  sine/cosine bc-flip, and the Chebyshev recurrence. The bc-flipping
  signature `d/dx : SineCoeff -> CosineCoeff` (§3.2) replaces today's
  `SpectralDiff` metadata mutation.
- **Mode-index bookkeeping (sine/cosine pairs).** The index maps are
  explicit and shape-honest (§3.5):
  - *II-type pair (Center origin):* DST-II carries sine modes
    `k = 1..n`, DCT-II cosine modes `k = 0..n-1` (scipy layout:
    DST-II array index `j = 0..n-1` holds mode `k = j + 1`; DCT-II
    index `j` holds `k = j` — the forward map is the array shift
    `j -> j + 1`). `d/dx : Sine -> Cosine` maps sine-`k` to
    cosine-`k` for `k = 1..n-1`, **annihilates the top sine mode**
    `k = n` (its cosine image vanishes identically at the center
    nodes), and never populates cosine `k = 0`. The reverse
    `d/dx : Cosine -> Sine` annihilates the constant `k = 0` and
    lands inside sine `k = 1..n-1`; the top sine mode is never
    populated.
  - *I-type pair (`Inner` Dirichlet <-> `Outer` Neumann):* DST-I
    carries sine modes `k = 1..n-1` (scipy index `j = k - 1`), DCT-I
    cosine modes `k = 0..n` (index `j = k`). `d/dx : Sine -> Cosine`
    lands in cosine `k = 1..n-1` (neither `k = 0` nor `k = n` is
    populated); the reverse annihilates the constant `k = 0` **and**
    the Nyquist cosine `k = n` (its sine image vanishes identically
    at the interior faces) and lands in sine `k = 1..n-1`.
- `layout "local"`: the Chebyshev recurrence couples all modes of the
  factor; the Fourier diagonal case could relax this, but one
  conservative declaration keeps the negotiation simple (revisit knob
  noted in doc 04).

### PhaseShift

Exact inter-origin conversion between Fourier coefficient spaces —
the `"interpolate"` entry in coefficient space
([§3.2](../02_rules.md#32-coefficient-representations-are-separate-spaces),
sketch 4.3).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.2, §3.4, §6.1, sketch 4.3 |
| Module | `framework2.grid.operators.spectral` |

```python
@final
class PhaseShift(SeparableOperator):
    """Inter-origin e^{i k s dx} shift between Fourier spaces."""

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, to: NodeSet = NodeSet.CENTER) -> None:
        """Shift to the Fourier space of the given origin node set."""
        ...

    @property
    def to(self) -> NodeSet:
        """Target origin node set (NodeSet.CENTER, NodeSet.RIGHT, ...)."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Fourier(origin=A) -> Fourier(origin=<to>) on the same mesh."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "any" (pure diagonal multiply)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """The retagging phase symbol e^{i k s dx} itself."""
        ...
```

Notes:

- Diagonal: applying it *is* applying its own symbol. The default
  registry entry `("interpolate", Fourier(origin=Right)) ->
  PhaseShift(to=NodeSet.CENTER)` makes `u_hat.to(w_hat)` in
  sketch 4.3 work; other targets are explicit instances (the `.to`
  sugar errors if the registered codomain does not match the target,
  §3.4).
- **Exactness caveat (real origins, even n).** On the Hermitian
  half-spectrum of a `fr.Real` origin with even `n`, the
  `e^{i k dx/2}` shift is *not* an exact spectrum automorphism: the
  Center-origin Nyquist coefficient is real, and multiplying it by
  `±i` leaves no valid rfft layout — indeed the Nyquist cosine
  sampled at centers vanishes identically at the faces. `PhaseShift`
  therefore **zeroes the Nyquist mode** on real-origin even-`n`
  factors; this is the one non-exact DOF, documented here and in the
  matching §3.2 caveat added to `../02_rules.md` in this change set. On
  complex origins and odd `n` the shift is exact and unitary.
- **One-directional seeding.** Only
  `("interpolate", Fourier(o != center)) -> PhaseShift(to=NodeSet.CENTER)`
  is seeded: a single-codomain entry cannot express per-target
  defaults, so Center -> staggered conversions require explicit
  `PhaseShift(to=NodeSet.RIGHT)`-style instances. Sketch-4.9-style
  eigenmode assembly constructs its inter-origin shifts explicitly
  for the same reason.
- The target is doc 01's `NodeSet` enum member, not a space object —
  the operator stays mesh-agnostic; `codomain` resolves the member
  against the domain's mesh. A string token (`"center"`) is **not**
  accepted, uniform with the no-string-keys rule for interned space
  identities.
- Identity when the origin already matches. On a purely collocated
  spectral grid (§6.1) no entry ever fires — the `DummyInterpolation`
  pathology disappears structurally.

### SincShift

Diagonal conversion between **average-origin** and nodal-origin
Fourier spaces — the named `sinc(k dx / 2)` factor of §3.2/§3.9.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (G2: average-origin coefficient spaces are day one) |
| Concept refs | §3.2, §3.9 |
| Module | `framework2.grid.operators.spectral` |

```python
@final
class SincShift(SeparableOperator):
    """sinc(k dx / 2) conversion between average and nodal origins."""

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, to: NodeSet = NodeSet.CENTER) -> None:
        """Convert to the Fourier space of the given origin node set."""
        ...

    @property
    def to(self) -> NodeSet:
        """Target origin node set (nodal or average)."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Fourier(origin=cell_avg/face_avg) <-> Fourier(origin=nodal)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "any" (pure diagonal multiply)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """The retagging sinc (times phase) symbol itself."""
        ...
```

Notes:

- Cell-averaging is convolution with a top-hat (§3.2), so an
  average-origin spectrum differs from the nodal-origin one by
  `sinc(k dx / 2)`: nodal -> average multiplies, average -> nodal
  divides. Where the origins are additionally offset by half a cell
  (`face_avg -> center`), the diagonal composes the sinc factor with
  the corresponding phase — one operator, one symbol.
- Exists because `PhaseShift(to=...)` targets **nodal** origins only;
  average origins dispatch here (default rows below). The division is
  **invertible on the resolved band**: `sinc(k dx / 2)` has its first
  zero at `k dx = 2 pi`, beyond the Nyquist `k dx = pi`
  (`sinc(pi / 2) = 2 / pi`), so no resolved mode is annihilated.

---

