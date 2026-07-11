---
status: normative
date: 2026-07-06
---

# Grid abstraction redesign — Class designs: transforms

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. Base hierarchy and `Symbol` are in [`operators_base.md`](operators_base.md).

---

## Transforms

Transforms are the **deliberate exception** to grid-freedom (§2.5):
they bind the grid at construction because they need the domain
decomposition and FFT plan up front, and they are applied through
`forward`/`backward`.

### Transform

| | |
|---|---|
| Kind | ABC |
| Pytree | static structure; `_grid` is a fully static reference (plans, layouts, refined meshes — G1) |
| Iteration | 1 |
| Concept refs | §2.5, §3.2, §3.12, sketches 4.3, 4.10 |
| Module | `framework2.grid.operators.transform` |

```python
@fr.utils.jaxify
class Transform(UnaryOperator, ABC):
    """Grid-bound change of representation: nodal <-> coefficient."""

    dispatch_kind: ClassVar[str | None] = "transform"

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid; transform along ``axes`` (default: all)."""
        ...

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def grid(self) -> Grid:
        """The bound grid (decomposition + plans)."""
        ...

    @property
    def axes(self) -> tuple[str, ...]:
        """Coordinate names this transform acts along."""
        ...

    @property
    def pad(self) -> PadFactor | None:
        """Dealiasing pad factor, or None for the plain transform."""
        ...

    # ------------------------------------------------------------
    #  Application
    # ------------------------------------------------------------
    @abstractmethod
    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Nodal/average -> coefficient (trims if padded)."""
        ...

    @abstractmethod
    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Coefficient -> origin nodal space (padded if ``pad``)."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Delegate to ``forward`` (registry-uniform application)."""
        ...

    # ------------------------------------------------------------
    #  Space resolution
    # ------------------------------------------------------------
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Forward target: the per-origin coefficient space."""
        ...

    def backward_space(self, domain: FunctionSpace) -> FunctionSpace:
        """Backward target: origin space, refined by ``pad`` if set."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "transpose" (distributed transforms)."""
        ...
```

Notes:

- **Per-origin coefficient handling** (sketch 4.3): `codomain` maps
  each named-axis factor `s` to `s.mesh` Fourier/…-space *with origin
  `s`* (`mx.fourier(origin=...)` — the fixed space-factory anchor).
  `backward` needs no target argument: the origin is constitutive of
  the coefficient space (§3.2), so the round trip is unambiguous.
- `forward`/`backward` on a `VectorField`/`State` map componentwise
  via the inherited `VectorField.map` (sketch 4.9); each component
  keeps its own per-origin coefficient spaces.
- **The plan is a lowered composite**
  ([§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space)):
  binding builds one 1D kernel per axis; `axes` is a **set** — the
  planner reorders the stages for maximum speed (owner decision:
  locally-available axes first), and the requirements-driven lowering
  (doc 03) inserts `Reshard` nodes with explicit endpoints, chosen by
  shortest path over the negotiated layout graph. The codomain's
  layout is the schedule's final pencil — nothing reshards back to
  the default layout implicitly. Users who need a specific stage
  order compose single-axis transforms
  (`Fourier(axes="x") @ Fourier(axes="y")` pins it).
- **Dealiasing is a property of the transform** (§3.12): with
  `pad=degree(p)`, `backward` lands in the finer nodal space
  (`Fourier(N) -> Center((p+1)/2 N)`, a genuine first-class space)
  and `forward` from that finer space trims back to the unpadded
  coefficient space. Both space families are fixed at construction:
  the finer nodal space lives on the refined mesh obtained via
  doc 01's `StructuredMesh1D.refined(factor)` (iteration 1 on
  `IntervalMesh`, `Fraction` factor), called once when the transform
  binds the grid.
- **Padded-`forward` codomain exception.** The padded `forward` takes
  the *refined* mesh's `Center((3/2) N)` and lands on the **coarse**
  space's coefficient space — a deliberate exception to the
  per-origin codomain rule (the codomain's origin is the coarse
  `Center(N)`, not the refined domain). The transform stores both
  space families at construction and derives the coarse target
  through doc 01's `refined_from` parent link on the refined mesh.
- **Deviation callout.** §2.5 says transforms are applied through
  `.forward`/`.backward` "rather than" `op(field, axis=...)`; the
  inherited `__call__` (via `_apply` delegating to `forward`)
  deviates from that letter for registry uniformity — a
  `("transform", space)` entry must be applicable through the
  uniform operator calling convention. User code is still expected
  to write `forward`/`backward`; the deviation is recorded here
  explicitly.

### Fourier

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static structure (bound grid) |
| Iteration | 1 (`pad=` included: parity with `FFTPadding`); `truncation_mask` designed-for (with `Symbol`) |
| Concept refs | §3.1, §3.2, §3.12, sketches 4.3, 4.10, 4.11 |
| Module | `framework2.grid.operators.fourier` |

```python
@final
class Fourier(Transform):
    """FFT-family transform to per-origin Fourier coefficient spaces."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan (r)FFTs along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """rfft for fr.Real spaces, full fft for fr.Complex spaces."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Inverse transform to the (padded) origin nodal space."""
        ...

    def truncation_mask(
        self,
        space: FunctionSpace | TensorProductSpace,
        keep: Fraction = Fraction(2, 3),
    ) -> Symbol:
        """0/1 truncation-filter Symbol on ``space`` (2/3 rule)."""
        ...
```

Notes:

- **Scalars drive the realization with no special-casing** (§3.1,
  sketch 4.11): `fr.Real` origins produce the Hermitian half-spectrum
  coefficient space (rfft layout as *shape*, §3.2), `fr.Complex`
  origins the full spectrum. The `RFFTPressureSolver` bypass
  disappears: the rfft *is* the dispatched default. **Multi-axis real
  fields** (§5.1): only the schedule's *first* stage is real→complex
  — its factor keeps the real origin and the half spectrum; every
  later stage sees complex data and targets the complexified origin
  (`s.as_complex()`, full spectrum). Per-axis real transforms
  therefore do not commute; the planner's stage order picks the
  half-spectrum factor, statically per grid.
- **Average origins are ordinary origins** (G2): `forward` from
  `CellAvg`/`FaceAvg` spaces is the (r)fft of the stored averages,
  landing on `Fourier(origin=cell_avg)`-style coefficient spaces —
  the origin tag carries the `sinc` relationship to the nodal
  origins, converted explicitly by `SincShift` (§3.2/§3.9).
- `truncation_mask` returns the fixed 0/1 diagonal
  `Fourier(N) -> Fourier(N)` of §3.12's 2/3 rule as a `Symbol`
  (applied as a Hadamard); it lives on the transform because the
  retained-band bookkeeping does.
- Distributed operation is transpose-based (jaxDecomp-style), declared
  through `requirements`, negotiated by the grid (doc 04), and
  realized by the lowering-inserted `Reshard` stages of the plan
  (§5.1); the transform API is deliberately rich enough that solvers
  no longer bypass it
  ([§5](../04_decomposition.md#5-domain-decomposition)).

### Sine / Cosine

DST/DCT transforms for bounded nodal spaces with Dirichlet/Neumann
structure.

| | |
|---|---|
| Kind | concrete (final), two classes |
| Pytree | static structure (bound grid) |
| Iteration | 1 (U4: all coefficient spaces and transforms are day one; bounded-axis parity with today's `cartesian/fft.py` DST/DCT paths) |
| Concept refs | §3.2, §3.5, §3.6 |
| Module | `framework2.grid.operators.trig` |

```python
@final
class Sine(Transform):
    """DST transform; type (I/II) selected by the origin node set."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan DSTs along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Nodal Dirichlet space -> sine coefficient space."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Sine coefficients -> the origin nodal space."""
        ...


@final
class Cosine(Transform):
    """DCT transform; type selected by the origin node set."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan DCTs along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Nodal Neumann space -> cosine coefficient space."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Cosine coefficients -> the origin nodal space."""
        ...
```

Notes:

- The DST-I vs DST-II choice is **not a parameter**: it follows from
  the origin space (Dirichlet `Inner` -> DST-I with n - 1 modes,
  Dirichlet `Center` -> DST-II with n modes, §3.2/§3.5) — coefficient
  counts equal space shapes by construction, replacing today's
  position-driven if/else in `cartesian/fft.py`.
- A single `Trig` class multiplexing sine/cosine by BC was rejected:
  the two have different codomain families and `d/dx` couples them
  (`Sine <-> Cosine` under `SpectralDerivative`), so distinct classes
  keep signatures honest.
- The mode-index alignment between the sine and cosine coefficient
  families (scipy type-I/II conventions) is fixed in the
  `SpectralDerivative` bookkeeping notes above; the transforms and
  the derivative share those index maps.

### Chebyshev

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static structure (bound grid) |
| Iteration | 1 (U4: `ChebyshevMesh` and its spaces are iteration 1 per doc 01) |
| Concept refs | §3.2, §6.2 |
| Module | `framework2.grid.operators.chebyshev` |

```python
@final
class Chebyshev(Transform):
    """Chebyshev transform (Gauss-Lobatto nodes <-> Cheb coefficients)."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan Chebyshev transforms along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Gauss-Lobatto nodal -> Chebyshev/Shen coefficient space."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Chebyshev/Shen coefficients -> the origin nodal space."""
        ...
```

Shen-basis codomains (BCs baked in, shape n - k) follow the origin
space's BC structure (§3.6). `requirements` declares
`layout "transpose"` along its axes: per the owner directive,
Chebyshev meshes declare **transpose-capable** decomposition traits,
so the transform plans pencil layouts exactly like `Fourier` (the
§6.2 shard-x/y-keep-z-local layout remains a valid negotiation
outcome, no longer a structural restriction).

### PadFactor and `dealias.degree`

| | |
|---|---|
| Kind | final frozen dataclass + factory function |
| Pytree | static value |
| Iteration | 1 (consumed by `pad=`) |
| Concept refs | §3.12, sketch 4.10 |
| Module | `framework2.grid.operators.dealias` |

```python
@dataclass(frozen=True)
class PadFactor:
    """Dealiasing pad factor for padded transform variants."""

    #: refinement ratio of the padded nodal space, e.g. 3/2
    factor: Fraction


def degree(p: int) -> PadFactor:
    """Pad factor (p + 1) / 2 for a degree-p nonlinearity."""
    ...
```

Sketch 4.10 spells this `fr.dealias.degree(2)`; normatively it lives
in `fr.operators.dealias` (only `fr.meshes` / `fr.operators` are
fixed top-level namespaces, §8). A top-level `fr.dealias` alias is an
open question below.

---

