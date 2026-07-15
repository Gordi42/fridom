r"""
Fused distributed spectral solve (slab-decomposed multi-device FFTs).

Description
-----------
Owning plan: ``design/plans/active/distributed_transform_plan.md``
(Stages 2 and 4). This module holds both the fused slab **kernel** and
the transform-driven **resolution** of the distributed spectral solve.

The kernel (:class:`SlabPlan`) is the multi-device realization of the
transform requirement ``layout="transpose"`` (jaxDecomp-style), scoped
to the spectral solve: on a 1-D device mesh the nodal operand is
sharded along one coordinate (the default layout's slab axis ``a``), so
a global ``jnp.fft.rfftn`` makes GSPMD all-gather the full cube onto
every device. :class:`SlabPlan` instead runs the whole pipeline inside
one ``jax.shard_map`` region:

1. local (batched) FFTs along the axes that are **not** sharded — the
   Hermitian ``rfft`` along a local half axis ``h`` first, then the
   full-spectrum axes (including the transpose partner ``b``);
2. one ``jax.lax.all_to_all`` transposing the decomposition (``a``
   becomes local, ``b`` becomes sharded);
3. the FFT along the now-local ``a``;
4. (``solve`` only) the Hadamard multiply by the per-shard slice of the
   inverse-eigenvalue diagonal — sliced by the ``shard_map``
   ``in_specs``, so no device ever holds the unsharded cube;
5. the mirrored inverse path (``all_to_all`` back), so the output
   sharding equals the input sharding — sharding-neutral end to end.

The pipeline lives at the **solve** level (consumed by
``SpectralSolve``) rather than inside ``Fourier``'s fused kernels
because the iteration-1 storage contract keeps coefficient factors
device-local: a standalone distributed ``forward`` would re-replicate
at ``store``. Internally the plan therefore owns its coefficient
representation: the half spectrum sits on a **local** axis (never the
sharded one), every other transformed axis carries the full spectrum on
the complexified origin, and the solve's eigenvalue diagonal is queried
on exactly this internal space — honest per-mode values for negative
wavenumbers included, no Hermitian mirroring of sharded axes. When no
local half axis exists the all-Fourier 2-D real case runs fully complex
and takes the real part on synthesis; an all-trig domain instead stays
real throughout -- the real-to-real kernels never complexify.

The resolution (:func:`resolve_distributed_solve`) is the seam that lets
:class:`SpectralSolve` obtain distribution by composing the *ordinary*
transform: it derives the kernel geometry (sharded axis ``a``, transpose
partner ``b``, local half axis ``h``, internal coefficient space) from
the transform's ``distributed_forward_plan`` (the layout-annotated
planner) rather than a solve-scoped geometry, then pairs the resolved
:class:`SlabPlan` with the elliptic operator's inverse eigenvalue
diagonal on the plan's internal coefficient space. Mixed (walled)
transforms distribute through the same seam: the ``ComposedTransform``
plans jointly across its families and the region lowers to the
families' own 1-D stage kernels (Fourier rfft/fft and the trig
DST/DCT kernels run per shard on locally-held axes; only the
``all_to_all`` is collective). The replicated composite path is kept
whenever any of the fallback conditions fail (a non-1-D mesh, a
padded part, a family outside Fourier/Sine/Cosine — Chebyshev's
block-diagonal solve stays replicated — an operand the planner
declines, or eigenvalues that do not materialize as a
broadcast-shaped diagonal on the internal space); the single-device
program is bitwise unchanged.

Trace stability: the jit-wrapped ``shard_map`` callables are built once
per plan (plans are memoized per ``(grid, bare space)``; solve bodies
additionally per diagonal broadcast shape), so repeated eager
applications hit jax's tracing cache — a warmed model re-run adds zero
compiles (the ``_ReblockPlan`` idiom of ``decomposition/tensor.py``).
"""
# S2/S4: fused distributed solve + kernel, resolved from the transform
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import storage_dtype
from fridom.spatial.operators.base import EigenbasisError
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import ComposedTransform
from fridom.spatial.operators.trig import Cosine, Sine

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.operators.base import FieldLike, Operator
    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.operators.transform import (
        Transform,
        TransformStage,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: transform families with distributable per-stage kernels: their
#: unpadded 1-D stage kernels are pure per-axis functions, so they
#: run per-shard inside the fused region on any locally-held axis.
#: Chebyshev stays on the replicated composite (its solve is
#: block-diagonal, not a diagonal eigenvalue divide).
_STAGE_FAMILIES = (Fourier, Sine, Cosine)


# ================================================================
#  The static plan (geometry + cached shard_map callables)
# ================================================================
class SlabPlan:

    r"""
    One grid's slab-decomposed transform/solve pipeline.

    Description
    -----------
    Static structure resolved once per ``(grid, bare space)`` by
    :func:`resolve_distributed_plan` (build through it, not this
    plumbing constructor): the device mesh, the slab geometry — sharded
    coordinate ``a``, transpose partner ``b``, optional local half
    axis ``h`` — and the internal coefficient space the spectral
    frame lives on. The jit-wrapped ``jax.shard_map`` callables are
    created here once (stable identity: eager re-application adds
    zero compiles); ``solve`` bodies are cached per diagonal
    broadcast shape.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        The decomposition's device mesh (1-D).
    axis_name : str
        The device-mesh axis name.
    domain : SpaceLike
        The bare nodal domain space.
    coeff : SpaceLike
        The internal coefficient space (half spectrum on ``h``,
        full complexified spectra elsewhere).
    layout : Layout
        The nodal layout the pipeline consumes and produces (the
        decomposition's default layout).
    a : int
        Array axis sharded in the nodal frame.
    b : int
        Array axis sharded in the spectral frame (the transpose
        partner).
    h : int | None
        The local Hermitian half-spectrum axis, or None for the
        fully complex internal representation.
    fft_axes : tuple[int, ...]
        All transformed array axes.
    real : bool
        Whether the domain storage is real (synthesis lands real).
    stages : tuple[tuple[Transform, TransformStage], ...] | None, optional
        The per-stage kernel schedule of a mixed (or homogeneous
        trig) plan, pairing each layout-annotated stage with the
        per-family transform whose 1-D kernels realize it, in
        forward execution order (the sharded-axis stage last). None
        selects the fused all-Fourier bodies — byte-for-byte the
        original slab kernel (default: None).
    """

    def __init__(
        self,
        mesh: jax.sharding.Mesh,
        axis_name: str,
        domain: SpaceLike,
        coeff: SpaceLike,
        layout: Layout,
        a: int,
        b: int,
        h: int | None,
        fft_axes: tuple[int, ...],
        real: bool,
        stages: tuple[
            tuple[Transform, TransformStage], ...] | None = None,
    ) -> None:
        """Store the geometry and build the shard_map callables."""
        self._mesh: jax.sharding.Mesh = mesh
        self._axis_name: str = axis_name
        self._domain: SpaceLike = domain
        self._coeff: SpaceLike = coeff
        self._layout: Layout = layout
        self._a: int = a
        self._b: int = b
        self._h: int | None = h
        self._fft_axes: tuple[int, ...] = fft_axes
        self._real: bool = real
        self._stages: tuple[
            tuple[Transform, TransformStage], ...] | None = stages

        ndim = len(domain.shape)
        nodal_spec: list[str | None] = [None] * ndim
        nodal_spec[a] = axis_name
        spectral_spec: list[str | None] = [None] * ndim
        spectral_spec[b] = axis_name
        self._nodal_spec = jax.sharding.PartitionSpec(*nodal_spec)
        self._spectral_spec = jax.sharding.PartitionSpec(
            *spectral_spec)

        # local full-spectrum stages: the transpose partner and any
        # further local axes (everything but ``a`` and ``h``)
        self._pre: tuple[int, ...] = tuple(
            i for i in fft_axes if i not in (a, h))
        self._n_h: int = 0 if h is None else domain.shape[h]

        self._forward: Callable[[jax.Array], jax.Array] = jax.jit(
            jax.shard_map(
                self._forward_local, mesh=mesh,
                in_specs=self._nodal_spec,
                out_specs=self._spectral_spec))
        self._backward: Callable[[jax.Array], jax.Array] = jax.jit(
            jax.shard_map(
                self._backward_local, mesh=mesh,
                in_specs=self._spectral_spec,
                out_specs=self._nodal_spec))
        #: solve callables, cached per diagonal broadcast shape
        self._solve_cache: dict[
            tuple[int, ...], Callable[..., jax.Array]] = {}

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def domain(self) -> SpaceLike:
        """The bare nodal domain space of the pipeline."""
        return self._domain

    @property
    def coeff(self) -> SpaceLike:
        """
        The internal coefficient space of the spectral frame.

        Description
        -----------
        Half spectrum on the local axis ``h`` (when the domain is
        real and a local axis exists), the full complexified spectrum
        on every other Fourier axis, and the real trig coefficient
        space on the trig axes. This deliberately differs from
        the single-device codomain (half spectrum on the first stage
        axis): the representation is internal to the pipeline, and
        eigenvalue diagonals must be queried on it.
        """
        return self._coeff

    @property
    def layout(self) -> Layout:
        """The nodal layout the pipeline consumes and produces."""
        return self._layout

    # ================================================================
    #  Per-shard kernels (run under jax.shard_map)
    # ================================================================
    def _forward_local(self, piece: jax.Array) -> jax.Array:
        """Analysis body: local FFTs, all-to-all, sharded-axis FFT."""
        if self._stages is not None:
            return self._forward_staged(piece)
        c = piece
        if self._h is not None:
            c = jnp.fft.rfft(c, axis=self._h, norm="forward")
        if self._pre:
            c = jnp.fft.fftn(c, axes=self._pre, norm="forward")
        c = jax.lax.all_to_all(
            c, self._axis_name, split_axis=self._b,
            concat_axis=self._a, tiled=True)
        return jnp.fft.fft(c, axis=self._a, norm="forward")

    def _backward_local(self, c: jax.Array) -> jax.Array:
        """Synthesis body: the exact mirror of the analysis."""
        if self._stages is not None:
            return self._backward_staged(c)
        c = jnp.fft.ifft(c, axis=self._a, norm="forward")
        c = jax.lax.all_to_all(
            c, self._axis_name, split_axis=self._a,
            concat_axis=self._b, tiled=True)
        if self._pre:
            c = jnp.fft.ifftn(c, axes=self._pre, norm="forward")
        if self._h is not None:
            return jnp.fft.irfft(c, n=self._n_h, axis=self._h,
                                 norm="forward")
        return c.real if self._real else c

    def _forward_staged(self, piece: jax.Array) -> jax.Array:
        """
        Mixed analysis body: per-stage family kernels.

        Description
        -----------
        Runs each local stage's single-device 1-D kernel (rfft on
        the Hermitian axis first, then the remaining local Fourier /
        trig stages), one ``all_to_all``, and the sharded-axis stage
        on the now-local axis — the per-stage lowering of
        ``distributed_transform_plan.md`` section 2 A4. Every kernel
        is exactly the family's single-device stage kernel, so the
        internal spectrum matches the replicated composite's
        convention per mode.
        """
        c = piece
        for part, stage in self._stages[:-1]:
            c = part._forward_kernel(c, stage)  # noqa: SLF001 — lowering seam
        c = jax.lax.all_to_all(
            c, self._axis_name, split_axis=self._b,
            concat_axis=self._a, tiled=True)
        part, stage = self._stages[-1]
        return part._forward_kernel(c, stage)  # noqa: SLF001 — lowering seam

    def _backward_staged(self, c: jax.Array) -> jax.Array:
        """
        Mixed synthesis body: the exact mirror of the analysis.

        Description
        -----------
        The sharded-axis inverse stage first, the ``all_to_all``
        back, then the local inverse stages in reverse order — the
        Hermitian half stage (when present) last, landing real. An
        all-trig real domain (no Hermitian stage) carries real data
        throughout -- the real-to-real kernels never complexify, so
        the closing real-part guard is a no-op.
        """
        part, stage = self._stages[-1]
        c = part._backward_kernel(c, stage)  # noqa: SLF001 — lowering seam
        c = jax.lax.all_to_all(
            c, self._axis_name, split_axis=self._a,
            concat_axis=self._b, tiled=True)
        for part, stage in reversed(self._stages[:-1]):
            c = part._backward_kernel(c, stage)  # noqa: SLF001 — lowering seam
        if self._real and jnp.iscomplexobj(c):
            return c.real
        return c

    # ================================================================
    #  Application
    # ================================================================
    def forward(self, data: jax.Array) -> jax.Array:
        """
        Analyze a true-shape nodal array into the spectral frame.

        Parameters
        ----------
        data : jax.Array
            The true-shape nodal array, sharded along ``a`` (any
            other input sharding is resharded by the region).

        Returns
        -------
        jax.Array
            The internal-representation coefficient array, sharded
            along ``b``.
        """
        return self._forward(data)

    def backward(self, data: jax.Array) -> jax.Array:
        """
        Synthesize an internal-representation coefficient array.

        Parameters
        ----------
        data : jax.Array
            The coefficient array on ``coeff``, sharded along ``b``.

        Returns
        -------
        jax.Array
            The true-shape nodal array, sharded along ``a``.
        """
        return self._backward(data)

    def solve(self, data: jax.Array, diag: jax.Array) -> jax.Array:
        """
        Run ``backward(diag * forward(data))`` in one region.

        Description
        -----------
        The distributed spectral solve: analysis, per-mode Hadamard
        multiply, synthesis — one ``shard_map`` region, so the
        spectral cube only ever exists as per-device slabs. ``diag``
        enters with a sharded ``in_spec`` along ``b`` (when it
        carries the full extent there), which slices it per shard;
        size-1 (broadcast) axes stay replicated.

        Parameters
        ----------
        data : jax.Array
            The true-shape nodal right-hand side, sharded along
            ``a``.
        diag : jax.Array
            The inverse-eigenvalue diagonal, broadcast-shaped over
            ``coeff`` (every axis size 1 or full).

        Returns
        -------
        jax.Array
            The true-shape nodal solution, sharded along ``a``.
        """
        key = tuple(diag.shape)
        fn = self._solve_cache.get(key)
        if fn is None:
            fn = self._build_solve(key)
            self._solve_cache[key] = fn
        return fn(data, diag)

    def _build_solve(
        self, shape: tuple[int, ...],
    ) -> Callable[..., jax.Array]:
        """Build the jitted solve callable of one diagonal shape."""
        spec_shape = self._coeff.shape
        if len(shape) != len(spec_shape) or any(
                s not in (1, n)
                for s, n in zip(shape, spec_shape, strict=True)):
            raise ValueError(
                f"the solve diagonal must be broadcast-shaped over "
                f"the internal coefficient space {spec_shape}, got "
                f"{shape}")
        spec: list[str | None] = [None] * len(shape)
        if shape[self._b] == spec_shape[self._b]:
            spec[self._b] = self._axis_name
        diag_spec = jax.sharding.PartitionSpec(*spec)

        def body(piece: jax.Array, dloc: jax.Array) -> jax.Array:
            return self._backward_local(
                dloc * self._forward_local(piece))

        return jax.jit(jax.shard_map(
            body, mesh=self._mesh,
            in_specs=(self._nodal_spec, diag_spec),
            out_specs=self._nodal_spec))


# ================================================================
#  The solve wrapper (consumed by SpectralSolve)
# ================================================================
class SlabSolve:

    r"""
    The distributed realization of ``backward @ inverse @ forward``.

    Description
    -----------
    A trace-time object pairing a :class:`SlabPlan` with the inverse
    eigenvalue diagonal on the plan's internal coefficient space.
    ``SpectralSolve`` applies it whenever the operand matches the
    plan (``applies``), falling back to the replicated composite
    otherwise.

    Parameters
    ----------
    plan : SlabPlan
        The resolved slab pipeline.
    inverse : Symbol
        The inverse-eigenvalue diagonal on ``plan.coeff``
        (endomorphic; validate with :func:`symbol_fits` first).
    """

    def __init__(self, plan: SlabPlan, inverse: Symbol) -> None:
        """Pair the plan with its inverse diagonal."""
        self._plan: SlabPlan = plan
        self._inverse: Symbol = inverse

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def plan(self) -> SlabPlan:
        """The resolved slab pipeline."""
        return self._plan

    @property
    def inverse_symbol(self) -> Symbol:
        """The inverse diagonal on the internal coefficient space."""
        return self._inverse

    # ================================================================
    #  Application
    # ================================================================
    def applies(self, f: FieldLike) -> bool:
        """
        Whether the operand matches the plan's domain and layout.

        Parameters
        ----------
        f : FieldLike
            The prospective right-hand-side field.

        Returns
        -------
        bool
            True when the distributed pipeline may run on ``f``.
        """
        space = f.function_space
        return (space.bare is self._plan.domain
                and space.layout == self._plan.layout)

    def __call__(self, f: FieldLike) -> FieldLike:
        """
        Solve on the distributed pipeline (never gathers the cube).

        Parameters
        ----------
        f : FieldLike
            The right-hand-side field (``applies(f)`` must hold).

        Returns
        -------
        FieldLike
            The solution on the same space and layout.
        """
        out = self._plan.solve(jnp.asarray(f.data),
                               self._inverse.data)
        return f.with_data(out)


def symbol_fits(plan: SlabPlan, symbol: Symbol) -> bool:
    """
    Whether a diagonal is consumable by ``plan.solve``.

    Description
    -----------
    The diagonal must be endomorphic on the plan's internal
    coefficient space and broadcast-shaped over it (every axis size
    1 or full) — the preconditions of the per-shard Hadamard
    multiply.

    Parameters
    ----------
    plan : SlabPlan
        The resolved slab pipeline.
    symbol : Symbol
        The materialized eigenvalue diagonal.

    Returns
    -------
    bool
        True when the diagonal fits the plan.
    """
    coeff = plan.coeff
    data = symbol.data
    return (symbol.space is coeff and symbol.codomain is coeff
            and data.ndim == len(coeff.shape)
            and all(s in (1, n) for s, n in
                    zip(data.shape, coeff.shape, strict=True)))


# ================================================================
#  Resolution (from the transform's layout plan; memoized per grid)
# ================================================================
#: per-grid memo of resolved plans, keyed on the interned bare space
#: (the ``mixed.py`` ``WeakKeyDictionary`` idiom: a dropped grid
#: auto-evicts its memo)
_PLANS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _owning_part(
    parts: tuple[Transform, ...], stage: TransformStage,
) -> Transform:
    """
    Return the part whose coefficient family produced ``stage``.

    Parameters
    ----------
    parts : tuple[Transform, ...]
        The candidate per-family transforms.
    stage : TransformStage
        One layout-annotated stage of the joint plan.

    Returns
    -------
    Transform
        The family transform whose kernels realize the stage.
    """
    return next(
        part for part in parts
        if isinstance(stage.coeff, part._space_family))  # noqa: SLF001


def build_distributed_plan(
    transform: Transform | ComposedTransform,
    grid: object,
    bare: SpaceLike,
) -> SlabPlan | None:
    """
    Build the fused solve plan from the transform's layout plan.

    Description
    -----------
    Derives the slab kernel geometry (mesh, sharded axis ``a``,
    transpose partner ``b``, local Hermitian axis ``h``, transformed
    axes, internal coefficient space) from
    ``transform.distributed_forward_plan(bare)``. Eligible transforms
    are the unpadded per-stage-kernel families — plain
    :class:`Fourier`, :class:`Sine` / :class:`Cosine`, and their
    mixed :class:`ComposedTransform` (the walled grid) — a pure
    Fourier plan keeps the fused all-Fourier bodies (byte-for-byte
    the original slab kernel), every other eligible plan lowers to
    the per-stage kernels. Returns None when a part is outside those
    families or padded, the mesh is not 1-D, or the operand is
    single-device / ineligible (the planner returns None) — the
    caller then keeps the replicated composite.

    Parameters
    ----------
    transform : Transform | ComposedTransform
        The transform resolved for ``bare``.
    grid : object
        The grid carrying the decomposition.
    bare : SpaceLike
        The bare nodal operand space.

    Returns
    -------
    SlabPlan | None
        The reusable slab pipeline, or None when ineligible.
    """
    parts = (transform.parts
             if isinstance(transform, ComposedTransform)
             else (transform,))
    if any(not isinstance(part, _STAGE_FAMILIES)
           or part.pad is not None for part in parts):
        return None
    forward = transform.distributed_forward_plan(bare)
    if forward is None:
        return None
    decomposition = grid.decomposition
    mesh = decomposition.device_mesh
    if len(mesh.axis_names) != 1:
        return None
    names = bare.names
    ((name_b, _),) = forward.codomain.layout.device_axes
    half = next((s for s in forward.stages if s.half), None)
    all_fourier = all(isinstance(part, Fourier) for part in parts)
    return SlabPlan(
        mesh=mesh,
        axis_name=mesh.axis_names[0],
        domain=bare,
        coeff=forward.codomain.bare,
        layout=decomposition.default_layout,
        a=forward.stages[-1].index,
        b=names.index(name_b),
        h=None if half is None else half.index,
        fft_axes=tuple(sorted(stage.index for stage in forward.stages)),
        real=not jnp.issubdtype(storage_dtype(bare),
                                jnp.complexfloating),
        stages=(None if all_fourier else tuple(
            (_owning_part(parts, stage), stage)
            for stage in forward.stages)),
    )


def resolve_distributed_plan(
    transform: Transform, grid: object, bare: SpaceLike,
) -> SlabPlan | None:
    """
    Resolve (and memoize) the distributed plan of ``bare``, or None.

    Parameters
    ----------
    transform : Transform
        The transform resolved for ``bare``.
    grid : object
        The grid carrying the decomposition and dispatch registry.
    bare : SpaceLike
        The bare nodal operand space.

    Returns
    -------
    SlabPlan | None
        The memoized plan, or None when ineligible.
    """
    memo = _PLANS.setdefault(grid, {})
    if bare not in memo:
        memo[bare] = build_distributed_plan(transform, grid, bare)
    return memo[bare]


def resolve_distributed_solve(
    elliptic: Operator,
    transform: Transform,
    grid: object,
    bare: SpaceLike,
    where_zero: complex,
) -> SlabSolve | None:
    """
    Resolve the transform-driven distributed solve, or None.

    Description
    -----------
    Pairs the resolved plan with the inverse eigenvalue diagonal on the
    plan's internal coefficient space, through the transform's layout
    plan. Returns None (fall back to the replicated composite) when the
    plan is ineligible, the eigenvalues do not materialize on the
    internal space, or the diagonal is not a broadcast-shaped
    endomorphism there.

    Parameters
    ----------
    elliptic : Operator
        The elliptic operator to invert (an :class:`Operator` recipe; a
        pre-assembled ``Symbol`` is not eligible — it is bound to the
        replicated codomain).
    transform : Transform
        The transform resolved for ``bare``.
    grid : object
        The grid.
    bare : SpaceLike
        The bare nodal operand space.
    where_zero : complex
        The inverse value at structural zeros (the nullspace gauge).

    Returns
    -------
    SlabSolve | None
        The distributed solve, or None when ineligible.
    """
    plan = resolve_distributed_plan(transform, grid, bare)
    if plan is None:
        return None
    try:
        symbol = elliptic.eigenvalues(grid, plan.coeff)
    except (EigenbasisError, SpaceMismatchError):
        return None
    if not symbol_fits(plan, symbol):
        return None
    return SlabSolve(plan, symbol.inverse(where_zero))
