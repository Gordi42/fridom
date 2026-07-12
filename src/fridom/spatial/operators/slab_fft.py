r"""
Distributed slab-decomposed spectral pipeline (multi-device FFTs).

Description
-----------
The multi-device realization of the transform requirement
``layout="transpose"`` (jaxDecomp-style), scoped to the spectral
solve: on a 1-D device mesh the nodal operand is sharded along one
coordinate (the default layout's slab axis ``a``), so a global
``jnp.fft.rfftn`` makes GSPMD all-gather the full cube onto every
device. The :class:`SlabPlan` instead runs the whole pipeline inside
one ``jax.shard_map`` region:

1. local (batched) FFTs along the axes that are **not** sharded —
   the Hermitian ``rfft`` along a local half axis ``h`` first, then
   the full-spectrum axes (including the transpose partner ``b``);
2. one ``jax.lax.all_to_all`` transposing the decomposition (``a``
   becomes local, ``b`` becomes sharded);
3. the FFT along the now-local ``a``;
4. (``solve`` only) the Hadamard multiply by the per-shard slice of
   the inverse-eigenvalue diagonal — sliced by the ``shard_map``
   ``in_specs``, so no device ever holds the unsharded cube;
5. the mirrored inverse path (``all_to_all`` back), so the output
   sharding equals the input sharding — the pipeline is
   sharding-neutral end to end.

The pipeline lives at the **solve** level (consumed by
``SpectralSolve``) rather than inside ``Fourier``'s fused kernels
because the iteration-1 storage contract keeps coefficient factors
device-local: a standalone distributed ``forward`` would re-replicate
at ``store``. Internally the plan therefore owns its coefficient
representation: the half spectrum sits on a **local** axis (never the
sharded one), every other transformed axis carries the full spectrum
on the complexified origin, and the solve's eigenvalue diagonal is
queried on exactly this internal space — honest per-mode values for
negative wavenumbers included, no Hermitian mirroring of sharded
axes. When no local half axis exists (the 2-D real case) the plan
runs fully complex and takes the real part on synthesis.

Fallback conditions (the replicated composite path is kept whenever
any of these fail; the single-device program is bitwise unchanged):

- more than one device on a 1-D device mesh (2-D pencil meshes are
  the documented extension point: a second device axis adds a second
  ``all_to_all`` stage per direction, the geometry below generalizes
  to two transpose partners);
- the resolved transform is a plain unpadded ``Fourier`` with at
  least two stage axes (walled grids compose trig families and stay
  replicated);
- the default layout shards exactly one stage coordinate ``a`` whose
  extent divides the device count, and some other stage coordinate
  ``b`` divides it too (grid sizes divisible by the device count are
  the supported case);
- the elliptic operator is an :class:`Operator` recipe whose
  eigenvalues materialize on the internal coefficient space as an
  endomorphic broadcast-shaped diagonal.

Trace stability: the jit-wrapped ``shard_map`` callables are built
once per plan (plans are memoized per ``(grid, bare space)``; solve
bodies additionally per diagonal broadcast shape), so repeated eager
applications hit jax's tracing cache — a warmed model re-run adds
zero compiles (the ``_ReblockPlan`` idiom of
``decomposition/tensor.py``).
"""
# perf/distributed-slab-fft: slab-decomposed distributed solve
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.spatial.fields.storage import storage_dtype
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.registry import DispatchError
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


# ================================================================
#  The static plan (geometry + cached shard_map callables)
# ================================================================
class SlabPlan:

    r"""
    One grid's slab-decomposed transform/solve pipeline.

    Description
    -----------
    Static structure resolved once per ``(grid, bare space)`` by
    :func:`resolve_slab_plan` (build through it, not this plumbing
    constructor): the device mesh, the slab geometry — sharded
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
        real and a local axis exists), full complexified spectra on
        every other transformed axis. This deliberately differs from
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
#  Resolution (memoized per grid + bare space)
# ================================================================
#: per-grid memo of resolved plans, keyed on the interned bare
#: space (the ``mixed.py`` ``_RESOLVED`` idiom: the
#: WeakKeyDictionary auto-evicts a dropped grid with its memo)
_PLANS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

#: a slab needs a sharded axis and a transpose partner
_MIN_STAGES = 2


def resolve_slab_plan(
    grid: object, space: SpaceLike,
) -> SlabPlan | None:
    """
    Resolve (and memoize) the slab plan of ``space``, or None.

    Description
    -----------
    Returns None whenever the distributed pipeline does not apply —
    see the module docstring's fallback conditions — in which case
    callers keep the replicated path (bitwise-unchanged on a single
    device).

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition and dispatch registry.
    space : SpaceLike
        The (possibly laid-out) nodal operand space.

    Returns
    -------
    SlabPlan | None
        The memoized plan, or None when ineligible.
    """
    bare = space.bare
    memo = _PLANS.setdefault(grid, {})
    if bare not in memo:
        memo[bare] = _build_plan(grid, bare)
    return memo[bare]


def _build_plan(grid: object, bare: SpaceLike) -> SlabPlan | None:
    """Uncached resolution of the slab plan (or None)."""
    decomposition = grid.decomposition
    mesh = getattr(decomposition, "device_mesh", None)
    if mesh is None or decomposition.device_count <= 1:
        return None
    if len(mesh.axis_names) != 1:
        # 2-D pencil meshes: the documented extension point (a
        # second device axis adds a second all_to_all per direction)
        return None
    try:
        transform = resolve_transform(grid, bare)
    except DispatchError:
        return None
    if not isinstance(transform, Fourier) or transform.pad is not None:
        return None
    geometry = _slab_geometry(decomposition, transform, bare)
    if geometry is None:
        return None
    name_a, name_b, name_h, stage_names = geometry
    coeff = _internal_coeff(bare, stage_names, name_h)
    names = bare.names
    return SlabPlan(
        mesh=mesh,
        axis_name=mesh.axis_names[0],
        domain=bare,
        coeff=coeff,
        layout=decomposition.default_layout,
        a=names.index(name_a),
        b=names.index(name_b),
        h=None if name_h is None else names.index(name_h),
        fft_axes=tuple(names.index(n) for n in stage_names),
        real=not jnp.issubdtype(storage_dtype(bare),
                                jnp.complexfloating),
    )


def _slab_geometry(
    decomposition: object,
    transform: Fourier,
    bare: SpaceLike,
) -> tuple[str, str, str | None, tuple[str, ...]] | None:
    """
    Choose the slab axes ``(a, b, h, stage axes)``, or None.

    Description
    -----------
    ``a`` is the single coordinate the default layout shards; ``b``
    is the first other stage coordinate whose extent divides the
    device count (the transpose partner); ``h`` is the last stage
    coordinate that is neither — the local Hermitian axis of a real
    domain (None on complex domains or when no third axis exists).
    Divisibility of ``a`` and ``b`` by the device count is required
    (the ``shard_map`` even-shard contract); staggered surplus
    extents fail it and fall back.
    """
    mapped = dict(decomposition.default_layout.device_axes)
    if len(mapped) != 1:
        return None
    (name_a,) = mapped
    stages = transform.forward_plan(bare).stages
    stage_names = tuple(s.axis for s in stages)
    if len(stage_names) < _MIN_STAGES or name_a not in stage_names:
        return None
    if any(len(bare.factor(n).shape) != 1 for n in stage_names):
        return None
    shards = decomposition.device_count
    if bare.factor(name_a).shape[0] % shards:
        return None
    name_b = next(
        (n for n in stage_names
         if n != name_a and bare.factor(n).shape[0] % shards == 0),
        None)
    if name_b is None:
        return None
    real = not jnp.issubdtype(storage_dtype(bare),
                              jnp.complexfloating)
    local = tuple(n for n in stage_names
                  if n not in (name_a, name_b))
    name_h = local[-1] if (real and local) else None
    return name_a, name_b, name_h, stage_names


def _internal_coeff(
    bare: SpaceLike,
    stage_names: tuple[str, ...],
    half: str | None,
) -> SpaceLike:
    """
    Build the plan's internal coefficient space.

    Description
    -----------
    The half axis keeps its real origin (Hermitian half spectrum);
    every other stage factor targets the complexified origin (full
    spectrum); non-stage factors pass through.
    """
    mapping: dict[str, FunctionSpace] = {}
    for name in stage_names:
        factor = bare.factor(name)
        origin = factor if name == half else factor.as_complex()
        mapping[name] = factor.mesh.fourier(origin=origin)
    factors = tuple(
        mapping.get(factor.names[0], factor)
        if len(factor.names) == 1 else factor
        for factor in bare.factors)
    return TensorProductSpace.of(*factors)
