r"""Operator-sourced analytic eigenmodes of the discrete C-grid.

Description
-----------
The analytic closed-form eigenmodes of the discrete nonhydrostatic
C-grid linear operator, assembled **from the grid's own operator
symbols** instead of hand-coded trigonometric formulas: a
``fr.spatial.GridSymbols`` kit names the staggered component spaces
(``u``, ``v``, ``w``, ``b``, ``p``) once, and every derivative /
interpolation diagonal (``k``, ``kb``, ``a``, ``ab``) is the
corresponding operator's ``eigenvalues`` query on the matching
coefficient space. The dispersion relation is the symbol algebra
(``magnitude ** 2`` quantities composed with a structural-zero
``inverse``), the eigenvector column ``q^s`` is a tag-checked
composition of the same symbols, and the biorthonormal dual is the
derived ``fr.spatial.rayleigh_dual`` under the nonhydro energy metric —
no hand-written left vector and no caller-side masking (the ``k = 0``
mean and the degenerate ``k_h = 0`` wave strata drop through exact
structural zeros).

On the **Nyquist strata of an even grid** the interpolation symbols
hit exact structural zeros and couplings decouple; the steady
family carries the per-stratum eigenvectors of the degenerate plane
operator so the decomposition stays complete there. On a
horizontal-Nyquist plane (``|a_x| |a_y| = 0``) rotation decouples
and the geostrophic column continues as the horizontal
divergence-free mode ``(u, v) = (conj(k_y), -conj(k_x))`` with
``w = b = 0``; on the doubly degenerate strata (horizontal Nyquist
and ``|a_z| = 0``, where buoyancy decouples too and the wave pair
degenerates to ``omega = 0``) the vortical family additionally
holds the steady overturning mode ``(k_x kb_z, k_y kb_z,
k_h^2, 0)`` and the pure-buoyancy mode ``(0, 0, 0, 1)`` as extra
internal projector columns (``em.projector(0)`` /
``em.function(f, 0)`` cover them; ``em.q(0)`` / ``em.mode(0, ...)``
expose the primary column). Odd grids have no Nyquist strata and
are bitwise unaffected.

On a **walled** (bounded, rigid-lid) vertical the kit spaces carry
the physics-fixed parity tags (``w`` Dirichlet, ``u``/``v``/``p``
Neumann, ``b`` Dirichlet), the same formulas compose under the
derived-shift trig symbol algebra, and cross-component sums align
per physical vertical mode through the ``fr.spatial.ModeChart`` union
lattice — the per-component trig families hold different mode
ranges (``w`` modes ``1..n-1``, ``u``/``v``/``p`` ``0..n-1``, ``b``
``1..n``). The geostrophic column is re-referenced per component so
the ``m = 0`` barotropic and ``m = n`` buoyancy-top strata land in
the vortical family (``N + 1`` steady modes per horizontal
wavevector).

Surface: ``em.omega(s)`` returns the frequency ``Symbol`` (``.data``
for the half-spectrum array), ``em.q(s)`` the eigenvector as a
coefficient-space :class:`~fridom.nonhydro2.state.State`, and
``em.projector(s)`` a ``State -> State`` callable on coefficient
states satisfying ``L q^s = -i \omega^s q^s`` for the linearized,
Leray-projected tendency (the oceanographic sign convention: the
mode evolves as :math:`e^{i(kx - \omega t)}`, so positive ``omega``
propagates along ``+k``). ``em.grid`` and ``em.kit`` expose the grid
and the per-component transform kit — the ``nh.transforms``
projection surface for the physical round-trip.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.model.eigenstates import (
    assemble_operator_matrix,
    bare_coeff_field,
    coefficient_index,
    describe_nonfinite_branch,
    envelope_scale,
    evaluate_frequency_function,
    hermitian_mode_data,
    resolve_mode_branches,
)
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.energy import nonhydro_energy_weights
from fridom.nonhydro2.params import DSQR
from fridom.nonhydro2.state import State
from fridom.spatial.bc import BC
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.symbols import (
    GridSymbols,
    ModeChart,
    rayleigh_dual,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Iterator

    import jax

    from fridom.model.model import Model
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.meshes.mesh import Mesh
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _fv_tagged_vertical(
    space: SpaceLike, vertical: str, bc: BC,
) -> SpaceLike:
    r"""Mint the BC-tagged average sibling of an FV analysis space.

    Description
    -----------
    The walled-vertical FV eigenmode kit needs BC-tagged vertical
    factors for its trig transforms, but C8 forbids a ``wall_bc`` on an
    FV collocated (average) declaration — average factors are BC-free,
    so the declared-space resolver rejects the pattern. Instead the kit
    resolves the pattern **BC-free** and swaps the bounded vertical
    factor for its tagged sibling here, through the same
    ``mesh.average(kind, bc=...)`` factory the walled FV pressure solve
    uses (``modules.pressure._neumann_sibling``, F4). Applied only to a
    resolved **average** vertical factor of a **bounded** mesh: a
    periodic vertical (a fully periodic FV grid) is BC-free and passes
    through untouched (the kit spaces stay the F3 periodic ones), and
    the ``w`` face leg (``Inner``, Dirichlet) rides its STAGGERED
    coordinate through the pattern resolver and never reaches here.

    Parameters
    ----------
    space : SpaceLike
        The BC-free resolved analysis space (a bare product).
    vertical : str
        The vertical coordinate name.
    bc : BC
        The physics-fixed vertical wall-value claim (Neumann for
        ``u``/``v``/``p``, Dirichlet for ``b``).

    Returns
    -------
    SpaceLike
        ``space`` with its bounded-average vertical factor BC-tagged;
        ``space`` unchanged on a periodic / nodal vertical.
    """
    factor = space.factor(vertical)
    if (isinstance(factor, AverageSpace)
            and not getattr(factor.mesh, "periodic", False)):
        return space.replace(
            **{vertical: factor.mesh.average(type(factor), bc=bc)})
    return space


class _LazySymbols(Mapping):

    """
    Axis-keyed symbol family, built on first access (memoized).

    Description
    -----------
    The public ``k``/``kb``/``a``/``ab`` surface stays dict-like,
    but entries materialize lazily: on a walled grid the
    interpolation threaded on the DCT-II pressure factor (``a[z]``)
    raises the eigen-layer skip signal by design and is unused by
    the dispersion / eigenvector formulas — it must never be built
    eagerly.

    Parameters
    ----------
    axes : tuple[str, ...]
        The coordinate names keyed by the mapping.
    build : Callable[[str], Symbol]
        The per-axis symbol builder (called once per axis).
    """

    def __init__(
        self, axes: tuple[str, ...],
        build: Callable[[str], Symbol],
    ) -> None:
        """Store the axis keys and the memoized builder."""
        self._axes: tuple[str, ...] = axes
        self._build: Callable[[str], Symbol] = build
        self._cache: dict[str, Symbol] = {}

    def __getitem__(self, axis: str) -> Symbol:
        """Build (once) and return the symbol along ``axis``."""
        if axis not in self._cache:
            if axis not in self._axes:
                raise KeyError(axis)
            self._cache[axis] = self._build(axis)
        return self._cache[axis]

    def __iter__(self) -> Iterator[str]:
        """Iterate the axis keys."""
        return iter(self._axes)

    def __len__(self) -> int:
        """Count the axis keys."""
        return len(self._axes)


class Eigenmodes:

    r"""Discrete inertia-gravity / geostrophic eigenmodes on a grid.

    Description
    -----------
    Binds a :class:`~fridom.spatial.symbols.GridSymbols` kit
    on the canonical C-grid component spaces and exposes the analytic
    eigenmode surface: the dispersion ``omega(s)`` (a ``Symbol``),
    the eigenvector column ``q(s)`` (a coefficient-space ``State``),
    and the ``projector(s)`` closure (coefficient ``State ->
    State``). The per-axis symbol families are public lazy mappings:
    ``k`` (centre -> face derivative), ``kb`` (face -> centre
    derivative), ``a`` (centre -> face interpolation) and ``ab``
    (face -> centre interpolation), keyed by coordinate name.

    Parameters
    ----------
    grid : fr.spatial.Grid
        A 3-D grid carrying the vertical coordinate; the horizontal
        axes must be periodic, the vertical may be bounded (rigid
        lids).
    f0 : float
        The (constant) Coriolis parameter.
    n2 : float
        The (constant) squared buoyancy frequency.
    dsqr : float
        The squared aspect ratio.
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    family : str | None, optional
        The discretization family the component spaces resolve into
        (FV-D3, stage F3): ``"fv"`` builds the analysis spaces on the
        finite-volume C-grid (scalars on ``CellAvg``, velocities on
        the faces), matching an FV nonhydro model; ``"nodal"`` the
        point-value C-grid. ``None`` defers to the grid default. On a
        **periodic** FV grid the ``GridSymbols`` kit resolves ``diff``
        / ``interp`` through the grid's FV C-grid profile
        (``FaceDifference`` / ``FluxDifference``), whose symbols are
        bit-identical to the nodal ones (scoping study §1) — so
        ``from_model`` on a periodic FV model builds FV-consistent
        eigenmodes without new numerics. On a **walled** FV grid the
        analytic kit builds its BC-tagged analysis spaces by minting
        the tagged vertical siblings itself (``_fv_tagged_vertical``,
        the pressure solver's ``mesh.average(kind, bc=...)`` seam) —
        C8 keeps the *declaration* layer BC-free, so no
        ``Collocated(wall_bc=..., family="fv")`` pattern is used. The
        vertical trig symbols are the FV C-grid staggering diagonals
        (``FaceDifference`` / ``FluxDifference`` for the derivatives,
        ``LinearReconstruction`` for the interpolations), which at
        second order are **bitwise** the nodal ``Center``-family ones
        (the FV stencils are the nodal ones), so the walled FV
        eigenbasis is bit-identical to the walled nodal one. Build the
        model ``family="nodal"`` for the point-value C-grid instead
        (default: None).
    """

    def __init__(
        self, grid: Grid, *, f0: float, n2: float, dsqr: float,
        vertical: str = "z", family: str | None = None,
    ) -> None:
        """Build the symbol kit and the per-axis operator diagonals."""
        if float(f0) == 0.0 and float(n2) == 0.0:
            raise ValueError(
                "an eigenmode set needs f0 != 0 or n2 != 0 "
                "(both zero has no inertia-gravity structure)")
        self.f0 = float(f0)
        self.n2 = float(n2)
        self.dsqr = float(dsqr)

        names = grid.names
        if vertical not in names or len(names) != 3:  # noqa: PLR2004
            raise ValueError(
                "eigenmodes need a 3-D grid with the vertical "
                f"coordinate {vertical!r}; got names {names}")
        x, y = (n for n in names if n != vertical)
        z = vertical
        self._axes: tuple[str, str, str] = (x, y, z)
        self._grid: Grid = grid
        self._mz: Mesh = next(
            m for m in grid.factors if z in m.names)
        self._walled: bool = not getattr(self._mz, "periodic", False)
        self._chart: ModeChart = ModeChart(grid)

        # analysis (kit) spaces: the eigenmode physics fixes the
        # vertical parity of every component on a walled grid
        # (impermeable w -> Dirichlet, free-slip u/v and pressure ->
        # Neumann, buoyancy -> Dirichlet); wall_bc entries are
        # ignored on periodic factors, so a periodic grid resolves
        # to the exact BC-free spaces. On the **FV** family the
        # C8 rule forbids a wall_bc on the collocated (average) u/v/b/p
        # declarations (average factors are BC-free); the kit instead
        # mints the BC-tagged CellAvg siblings after a BC-free resolve
        # (_fv_tagged_vertical, the pressure solver's mesh.average(kind,
        # bc=...) precedent). w's Dirichlet rides its STAGGERED z
        # coordinate onto the Inner face, so its leg resolves through
        # the pattern unchanged on both families.
        eff_family = (family if family is not None
                      else getattr(grid, "default_family", "nodal"))
        if eff_family == "fv":
            spaces = {
                "u": _fv_tagged_vertical(fr.spatial.Staggered(
                    x, family=family).resolve(grid), z, BC.NEUMANN),
                "v": _fv_tagged_vertical(fr.spatial.Staggered(
                    y, family=family).resolve(grid), z, BC.NEUMANN),
                "w": fr.spatial.Staggered(
                    z, wall_bc={z: BC.DIRICHLET},
                    family=family).resolve(grid),
                "b": _fv_tagged_vertical(fr.spatial.Collocated(
                    family=family).resolve(grid), z, BC.DIRICHLET),
                "p": _fv_tagged_vertical(fr.spatial.Collocated(
                    family=family).resolve(grid), z, BC.NEUMANN),
            }
        else:
            spaces = {
                "u": fr.spatial.Staggered(
                    x, wall_bc={z: BC.NEUMANN}, family=family).resolve(grid),
                "v": fr.spatial.Staggered(
                    y, wall_bc={z: BC.NEUMANN}, family=family).resolve(grid),
                "w": fr.spatial.Staggered(
                    z, wall_bc={z: BC.DIRICHLET}, family=family).resolve(grid),
                "b": fr.spatial.Collocated(
                    wall_bc={z: BC.DIRICHLET}, family=family).resolve(grid),
                "p": fr.spatial.Collocated(
                    wall_bc={z: BC.NEUMANN}, family=family).resolve(grid),
            }
        # the model-facing physical spaces (u, v, b BC-free; w's
        # Dirichlet wall tag matches the Velocity declaration) —
        # identical to the kit spaces on a periodic grid
        self._physical: dict[str, SpaceLike] = {
            "u": fr.spatial.Staggered(x, family=family).resolve(grid),
            "v": fr.spatial.Staggered(y, family=family).resolve(grid),
            "w": spaces["w"],
            "b": fr.spatial.Collocated(family=family).resolve(grid),
        }
        kit = GridSymbols(grid, spaces)
        self._kit: GridSymbols = kit
        #: the (parity-tagged) analysis spaces the kit threads -- the
        #: frame hook rebuilds the symbol kit on the distributed internal
        #: coefficient frame from these (see :meth:`_reframe`)
        self._analysis: dict[str, SpaceLike] = spaces
        #: the prognostic component order the matrix stacks
        self._components: tuple[str, ...] = ("u", "v", "w", "b")
        #: True on a frame clone built for the distributed matrix route:
        #: switches the host ``np.any`` Nyquist gates to unconditional
        #: jnp masks (:meth:`_extend_nyquist_steady`, :meth:`_columns`) so
        #: the assembled matrix is shard-safe (build unconditionally)
        self._distributed_matrix: bool = False
        face = {x: "u", y: "v", z: "w"}
        axes = (x, y, z)
        self.k: Mapping[str, Symbol] = _LazySymbols(
            axes, lambda n: kit.diff(n, on="p"))
        self.kb: Mapping[str, Symbol] = _LazySymbols(
            axes, lambda n: kit.diff(n, on=face[n]))
        self.a: Mapping[str, Symbol] = _LazySymbols(
            axes, lambda n: kit.interp(n, on="p"))
        self.ab: Mapping[str, Symbol] = _LazySymbols(
            axes, lambda n: kit.interp(n, on=face[n]))
        # coefficient-space zero fields (shape / dtype / wrap templates);
        # built directly on the coefficient space -- NOT by forward-
        # transforming a nodal field, which would hit the Tier-1 taught
        # error on a grid that shards a transform axis (the analytic
        # surface must build on a sharded grid for the distributed route)
        self._templates: dict[str, ScalarField] = {
            c: grid.create_field(kit.coeff(c)).with_metadata(name=c)
            for c in ("u", "v", "w", "b")}

    # ================================================================
    #  Accessors (the wave-7 ``nh.transforms`` projection surface)
    # ================================================================
    @property
    def grid(self) -> Grid:
        """The grid the eigenmode set is built on."""
        return self._grid

    @property
    def kit(self) -> GridSymbols:
        """The per-component transform kit (``forward``/``backward``)."""
        return self._kit

    def physical_space(self, name: str) -> SpaceLike:
        """
        Model-facing physical space of a prognostic component.

        Description
        -----------
        The bare space the *model* resolves for the component —
        BC-free for ``u``/``v``/``b``, the Dirichlet wall tag for
        ``w`` — as opposed to the parity-tagged analysis space the
        kit transforms on. On a periodic grid the two coincide.

        Parameters
        ----------
        name : str
            The component name (``u``, ``v``, ``w`` or ``b``).

        Returns
        -------
        SpaceLike
            The bare physical space.
        """
        return self._physical[name]

    # ================================================================
    #  Dispersion
    # ================================================================
    def _omega2(self) -> Symbol:
        r"""Squared dispersion symbol (w-referenced, k = 0 exact).

        Description
        -----------
        The discrete relation

        .. math::

            \omega^2 = \frac{f_0^2\,|\hat a_x|^2 |\hat a_y|^2
                             |\hat k_z|^2
                             + N^2\,|\hat a_z|^2 \hat k_h^2}
                            {\delta^2 \hat k_h^2 + |\hat k_z|^2}

        assembled from the operator symbols' ``magnitude ** 2``
        quantities on ``w``'s coefficient space. The ``k = 0`` mean
        mode drops structurally: the numerator and denominator are
        both exact zeros there, and ``Symbol.inverse`` maps the
        structural zero to zero (no caller-side masking). On a
        walled vertical the same composition holds with the trig
        tables ``|\hat k_z| = 2 \sin(\pi m \Delta z / (2 L_z)) /
        \Delta z`` and ``|\hat a_z| = \cos(\pi m \Delta z /
        (2 L_z))`` on ``w``'s DST-I mode lattice ``m = 1..n-1``.

        Returns
        -------
        Symbol
            The real, non-negative ``omega ** 2`` diagonal.
        """
        x, y, z = self._axes
        kh2 = self.k[x].magnitude ** 2 + self.k[y].magnitude ** 2
        coriolis = self.f0 ** 2 * (
            self.a[x].magnitude ** 2 * self.a[y].magnitude ** 2
            * self.kb[z].magnitude ** 2)
        buoyancy = self.n2 * (self.ab[z].magnitude ** 2 * kh2)
        denom = self.dsqr * kh2 + self.kb[z].magnitude ** 2
        return (coriolis + buoyancy) @ denom.inverse()

    def omega(self, s: int = 1) -> Symbol:
        r"""Discrete frequency symbol ``omega^s`` (a ``Symbol``).

        Description
        -----------
        ``s = 0`` is geostrophic (zero frequency); ``s = +/-1`` are
        the inertia-gravity branches ``s * sqrt(omega ** 2)``. The
        diagonal lives on the grid's real-FFT coefficient layout
        (first transformed axis half-spectrum); read the array off
        ``.data`` (broadcast-shaped).

        Parameters
        ----------
        s : int, optional
            The mode branch: 0, +1 or -1 (default: 1).

        Returns
        -------
        Symbol
            The real frequency diagonal.
        """
        om2 = self._omega2()
        return 0.0 * om2 if s == 0 else float(s) * om2.sqrt()

    # ================================================================
    #  Eigenvectors and the projector
    # ================================================================
    def _vec_q(self, s: int) -> dict[str, Symbol]:
        r"""Eigenvector column ``q^s`` as per-component ``Symbol``s.

        Description
        -----------
        Tag-checked operator compositions; each entry's codomain is
        the component's own coefficient space. The degenerate modes
        (``k_h = 0`` for ``s != 0`` and the ``k = 0`` mean) are
        **exact** structural zeros of every entry, so the Rayleigh
        dual vanishes there with no masking. The ``s = 0`` column is
        extended over the even-grid Nyquist strata, where the
        composed entries are exact structural zeros while the
        rotation-decoupled plane operator keeps steady modes (see
        :meth:`_extend_nyquist_steady` and :meth:`_columns`).

        The ``s != 0`` column is built on its own dispersion root
        ``omega(s)``: the linearized tendency satisfies
        ``L q = -i omega q`` for the column written with ``omega``
        — the :math:`e^{i(kx - \omega t)}` convention asserted by
        the tests, positive ``omega`` propagating along ``+k``. On a
        walled vertical the common domain is ``w``'s DST-I lattice
        and the entries retag onto each component's own trig family
        through the derived-shift algebra.

        The ``s = 0`` column on a walled vertical is re-referenced
        per component (each entry endo on its **own** vertical mode
        lattice): the w-referenced column would structurally miss
        the ``m = 0`` barotropic and ``m = n`` buoyancy-top strata,
        which belong to the steady family under rigid lids. The
        interp magnitude table ``cos(k dz/2)`` (``C``) and the
        staggered-derivative magnitude table ``2 sin(k dz/2)/dz``
        (``K``) reproduce the w-referenced column on the interior
        modes up to a positive per-mode scale (projector-invariant)
        and extend it to all ``N + 1`` strata with one formula.
        """
        x, y, z = self._axes
        k, kb, a, ab = self.k, self.kb, self.a, self.ab
        if s == 0:
            if self._walled:
                # discrete thermal wind: b sits on the sine lattice
                # of d(cos)/dz, whose derivative sign is -k_hat
                # (cos -> sin), hence the minus on the b entry
                return self._extend_nyquist_steady({
                    "u": -(a[x] @ (ab[y] @ k[y]))
                    * self._interp_table("u"),
                    "v": (a[y] @ (ab[x] @ k[x]))
                    * self._interp_table("v"),
                    "w": self.omega(0),
                    "b": -self.f0 * (a[x].magnitude ** 2
                                     * a[y].magnitude ** 2
                                     * self._diff_table("b")),
                })
            return self._extend_nyquist_steady({
                "u": -(a[x] @ (ab[y] @ k[y]) @ ab[z]),
                "v": a[y] @ (ab[x] @ k[x]) @ ab[z],
                "w": self.omega(0),
                "b": self.f0 * (a[x].magnitude ** 2
                                * a[y].magnitude ** 2 * kb[z]),
            })
        om = self.omega(s)
        kh2 = k[x].magnitude ** 2 + k[y].magnitude ** 2
        return {
            "u": kb[z] @ (k[x] @ om
                          + 1j * self.f0 * (a[x] @ (ab[y] @ k[y]))),
            "v": kb[z] @ (k[y] @ om
                          - 1j * self.f0 * (a[y] @ (ab[x] @ k[x]))),
            "w": om @ kh2,
            "b": -1j * self.n2 * (ab[z] @ kh2),
        }

    def _horizontal_nyquist_mask(self) -> jax.Array:
        r"""Boolean mask of the horizontal interpolation-Nyquist strata.

        The exact structural-zero set of ``|a_x|^2 |a_y|^2`` — the
        planes where the horizontal staggering averages vanish and
        rotation decouples; empty on odd grid sizes.
        """
        x, y = self._axes[:2]
        return (self.a[x].magnitude ** 2
                * self.a[y].magnitude ** 2).data == 0

    def _extend_nyquist_steady(
        self, column: dict[str, Symbol],
    ) -> dict[str, Symbol]:
        r"""Merge the Nyquist steady strata into the geostrophic column.

        Description
        -----------
        On the horizontal-Nyquist planes of an even grid rotation
        decouples (``|a_x| |a_y|`` is an exact structural zero) and
        every composed geostrophic entry vanishes, while the
        constrained plane operator keeps the steady horizontal
        divergence-free mode

        .. math::

            (u, v, w, b) = (\overline{\hat k_y},
            -\overline{\hat k_x}, 0, 0)

        (``kb_x u + kb_y v = 0`` exactly since ``conj(k) = -kb`` in
        the forward/backward difference pair). The patch writes that
        value exactly on the structural-zero stratum — disjoint from
        the composed column's support. On a **walled** vertical the
        merge additionally covers the buoyancy-top stratum at
        horizontal Nyquist (the union mode ``m = n`` the interp
        ``b -> w`` annihilates exactly): the pure-buoyancy steady
        mode ``(0, 0, 0, 1)`` — on the union lattice it is disjoint
        from the ``u``/``v`` strata, so one column carries both. On
        a periodic vertical the analogous ``|a_z| = 0`` strata
        overlap the horizontal mode pointwise and live in separate
        internal columns instead (see :meth:`_columns`). Odd grids
        have no Nyquist strata: the column is returned untouched
        (bitwise the pre-Nyquist path).

        Parameters
        ----------
        column : dict[str, Symbol]
            The composed geostrophic column.

        Returns
        -------
        dict[str, Symbol]
            The column with the steady Nyquist strata merged.
        """
        mask = self._horizontal_nyquist_mask()
        # a frame clone for the distributed matrix builds the supplement
        # unconditionally (the mask is exact-zero off the strata, so the
        # merge is a no-op on odd grids) -- the host np.any gate would
        # gather the sharded internal-frame mask
        if not self._distributed_matrix and not bool(
                np.any(np.asarray(mask))):
            return column
        x, y, z = self._axes
        supplement = {
            "u": jnp.where(mask, jnp.conj(self.k[y].data), 0.0),
            "v": jnp.where(mask, -jnp.conj(self.k[x].data), 0.0),
        }
        if self._walled:
            top = (self._kit.interp(z, on="b").magnitude ** 2
                   ).data == 0
            supplement["b"] = jnp.where(
                mask & top, 1.0 + 0.0j, 0.0)
        return {
            c: (Symbol(sym.space, sym.data + supplement[c],
                       codomain=sym.codomain)
                if c in supplement else sym)
            for c, sym in column.items()}

    def _columns(self, s: int) -> tuple[dict[str, Symbol], ...]:
        r"""Eigenvector columns of a branch (the projector family).

        Description
        -----------
        Every branch is a single column except the vortical family
        of an even **periodic-vertical** grid: on the doubly
        degenerate Nyquist strata (horizontal Nyquist and
        ``|a_z| = 0``, where the wave pair degenerates to
        ``omega = 0``) the steady eigenspace is three-dimensional,
        so ``s = 0`` carries two extra internal columns there — the
        steady overturning mode ``(k_x kb_z, k_y kb_z, k_h^2, 0)``
        (the ``omega -> 0`` limit of the wave velocity polarization,
        exactly divergence-free) and the decoupled pure-buoyancy
        mode ``(0, 0, 0, 1)``. All three are mutually M-orthogonal
        on the shared stratum, so the summed rank-1 projectors stay
        an orthogonal projector. On odd grids (empty strata) and on
        the walled vertical (no ``w`` Nyquist mode; the buoyancy-top
        stratum merges into the main column) the family is the
        single main column — bitwise the pre-Nyquist path.

        Parameters
        ----------
        s : int
            The mode branch: 0, +1 or -1.

        Returns
        -------
        tuple[dict[str, Symbol], ...]
            The column family of the branch.
        """
        main = self._vec_q(s)
        if s != 0 or self._walled:
            return (main,)
        x, y, z = self._axes
        mask = (self._horizontal_nyquist_mask()
                & ((self.ab[z].magnitude ** 2).data == 0))
        if not self._distributed_matrix and not bool(
                np.any(np.asarray(mask))):
            return (main,)
        k, kb, ab = self.k, self.kb, self.ab
        kh2 = k[x].magnitude ** 2 + k[y].magnitude ** 2
        overturning = {
            "u": kb[z] @ k[x],
            "v": kb[z] @ k[y],
            "w": 1.0 * kh2,
            "b": 0.0 * (ab[z] @ kh2),
        }
        buoyancy = {
            "u": 0.0 * (kb[z] @ k[x]),
            "v": 0.0 * (kb[z] @ k[y]),
            "w": 0.0 * kh2,
            "b": 1.0 + 0.0 * (ab[z] @ kh2),
        }
        masked = tuple(
            {c: Symbol(sym.space, jnp.where(mask, sym.data, 0.0),
                       codomain=sym.codomain)
             for c, sym in col.items()}
            for col in (overturning, buoyancy))
        return (main, *masked)

    def q(self, s: int = 1) -> State:
        r"""Eigenvector ``q^s`` as a coefficient-space ``State``.

        Description
        -----------
        Each component symbol's diagonal is wrapped on that
        component's own coefficient space (the kit's per-component
        transform codomain), broadcast to the full spectral shape.
        This is the **primary** column of the branch; the extra
        internal steady columns of the even-grid vortical family
        (see :meth:`_columns`) are reachable through
        :meth:`projector` / :meth:`function` only.

        Parameters
        ----------
        s : int, optional
            The mode branch: 0, +1 or -1 (default: 1).

        Returns
        -------
        State
            The ``(u, v, w, b)`` coefficient-space eigenvector.
        """
        syms = self._vec_q(s)
        return State({c: self._wrap(c, syms[c].data) for c in syms})

    def projector(self, s: int = 1) -> Callable[[State], State]:
        r"""Return the spectral projector ``P^s`` on coefficient states.

        Description
        -----------
        ``P^s z = q^s \langle p^s, z\rangle`` with the biorthonormal
        dual ``p^s`` derived from ``q^s`` under the nonhydro energy
        metric (``fr.spatial.rayleigh_dual`` + the ``diag(1, 1, dsqr,
        1/N^2)`` weights): idempotent by biorthonormality, exactly
        zero on the structurally degenerate modes. On a walled
        vertical the amplitude is accumulated on the
        ``fr.spatial.ModeChart`` union mode lattice (the components'
        trig families hold different mode ranges); on a periodic
        grid the chart is identity and the data path is unchanged.
        The vortical branch of an even periodic-vertical grid sums
        the rank-1 projectors of its M-orthogonal internal column
        family (see :meth:`_columns`) — still an orthogonal
        projector.

        Parameters
        ----------
        s : int, optional
            The mode branch: 0, +1 or -1 (default: 1).

        Returns
        -------
        Callable[[State], State]
            The projection acting on coefficient-space states.
        """
        columns = self._columns(s)
        terms = [(q, self._dual(q, s)) for q in columns]
        chart = self._chart
        coeff = {c: self._kit.coeff(c) for c in columns[0]}

        def project(z: State) -> State:
            """Project ``z`` onto mode ``s`` (pointwise per mode)."""
            out: dict[str, jax.Array] | None = None
            for q, p in terms:
                amp = sum(chart.embed(jnp.conj(p[c]) * z[c].data,
                                      coeff[c]) for c in p)
                part = {c: q[c].data
                        * chart.restrict(amp, coeff[c]) for c in q}
                out = (part if out is None
                       else {c: out[c] + part[c] for c in part})
            return State({c: self._wrap(c, out[c]) for c in out})

        return project

    def function(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        s: int | Iterable[int] = 1,
    ) -> Callable[[State], State]:
        r"""
        Apply a scalar function of the linear operator on branches.

        Description
        -----------
        The general :math:`f(L)` applicator on coefficient states:

        .. math::

            \sum_s P^s\, f(\omega^s)

        with :math:`P^s` the branch projector of :meth:`projector`
        and :math:`\omega^s` the branch's pointwise dispersion
        diagonal on the ``fr.spatial.ModeChart`` union mode lattice —
        ``f = 1`` on a selection reproduces the summed projectors
        exactly. A branch with an internal column family (the
        vortical branch of an even periodic-vertical grid, see
        :meth:`_columns`) weights every column with the same
        ``f(omega^s)`` diagonal on its own represented set. ``s``
        is a single branch or an iterable of distinct
        branches from ``{0, +1, -1}``. ``f`` is evaluated once,
        host-side, on the **real** frequencies of the branch's
        *represented* modes only — the structural zeros of the
        column (the ``k = 0`` mean, the ``k_h = 0`` wave strata,
        the walled strata a component family lacks) never reach
        ``f``; complex return values are allowed; the columns
        satisfy ``L q = -i omega q``:
        ``f = lambda w: -1.0 / (1j * w)`` builds :math:`L^{-1}` on
        the selection, ``f = lambda w: -1j * w`` the forward
        operator. A
        singular ``f`` meeting a structurally represented zero
        frequency (the geostrophic branch ``s = 0``) is a taught
        ``ValueError``, never a floored division.

        Real-safety: for the conjugation-closed wave selection
        ``s = (1, -1)`` and ``f`` satisfying
        :math:`f(-\omega) = \overline{f(\omega)}` (true for
        :math:`-1/(i\omega)` and :math:`-i\omega`) the map sends
        Hermitian (real-state) coefficients to Hermitian
        coefficients — the backward synthesis stays real.

        Parameters
        ----------
        f : Callable[[np.ndarray], np.ndarray]
            The scalar spectral function, vectorized over an array
            of real frequencies.
        s : int | Iterable[int], optional
            The branch selection: 0, +1, -1 or an iterable of
            distinct branches (default: 1).

        Returns
        -------
        Callable[[State], State]
            The weighted application on coefficient-space states.

        Raises
        ------
        ValueError
            On an invalid branch selection, or if ``f`` evaluates
            non-finite on a represented mode of the selection.
        """
        branches = resolve_mode_branches(s)
        chart = self._chart
        metric = self._energy_weights()
        terms = []
        for b in branches:
            for q in self._columns(b):
                p = self._dual(q, b)
                coeff = {c: self._kit.coeff(c) for c in q}
                norm = sum(
                    chart.embed(jnp.broadcast_to(
                        metric[c] * jnp.abs(q[c].data) ** 2,
                        self._templates[c].data.shape), coeff[c])
                    for c in q)
                om_w = jnp.broadcast_to(
                    jnp.real(jnp.asarray(self.omega(b).data)),
                    self._templates["w"].data.shape)
                omega = np.asarray(chart.embed(om_w, coeff["w"]))
                weights = jnp.asarray(evaluate_frequency_function(
                    f, omega, np.asarray(norm) != 0,
                    lambda bad, b=b, om=omega:
                    describe_nonfinite_branch(b, om, bad)))
                terms.append((q, p, coeff, weights))

        def apply(z: State) -> State:
            """Apply ``sum_s P^s f(omega^s)`` (pointwise per mode)."""
            out: dict[str, ScalarField] | None = None
            for q, p, coeff, w in terms:
                amp = w * sum(
                    chart.embed(jnp.conj(p[c]) * z[c].data,
                                coeff[c]) for c in p)
                part = {
                    c: self._wrap(c, q[c].data
                                  * chart.restrict(amp, coeff[c]))
                    for c in q}
                out = (part if out is None
                       else {c: out[c] + part[c] for c in part})
            return State(out)

        return apply

    # ================================================================
    #  Mode-indexed single-mode states
    # ================================================================
    def mode(
        self,
        s: int,
        indices: Mapping[str, int],
        *,
        phase: float = 0.0,
    ) -> tuple[float, State]:
        r"""
        Return one discrete mode as ``(omega, physical state)``.

        Description
        -----------
        The mode-indexed accessor of the analytic eigenmodes:
        ``indices`` is an axis-keyed mapping of integer mode
        indices (e.g. ``{"x": 3, "y": 0, "z": 2}``) — the
        half-spectrum axis runs ``0..n//2``, full-spectrum axes
        take any integer modulo ``n``, and a walled vertical takes
        the **physical** vertical mode on the ``0..n`` union
        lattice (components whose trig family lacks the stratum
        contribute exact zeros). The state is the real
        Hermitian-closed physical mode
        :math:`\mathrm{Re}(q^s(k)\,e^{i(k\cdot x - \mathrm{phase})})`
        satisfying ``d/dt state(phase) = omega * state(phase +
        pi/2)`` under the linearized, Leray-projected tendency
        (the state at time ``t`` is the same mode at phase
        ``phase + omega * t``: positive ``omega`` propagates along
        ``+k``),
        normalized so the largest horizontal-velocity amplitude
        (the pointwise oscillation envelope over the ``u`` and
        ``v`` nodes) is one; a mode without horizontal velocity is
        left unnormalized. On the self-conjugate planes of the
        half-spectrum axis a wave branch synthesizes the standing
        (conjugate-mixed) real mode.

        Parameters
        ----------
        s : int
            The mode branch: 0, +1 or -1.
        indices : Mapping[str, int]
            Axis-keyed integer mode indices, one per grid axis.
        phase : float, optional
            The mode phase shift (default: 0.0).

        Returns
        -------
        tuple[float, State]
            The frequency and the single-mode physical state.

        Raises
        ------
        ValueError
            On bad indices, or a structurally unrepresented mode
            (e.g. a wave branch on a vortical-only stratum: the
            ``k_h = 0`` columns, the walled barotropic ``m = 0``
            and buoyancy-top ``m = n`` strata, the doubly
            degenerate Nyquist strata).
        """
        components = ("u", "v", "w", "b")
        q = self.q(s)
        slots = {c: coefficient_index(q[c].function_space, indices)
                 for c in components}
        amps = {c: q[c].data[slots[c]] for c in components
                if slots[c] is not None}
        if (s != 0 and slots["w"] is None) or all(
                float(jnp.abs(a)) == 0.0 for a in amps.values()):
            raise ValueError(
                f"mode s={s} at {dict(indices)!r} is structurally "
                "unrepresented on the discrete lattice (wave "
                "branches vanish at k_h = 0, outside the vertical "
                "w strata, and on the doubly degenerate Nyquist "
                "strata; the geostrophic column vanishes at the "
                "k = 0 mean)")

        def synth(shift: float) -> dict[str, ScalarField]:
            out = {}
            for c in components:
                if slots[c] is None:
                    data = jnp.zeros_like(q[c].data)
                else:
                    value = amps[c] * jnp.exp(
                        -1j * (float(phase) + shift))
                    data = hermitian_mode_data(
                        q[c].function_space, slots[c], value)
                # the Hermitian mirror pair is already placed in ``data``
                # (host-side, before any region); on a multi-device grid
                # rebuild on the bare coefficient space so the backward
                # runs unguarded (replicated, device invariant) instead
                # of tripping the Tier-1 error -- single device keeps the
                # original path bitwise
                field = (
                    bare_coeff_field(
                        self._grid, self._grid.decomposition,
                        q[c].function_space.bare, q[c], data)
                    if getattr(self._grid.decomposition,
                               "device_count", 1) > 1
                    else q[c].with_data(data))
                out[c] = self._kit.backward(c)(field).real.retag(
                    self._physical[c])
            return out

        z0 = synth(0.0)
        z1 = synth(jnp.pi / 2.0)
        scale = envelope_scale(z0, z1, ("u", "v"))
        state = State({c: z0[c] / scale for c in components})
        if s == 0:
            return 0.0, state
        shape = self._templates["w"].data.shape
        omega = float(jnp.broadcast_to(
            self.omega(s).data, shape)[slots["w"]])
        return omega, state

    # ================================================================
    #  Internals
    # ================================================================
    def _dual(
        self, q: dict[str, Symbol], s: int,
    ) -> dict[str, jax.Array]:
        r"""Biorthonormal dual diagonals of a column, as data arrays.

        Description
        -----------
        The Rayleigh dual under the energy metric. The wave columns
        (and every periodic column) share one domain lattice, so
        ``fr.spatial.rayleigh_dual`` applies in the symbol algebra.
        The walled geostrophic column is per-component endo (each
        entry on its own vertical lattice), so its norm is
        accumulated on the union mode lattice instead — the same
        pseudo-inverse with the same exact structural-zero
        regularization, chart-aligned like ``q``.
        """
        weights = self._energy_weights()
        if not (s == 0 and self._walled):
            dual = rayleigh_dual(q, weights)
            return {c: dual[c].data for c in q}
        chart = self._chart
        coeff = {c: self._kit.coeff(c) for c in q}
        norm = sum(
            chart.embed(weights[c]
                        * jnp.real(jnp.conj(q[c].data) * q[c].data),
                        coeff[c])
            for c in q)
        zero = norm == 0
        inv = jnp.where(zero, 0.0,
                        1.0 / jnp.where(zero, jnp.ones_like(norm),
                                        norm))
        return {c: weights[c] * q[c].data
                * chart.restrict(inv, coeff[c]) for c in q}

    # ================================================================
    #  Frame hook (the distributed matrix route)
    # ================================================================
    def _reframe(
        self, coeff_of: Callable[[str], SpaceLike],
    ) -> Eigenmodes:
        r"""
        Return a shallow clone reading a different coefficient frame.

        Description
        -----------
        The frame hook: rebuilds the ``GridSymbols`` kit (and the
        ``k`` / ``kb`` / ``a`` / ``ab`` diagonals, the coefficient-space
        templates) on the supplied per-component coefficient frame -- the
        transpose engine's internal frame (``dt.coeff.bare``, the half
        axis re-designated) -- while sharing every frame-independent
        attribute (grid, axes, scalars, chart). The clone carries the
        ``_distributed_matrix`` flag, so its ``_columns`` /
        ``_extend_nyquist_steady`` build the Nyquist supplements
        unconditionally (jnp masks, no host ``np.any``). Used only to
        assemble the per-mode matrix on the frame; the home instance is
        untouched (the single-device path stays bit-identical).

        Parameters
        ----------
        coeff_of : Callable[[str], SpaceLike]
            The per-component coefficient frame the symbols read.

        Returns
        -------
        Eigenmodes
            The frame clone.
        """
        clone = copy.copy(self)
        override = {c: coeff_of(c) for c in self._analysis}
        kit = GridSymbols(self._grid, self._analysis,
                          coeff_spaces=override)
        clone._kit = kit  # noqa: SLF001 — populating the clone
        x, y, z = self._axes
        face = {x: "u", y: "v", z: "w"}
        axes = (x, y, z)
        clone.k = _LazySymbols(axes, lambda n: kit.diff(n, on="p"))
        clone.kb = _LazySymbols(axes, lambda n: kit.diff(n, on=face[n]))
        clone.a = _LazySymbols(axes, lambda n: kit.interp(n, on="p"))
        clone.ab = _LazySymbols(axes, lambda n: kit.interp(n, on=face[n]))
        clone._templates = {  # noqa: SLF001 — populating the clone
            c: self._grid.create_field(override[c]).with_metadata(name=c)
            for c in ("u", "v", "w", "b")}
        clone._distributed_matrix = True  # noqa: SLF001 — the clone
        return clone

    def operator_matrix(
        self,
        coeff_of: Callable[[str], SpaceLike],
        *,
        branches: tuple[int, ...],
        f: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> jax.Array:
        r"""
        Assemble the per-mode ``4 x 4`` operator matrix on a coeff frame.

        Description
        -----------
        The frame-local operator for the fused distributed route:
        ``sum_s w_s q^s (p^s)^H`` over ``branches`` (``f is None`` is the
        projector ``w_s = 1``; otherwise ``w_s = f(omega_s)``), on the
        internal coefficient frame ``coeff_of`` returns per component
        (``dt.coeff.bare``). The matrix threads sharded into
        :meth:`~fridom.spatial.operators.distributed_transform.DistributedTransform.apply_matrix`.

        Parameters
        ----------
        coeff_of : Callable[[str], SpaceLike]
            The per-component internal coefficient frame.
        branches : tuple[int, ...]
            The mode-branch selection.
        f : Callable[[np.ndarray], np.ndarray] | None, optional
            The scalar spectral function (``None`` is the projector)
            (default: None).

        Returns
        -------
        jax.Array
            The per-mode matrix, shape ``(*coeff_bare, 4, 4)``.
        """
        clone = self._reframe(coeff_of)
        shape = coeff_of(self._components[-1]).shape
        return assemble_operator_matrix(
            clone, branches=branches, components=self._components,
            shape=shape, f=f)

    def _interp_table(self, component: str) -> Symbol:
        r"""Interp magnitude table ``cos(k dz/2)``, endo on ``component``.

        Description
        -----------
        The vertical staggering-interpolation magnitude materialized
        as a plain real diagonal on the component's **own** vertical
        mode lattice via ``Symbol.from_field(grid.wavenumbers(...))``
        (the sanctioned coefficient-coordinate crossing) — not a
        retagging operator symbol.
        """
        z = self._axes[2]
        dz = self._mz.dx
        kz = self._grid.wavenumbers(self._kit.coeff(component),
                                    name=z)
        return Symbol.from_field(
            kz.with_data(jnp.cos(kz.data * (dz / 2.0))))

    def _diff_table(self, component: str) -> Symbol:
        r"""Build the ``2 sin(k dz/2)/dz`` derivative table, endo.

        Description
        -----------
        The vertical staggered-derivative magnitude on the
        component's **own** vertical mode lattice (see
        :meth:`_interp_table`); nonzero at the buoyancy top mode
        ``m = n`` (``2/dz``), which anchors the pure-``b`` stratum.
        """
        z = self._axes[2]
        dz = self._mz.dx
        kz = self._grid.wavenumbers(self._kit.coeff(component),
                                    name=z)
        return Symbol.from_field(
            kz.with_data(2.0 * jnp.sin(kz.data * (dz / 2.0)) / dz))

    def _wrap(self, name: str, data: jax.Array) -> ScalarField:
        """Broadcast a diagonal onto the component's coefficient field."""
        template = self._templates[name]
        full = jnp.broadcast_to(data, template.data.shape)
        return template.with_data(full.astype(template.data.dtype))

    def _energy_weights(self) -> dict[str, float]:
        r"""Per-component energy weights (the ``fr.model.EnergyMetric`` diag).

        Description
        -----------
        ``diag(1, 1, dsqr, 1/N^2)`` on ``(u, v, w, b)`` -- the
        canonical nonhydro energy metric ``M`` (a single source of
        truth with ``fr.model.EnergyMetric.from_model``). The ``1/N^2``
        reciprocal falls back to ``1`` for the degenerate ``N^2 = 0``
        (pure-inertial) grid the constructor permits -- the metric
        proper (and ``fr.model.EnergyMetric``) needs ``N^2 != 0``.
        """
        inv_n2 = 1.0 / self.n2 if self.n2 != 0.0 else 1.0
        return nonhydro_energy_weights(self.dsqr, inv_n2)


def _bounded_names(grid: Grid) -> tuple[str, ...]:
    """Return the names of the grid's bounded (walled) axes."""
    return tuple(
        name for mesh in grid.factors
        if not getattr(mesh, "periodic", True)
        for name in mesh.names)


def eigenbasis(
    model: Model, *, at_time: float = 0.0,
) -> ChannelEigenmodes:
    r"""
    Build the labeled numeric eigenbasis of a channel model.

    Description
    -----------
    The user surface of the dense-column channel engine: returns the
    :class:`~fridom.nonhydro2.channel_eigenmodes.ChannelEigenmodes`
    of a model with exactly one bounded **horizontal** axis (the
    rotating stratified channel — the walls the rotation couples to,
    where no trigonometric basis exists) — ``eb.omega`` / ``eb.q``
    / ``eb.labels`` per ``(kx, kz)`` mode plane, the ``families``
    vocabulary (the physical vortical / kelvin / wave families plus
    the non-physical ``constraint`` divergence-complement), the
    segment ``slices``, and ``eb.projector(sel)`` for family /
    predicate projections on physical states. Works on the beta
    plane (coefficients may vary along the bounded axis).

    A fully periodic grid and a walled-**vertical** grid both carry
    analytic eigenmodes (the trigonometric vertical basis survives
    rigid lids — rotation acts about the vertical); the taught
    errors point at :func:`from_model`. A multi-walled box has no
    periodic axis left to diagonalize over and is rejected.

    Parameters
    ----------
    model : Model
        The assembled nonhydrostatic channel model.
    at_time : float, optional
        The clock time at which to freeze time-dependent parameters
        (default: 0.0).

    Returns
    -------
    ChannelEigenmodes
        The labeled channel eigenmodes.

    Raises
    ------
    ValueError
        On a fully periodic, walled-vertical or multi-walled grid.
    """
    bounded = _bounded_names(model.grid)
    if not bounded:
        raise ValueError(
            "nh.eigenbasis is the numeric labeled eigenbasis of the "
            "horizontally walled channel; this grid is fully "
            "periodic — use the analytic eigenmodes instead "
            "(nh.eigenmodes.from_model(model)) and the "
            "nh.transforms projections")
    if len(bounded) > 1:
        raise ValueError(
            "nh.eigenbasis serves the single-walled channel; this "
            f"grid bounds {bounded!r} — a multi-walled box has no "
            "periodic axis left to diagonalize over")
    if bounded[0] == "z":
        raise ValueError(
            "nh.eigenbasis serves walls on a horizontal axis (where "
            "rotation obstructs the trigonometric basis); the "
            "walled-vertical (rigid-lid) grid keeps analytic "
            "eigenmodes — use nh.eigenmodes.from_model(model) and "
            "the nh.transforms projections")
    return ChannelEigenmodes(model, at_time=at_time)


def from_model(
    model: Model, *, at_time: float = 0.0,
) -> Eigenmodes | ChannelEigenmodes:
    """Build the eigenmodes of an assembled nonhydro model (D2.4).

    Description
    -----------
    Dispatches on the grid topology. A fully periodic or
    walled-**vertical** (rigid-lid) grid gets the analytic
    operator-sourced :class:`Eigenmodes`: ``coriolis.f0``,
    ``stratification.n2`` and ``nonhydro.dsqr`` are read from
    ``model.parameters`` with the constancy check — a
    ``BetaPlaneCoriolis`` model does not provide ``coriolis.f0`` and
    is rejected (not Fourier-diagonalizable); Ramp-valued parameters
    are evaluated at ``at_time``. The eigenmodes are a fixed-``at_time``
    snapshot — a time-dependent parameter is frozen at that instant
    (default 0.0) and the modes do not evolve with the run (TDF-D6, a
    deliberately time-frozen analysis surface). A grid with exactly one
    bounded **horizontal** axis gets the numeric
    :class:`~fridom.nonhydro2.channel_eigenmodes.ChannelEigenmodes`
    (the labeled dense-column channel eigenbasis, beta-plane
    included). A multi-walled box is rejected.

    Parameters
    ----------
    model : fr.model.Model
        An assembled nonhydrostatic model.
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).

    Returns
    -------
    Eigenmodes | ChannelEigenmodes
        The analytic eigenmodes (fully periodic / walled vertical)
        or the labeled channel eigenmodes (one bounded horizontal
        axis).
    """
    if getattr(model.grid, "immersed", None) is not None:
        raise NotImplementedError(
            "nonhydro eigenmodes do not serve immersed (cut-cell) "
            "grids: the eigenbasis of the masked cut-cell operator is "
            "not the tensor-product basis the analytic / channel "
            "engines diagonalize (immersed-partial-cells plan, IP-D8) "
            "— the masked spectrum is designed-for. Use an unimmersed "
            "grid for eigenmode analysis and from_model transforms.")
    bounded = _bounded_names(model.grid)
    if len(bounded) > 1:
        raise ValueError(
            "nonhydro eigenmodes serve the fully periodic grid, the "
            "walled-vertical grid (analytic) or the single-walled "
            f"horizontal channel (numeric); this grid bounds "
            f"{bounded!r} — a multi-walled box has no periodic axis "
            "left to diagonalize over")
    if bounded and bounded[0] != "z":
        return ChannelEigenmodes(model, at_time=at_time)
    params = model.parameters

    def _read(name: str) -> float:
        if name not in params:
            raise ValueError(
                f"eigenmodes need a constant {name!r}; the model does "
                "not provide it (a beta-plane / profile module is not "
                "Fourier-diagonalizable — the analytic path is "
                "constant-only; a varying profile is served on the "
                "single-walled horizontal channel by nh.eigenbasis, "
                "the numeric engine)")
        value = params[name]
        if isinstance(value, fr.model.TimeDependent):
            return float(value.at_time(at_time))
        return float(value)

    return Eigenmodes(
        model.grid,
        f0=_read(fr.model.params.CORIOLIS_F0),
        n2=_read(fr.model.params.STRATIFICATION_N2),
        dsqr=_read(DSQR),
        family=_model_family(model))


def _model_family(model: Model) -> str:
    """
    Read the discretization family off a model's velocity space.

    Description
    -----------
    The eigenmode analysis spaces must match the model's own family
    (FV-D3, stage F3): an FV nonhydro model carries its C-grid
    velocities on ``CellAvg`` transverse factors, so a single
    ``AverageSpace`` factor on the vertical velocity identifies the
    finite-volume family. A nodal model has none.

    Parameters
    ----------
    model : fr.model.Model
        The assembled nonhydrostatic model.

    Returns
    -------
    str
        ``"fv"`` when the velocity carries an average factor, else
        ``"nodal"``.
    """
    space = model.state["w"].function_space.bare
    is_fv = any(isinstance(factor, AverageSpace)
                for factor in space.factors)
    return "fv" if is_fv else "nodal"
