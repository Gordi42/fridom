r"""Operator-sourced analytic eigenmodes of the discrete C-grid.

Description
-----------
The analytic closed-form eigenmodes of the discrete nonhydrostatic
C-grid linear operator, assembled **from the grid's own operator
symbols** instead of hand-coded trigonometric formulas: a
``fr.grid.GridSymbols`` kit names the staggered component spaces
(``u``, ``v``, ``w``, ``b``, ``p``) once, and every derivative /
interpolation diagonal (``k``, ``kb``, ``a``, ``ab``) is the
corresponding operator's ``eigenvalues`` query on the matching
coefficient space. The dispersion relation is the symbol algebra
(``magnitude ** 2`` quantities composed with a structural-zero
``inverse``), the eigenvector column ``q^s`` is a tag-checked
composition of the same symbols, and the biorthonormal dual is the
derived ``fr.grid.rayleigh_dual`` under the nonhydro energy metric —
no hand-written left vector and no caller-side masking (the ``k = 0``
mean and the degenerate ``k_h = 0`` / Nyquist modes drop through
exact structural zeros).

On a **walled** (bounded, rigid-lid) vertical the kit spaces carry
the physics-fixed parity tags (``w`` Dirichlet, ``u``/``v``/``p``
Neumann, ``b`` Dirichlet), the same formulas compose under the
derived-shift trig symbol algebra, and cross-component sums align
per physical vertical mode through the ``fr.grid.ModeChart`` union
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
states satisfying ``L q^s = i \omega^s q^s`` for the linearized,
Leray-projected tendency. ``em.grid`` and ``em.kit`` expose the grid
and the per-component transform kit — the ``nh.transforms``
projection surface for the physical round-trip.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.operators.symbol import Symbol
from fridom.framework2.grid.symbols import (
    GridSymbols,
    ModeChart,
    rayleigh_dual,
)
from fridom.framework2.model.eigenstates import (
    coefficient_index,
    envelope_scale,
    hermitian_mode_data,
)
from fridom.framework2.model.energy import nonhydro_energy_weights
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.params import DSQR
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator

    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.meshes.mesh import Mesh
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike
    from fridom.framework2.model.model import Model


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
    Binds a :class:`~fridom.framework2.grid.symbols.GridSymbols` kit
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
    grid : fr.grid.Grid
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
    """

    def __init__(
        self, grid: Grid, *, f0: float, n2: float, dsqr: float,
        vertical: str = "z",
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
        # to the exact BC-free spaces.
        spaces = {
            "u": fr.Staggered(
                x, wall_bc={z: BC.NEUMANN}).resolve(grid),
            "v": fr.Staggered(
                y, wall_bc={z: BC.NEUMANN}).resolve(grid),
            "w": fr.Staggered(
                z, wall_bc={z: BC.DIRICHLET}).resolve(grid),
            "b": fr.Collocated(
                wall_bc={z: BC.DIRICHLET}).resolve(grid),
            "p": fr.Collocated(
                wall_bc={z: BC.NEUMANN}).resolve(grid),
        }
        # the model-facing physical spaces (u, v, b BC-free; w's
        # Dirichlet wall tag matches the Velocity declaration) —
        # identical to the kit spaces on a periodic grid
        self._physical: dict[str, SpaceLike] = {
            "u": fr.Staggered(x).resolve(grid),
            "v": fr.Staggered(y).resolve(grid),
            "w": spaces["w"],
            "b": fr.Collocated().resolve(grid),
        }
        kit = GridSymbols(grid, spaces)
        self._kit: GridSymbols = kit
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
        self._templates: dict[str, ScalarField] = {
            c: kit.forward(c)(grid.create_field(spaces[c]))
            .with_metadata(name=c)
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
        (``k_h = 0`` for ``s != 0``, the ``k = 0`` mean, and the
        interpolation-Nyquist zeros for ``s = 0``) are **exact**
        structural zeros of every entry, so the Rayleigh dual
        vanishes there with no masking.

        The ``s != 0`` column is built on the opposite dispersion
        root ``omega(-s)``: the linearized tendency satisfies
        ``L q = -i omega q`` for the column written with ``omega``
        (the :math:`e^{i(kx - \omega t)}` convention), so pairing
        branch ``s`` with the root ``-s`` yields the eigen-relation
        ``L q^s = +i omega^s q^s`` asserted by the tests. This is a
        pure branch relabelling: the projector family is unchanged
        (``P(0)`` identical, ``P(+1)`` and ``P(-1)`` swap). On a
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
                return {
                    "u": -(a[x] @ (ab[y] @ k[y]))
                    * self._interp_table("u"),
                    "v": (a[y] @ (ab[x] @ k[x]))
                    * self._interp_table("v"),
                    "w": self.omega(0),
                    "b": -self.f0 * (a[x].magnitude ** 2
                                     * a[y].magnitude ** 2
                                     * self._diff_table("b")),
                }
            return {
                "u": -(a[x] @ (ab[y] @ k[y]) @ ab[z]),
                "v": a[y] @ (ab[x] @ k[x]) @ ab[z],
                "w": self.omega(0),
                "b": self.f0 * (a[x].magnitude ** 2
                                * a[y].magnitude ** 2 * kb[z]),
            }
        om = self.omega(-s)
        kh2 = k[x].magnitude ** 2 + k[y].magnitude ** 2
        return {
            "u": kb[z] @ (k[x] @ om
                          + 1j * self.f0 * (a[x] @ (ab[y] @ k[y]))),
            "v": kb[z] @ (k[y] @ om
                          - 1j * self.f0 * (a[y] @ (ab[x] @ k[x]))),
            "w": om @ kh2,
            "b": -1j * self.n2 * (ab[z] @ kh2),
        }

    def q(self, s: int = 1) -> State:
        r"""Eigenvector ``q^s`` as a coefficient-space ``State``.

        Description
        -----------
        Each component symbol's diagonal is wrapped on that
        component's own coefficient space (the kit's per-component
        transform codomain), broadcast to the full spectral shape.

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
        metric (``fr.grid.rayleigh_dual`` + the ``diag(1, 1, dsqr,
        1/N^2)`` weights): idempotent by biorthonormality, exactly
        zero on the structurally degenerate modes. On a walled
        vertical the amplitude is accumulated on the
        ``fr.grid.ModeChart`` union mode lattice (the components'
        trig families hold different mode ranges); on a periodic
        grid the chart is identity and the data path is unchanged.

        Parameters
        ----------
        s : int, optional
            The mode branch: 0, +1 or -1 (default: 1).

        Returns
        -------
        Callable[[State], State]
            The projection acting on coefficient-space states.
        """
        q = self._vec_q(s)
        p = self._dual(q, s)
        chart = self._chart
        coeff = {c: self._kit.coeff(c) for c in q}

        def project(z: State) -> State:
            """Project ``z`` onto mode ``s`` (pointwise per mode)."""
            amp = sum(chart.embed(jnp.conj(p[c]) * z[c].data,
                                  coeff[c]) for c in p)
            return State({
                c: self._wrap(c, q[c].data
                              * chart.restrict(amp, coeff[c]))
                for c in q})

        return project

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
        :math:`\mathrm{Re}(q^s(k)\,e^{i(k\cdot x + \mathrm{phase})})`
        satisfying ``d/dt state(phase) = omega * state(phase +
        pi/2)`` under the linearized, Leray-projected tendency,
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
            and buoyancy-top ``m = n`` strata).
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
                "branches vanish at k_h = 0 and outside the "
                "vertical w strata; the geostrophic column "
                "vanishes on the interpolation-Nyquist planes)")

        def synth(shift: float) -> dict[str, ScalarField]:
            out = {}
            for c in components:
                if slots[c] is None:
                    data = jnp.zeros_like(q[c].data)
                else:
                    value = amps[c] * jnp.exp(
                        1j * (float(phase) + shift))
                    data = hermitian_mode_data(
                        q[c].function_space, slots[c], value)
                out[c] = self._kit.backward(c)(
                    q[c].with_data(data)).real.retag(
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
        ``fr.grid.rayleigh_dual`` applies in the symbol algebra.
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
        r"""Per-component energy weights (the ``fr.EnergyMetric`` diag).

        Description
        -----------
        ``diag(1, 1, dsqr, 1/N^2)`` on ``(u, v, w, b)`` -- the
        canonical nonhydro energy metric ``M`` (a single source of
        truth with ``fr.EnergyMetric.from_model``). The ``1/N^2``
        reciprocal falls back to ``1`` for the degenerate ``N^2 = 0``
        (pure-inertial) grid the constructor permits -- the metric
        proper (and ``fr.EnergyMetric``) needs ``N^2 != 0``.
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
    are evaluated at ``at_time``. A grid with exactly one bounded
    **horizontal** axis gets the numeric
    :class:`~fridom.nonhydro2.channel_eigenmodes.ChannelEigenmodes`
    (the labeled dense-column channel eigenbasis, beta-plane
    included). A multi-walled box is rejected.

    Parameters
    ----------
    model : fr.Model
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
        if isinstance(value, fr.TimeDependent):
            return float(value.at_time(at_time))
        return float(value)

    return Eigenmodes(
        model.grid,
        f0=_read(fr.params.CORIOLIS_F0),
        n2=_read(fr.params.STRATIFICATION_N2),
        dsqr=_read(DSQR))
