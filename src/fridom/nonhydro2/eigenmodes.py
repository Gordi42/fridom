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

from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework2.grid.symbols import GridSymbols, rayleigh_dual
from fridom.framework2.model.energy import nonhydro_energy_weights
from fridom.nonhydro2.params import DSQR
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.model.model import Model


class Eigenmodes:

    r"""Discrete inertia-gravity / geostrophic eigenmodes on a grid.

    Description
    -----------
    Binds a :class:`~fridom.framework2.grid.symbols.GridSymbols` kit
    on the canonical C-grid component spaces and exposes the analytic
    eigenmode surface: the dispersion ``omega(s)`` (a ``Symbol``),
    the eigenvector column ``q(s)`` (a coefficient-space ``State``),
    and the ``projector(s)`` closure (coefficient ``State ->
    State``). The per-axis symbol families are public attributes:
    ``k`` (centre -> face derivative), ``kb`` (face -> centre
    derivative), ``a`` (centre -> face interpolation) and ``ab``
    (face -> centre interpolation), keyed by coordinate name.

    Parameters
    ----------
    grid : fr.grid.Grid
        A 3-D periodic grid carrying the vertical coordinate.
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

        spaces = {
            "u": fr.Staggered(x).resolve(grid),
            "v": fr.Staggered(y).resolve(grid),
            "w": fr.Staggered(z).resolve(grid),
            "b": fr.Collocated().resolve(grid),
            "p": fr.Collocated().resolve(grid),
        }
        kit = GridSymbols(grid, spaces)
        self._kit: GridSymbols = kit
        face = {x: "u", y: "v", z: "w"}
        axes = (x, y, z)
        self.k: dict[str, Symbol] = {
            n: kit.diff(n, on="p") for n in axes}
        self.kb: dict[str, Symbol] = {
            n: kit.diff(n, on=face[n]) for n in axes}
        self.a: dict[str, Symbol] = {
            n: kit.interp(n, on="p") for n in axes}
        self.ab: dict[str, Symbol] = {
            n: kit.interp(n, on=face[n]) for n in axes}
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
        structural zero to zero (no caller-side masking).

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
        Tag-checked operator compositions with common domain = ``w``'s
        coefficient space (the codomain tags of the entries mix union
        factors; consumers rely on the data). The degenerate modes
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
        (``P(0)`` identical, ``P(+1)`` and ``P(-1)`` swap).
        """
        x, y, z = self._axes
        k, kb, a, ab = self.k, self.kb, self.a, self.ab
        if s == 0:
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
        ``P^s z = q^s \langle p^s, z\rangle`` with the Rayleigh dual
        ``p^s`` derived from ``q^s`` under the nonhydro energy metric
        (``fr.grid.rayleigh_dual`` + the ``diag(1, 1, dsqr, 1/N^2)``
        weights): idempotent by biorthonormality, exactly zero on the
        structurally degenerate modes.

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
        p = rayleigh_dual(q, self._energy_weights())

        def project(z: State) -> State:
            """Project ``z`` onto mode ``s`` (pointwise per mode)."""
            amp = sum(jnp.conj(p[c].data) * z[c].data for c in p)
            return State({c: self._wrap(c, q[c].data * amp)
                          for c in q})

        return project

    # ================================================================
    #  Internals
    # ================================================================
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


def from_model(model: Model, *, at_time: float = 0.0) -> Eigenmodes:
    """Build eigenmodes from an assembled model's parameters (D2.4).

    Description
    -----------
    Reads ``coriolis.f0``, ``stratification.n2`` and ``nonhydro.dsqr``
    from ``model.parameters`` with the constancy check: a
    ``BetaPlaneCoriolis`` model does not provide ``coriolis.f0`` and is
    rejected (not Fourier-diagonalizable); Ramp-valued parameters are
    evaluated at ``at_time``.

    Parameters
    ----------
    model : fr.Model
        An assembled nonhydrostatic model.
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).

    Returns
    -------
    Eigenmodes
        The eigenmode set.
    """
    params = model.parameters

    def _read(name: str) -> float:
        if name not in params:
            raise ValueError(
                f"eigenmodes need a constant {name!r}; the model does "
                "not provide it (a beta-plane / profile module is not "
                "Fourier-diagonalizable)")
        value = params[name]
        if isinstance(value, fr.TimeDependent):
            return float(value.at_time(at_time))
        return float(value)

    return Eigenmodes(
        model.grid,
        f0=_read(fr.params.CORIOLIS_F0),
        n2=_read(fr.params.STRATIFICATION_N2),
        dsqr=_read(DSQR))
