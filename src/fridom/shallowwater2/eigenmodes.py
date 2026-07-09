r"""Operator-sourced analytic eigenmodes of the discrete C-grid.

Description
-----------
The analytic closed-form eigenmodes of the discrete shallow-water
C-grid linear operator, assembled **from the grid's own operator
symbols** instead of the continuous formulas: a
``fr.grid.GridSymbols`` kit names the staggered component spaces
(``u``, ``v``, ``p``) once, and every derivative / interpolation
diagonal (``k``, ``kb``, ``a``, ``ab``) is the corresponding
operator's ``eigenvalues`` query on the matching coefficient space.
The dispersion relation is the symbol algebra
(``magnitude ** 2`` quantities), the eigenvector column ``q^s`` is a
tag-checked composition of the same symbols, and the biorthonormal
dual is the derived ``fr.grid.rayleigh_dual`` under the
shallow-water energy metric — no hand-written left vector and no
caller-side masking (the degenerate interpolation-Nyquist
geostrophic modes drop through exact structural zeros).

The one non-structural degeneracy is the ``k = 0`` mean: every
symbol-composed wave entry vanishes there while the physical
inertial pair ``omega = +/- f_0`` survives, so the ``s != 0``
columns carry an explicit inertial patch ``(u, v, p) =
(-i s, 1, 0)`` at the mean mode — the ``k = 0`` triple
``{geostrophic mean pressure, +f_0, -f_0}`` stays complete and
M-orthogonal.

Surface: ``em.omega(s)`` returns the frequency ``Symbol`` (``.data``
for the half-spectrum array), ``em.q(s)`` the eigenvector as a
coefficient-space :class:`~fridom.shallowwater2.state.State`, and
``em.projector(s)`` a ``State -> State`` callable on coefficient
states satisfying ``L q^s = i \omega^s q^s`` for the linearized
tendency (shallow water carries no constraint stage).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework2.grid.operators.symbol import Symbol
from fridom.framework2.grid.symbols import GridSymbols, rayleigh_dual
from fridom.framework2.model.energy import shallowwater_energy_weights
from fridom.framework2.model.time_dependent import resolve_at
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.model import Model


class Eigenmodes:

    r"""Discrete inertia-gravity / geostrophic eigenmodes on a grid.

    Description
    -----------
    Binds a :class:`~fridom.framework2.grid.symbols.GridSymbols` kit
    on the canonical C-grid component spaces (``u`` staggered along
    the first coordinate, ``v`` along the second, ``p`` collocated)
    and exposes the analytic eigenmode surface: the dispersion
    ``omega(s)`` (a ``Symbol``), the eigenvector column ``q(s)`` (a
    coefficient-space ``State``), and the ``projector(s)`` closure
    (coefficient ``State -> State``). The per-axis symbol families
    are public attributes: ``k`` (centre -> face derivative), ``kb``
    (face -> centre derivative), ``a`` (centre -> face interpolation)
    and ``ab`` (face -> centre interpolation), keyed by coordinate
    name.

    Parameters
    ----------
    grid : fr.grid.Grid
        A 2-D periodic grid.
    f0 : float
        The (constant) Coriolis parameter.
    csqr : float
        The (constant) squared gravity-wave phase speed.
    """

    def __init__(self, grid: Grid, *, f0: float, csqr: float) -> None:
        """Build the symbol kit and the per-axis operator diagonals."""
        if float(f0) == 0.0 and float(csqr) == 0.0:
            raise ValueError(
                "degenerate eigenmode system: f0 == csqr == 0")
        self.grid = grid
        self.f0 = float(f0)
        self.csqr = float(csqr)

        names = grid.names
        if len(names) != 2:  # noqa: PLR2004
            raise ValueError(
                "shallow-water eigenmodes need a 2-D grid; got "
                f"names {names}")
        x, y = names
        self._axes: tuple[str, str] = (x, y)

        spaces = {
            "u": fr.Staggered(x).resolve(grid),
            "v": fr.Staggered(y).resolve(grid),
            "p": fr.Collocated().resolve(grid),
        }
        kit = GridSymbols(grid, spaces)
        self._kit: GridSymbols = kit
        face = {x: "u", y: "v"}
        axes = (x, y)
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
            for c in ("u", "v", "p")}

    # ================================================================
    #  Dispersion
    # ================================================================
    def _omega2(self) -> Symbol:
        r"""Squared dispersion symbol (p-referenced, exact zeros).

        Description
        -----------
        The discrete relation

        .. math::

            \omega^2 = f_0^2\,|\hat a_x|^2 |\hat a_y|^2
                       + c^2\,(|\hat k_x|^2 + |\hat k_y|^2)

        assembled from the operator symbols' ``magnitude ** 2``
        quantities on ``p``'s coefficient space. Unlike the nonhydro
        relation there is no denominator: the ``k = 0`` mean carries
        the physical inertial ``omega^2 = f_0^2`` (no masking).

        Returns
        -------
        Symbol
            The real, non-negative ``omega ** 2`` diagonal.
        """
        x, y = self._axes
        kh2 = self.k[x].magnitude ** 2 + self.k[y].magnitude ** 2
        coriolis = self.f0 ** 2 * (
            self.a[x].magnitude ** 2 * self.a[y].magnitude ** 2)
        return coriolis + self.csqr * kh2

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
        Tag-checked operator compositions with common domain = ``p``'s
        coefficient space (the codomain tags of the entries are the
        per-component coefficient spaces). The interpolation-Nyquist
        geostrophic modes are **exact** structural zeros of every
        ``s = 0`` entry, so the Rayleigh dual vanishes there with no
        masking; the ``s != 0`` columns are patched at the ``k = 0``
        mean with the inertial pair (see :meth:`_patch_mean`).

        The branch pairing was fixed against the strong test
        ``L q^s = +i omega^s q^s`` (the C4 lesson): relative to the
        naive continuous port, the Coriolis terms flip sign while
        ``om = omega(s)`` is used directly — the ``omega(-s)``
        relabelling alone cannot fix the pairing here because the
        ``p`` entry is om-independent.
        """
        x, y = self._axes
        k, a, ab = self.k, self.a, self.ab
        if s == 0:
            return {
                "u": -(a[x] @ (ab[y] @ k[y])),
                "v": a[y] @ (ab[x] @ k[x]),
                "p": self.f0 * (a[x].magnitude ** 2
                                * a[y].magnitude ** 2),
            }
        om = self.omega(s)
        kh2 = k[x].magnitude ** 2 + k[y].magnitude ** 2
        column = {
            "u": k[x] @ om - 1j * self.f0 * (a[x] @ (ab[y] @ k[y])),
            "v": k[y] @ om + 1j * self.f0 * (a[y] @ (ab[x] @ k[x])),
            "p": -1j * self.csqr * kh2,
        }
        inertial = {"u": -1j * s, "v": 1.0 + 0j, "p": 0.0 + 0j}
        return {c: self._patch_mean(c, column[c], inertial[c])
                for c in column}

    def _patch_mean(
        self, name: str, sym: Symbol, value: complex,
    ) -> Symbol:
        r"""Set the ``k = 0`` mean entry of a wave-column symbol.

        Description
        -----------
        Every symbol-composed ``s != 0`` entry vanishes exactly at
        the ``k = 0`` mean (``k`` and ``kh2`` are structural zeros
        there), but the physical mean mode is the inertial
        oscillation ``omega = s f_0`` with eigenvector
        ``(u, v, p) = (-i s, 1, 0)`` (``L q = i s f_0 q`` for the
        pure-rotation ``k = 0`` system). The patch writes that value
        into the mean entry so the ``k = 0`` triple stays complete
        and M-orthogonal (``<q^+, q^->_M = 1 - 1 = 0``).

        Parameters
        ----------
        name : str
            The component name (selects the broadcast template).
        sym : Symbol
            The composed wave-column entry.
        value : complex
            The inertial eigenvector entry at ``k = 0``.

        Returns
        -------
        Symbol
            The patched diagonal (tags kept).
        """
        shape = self._templates[name].data.shape
        data = jnp.broadcast_to(sym.data, shape)
        data = data.at[(0,) * len(shape)].set(value)
        return Symbol(sym.space, data, codomain=sym.codomain)

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
            The ``(u, v, p)`` coefficient-space eigenvector.
        """
        syms = self._vec_q(s)
        return State({c: self._wrap(c, syms[c].data) for c in syms})

    def projector(self, s: int = 1) -> Callable[[State], State]:
        r"""Return the spectral projector ``P^s`` on coefficient states.

        Description
        -----------
        ``P^s z = q^s \langle p^s, z\rangle`` with the Rayleigh dual
        ``p^s`` derived from ``q^s`` under the shallow-water energy
        metric (``fr.grid.rayleigh_dual`` + the ``diag(1, 1, 1/c^2)``
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
        ``diag(1, 1, 1/c^2)`` on ``(u, v, p)`` -- the canonical
        shallow-water energy metric ``M`` (a single source of truth
        with ``fr.EnergyMetric.from_model``). The ``1/c^2``
        reciprocal falls back to ``1`` for the degenerate ``c^2 = 0``
        (no-gravity) grid the constructor permits -- the metric
        proper (and ``fr.EnergyMetric``) needs ``c^2 != 0``.
        """
        inv_csqr = 1.0 / self.csqr if self.csqr != 0.0 else 1.0
        return shallowwater_energy_weights(inv_csqr)


def from_model(model: Model, *, at_time: float = 0.0) -> Eigenmodes:
    r"""
    Build the eigenmodes of an assembled shallow-water model.

    Description
    -----------
    Extracts the constant Coriolis parameter and squared phase speed
    through ``model.parameters`` with structural validation: a
    beta-plane core provides no ``coriolis.f0`` (its ``f`` is a
    field, not Fourier-diagonalizable) and raises here. Ramp-valued
    parameters demand an explicit ``at_time`` (a fixed-time snapshot).

    Parameters
    ----------
    model : Model
        The assembled shallow-water model.
    at_time : float, optional
        The clock time at which to freeze time-dependent parameters
        (default: 0.0).

    Returns
    -------
    Eigenmodes
        The eigenmode object.

    Raises
    ------
    ValueError
        If ``coriolis.f0`` or ``shallowwater.csqr`` is not provided
        (a non-constant-coefficient system).
    """
    view = model.parameters
    for name, why in (
        (fr.params.CORIOLIS_F0,
         "a constant Coriolis parameter (a beta-plane f(y) is not "
         "Fourier-diagonalizable); assemble with "
         "sw.modules.FPlaneCoriolis"),
        (sw_params.CSQR,
         "a constant squared phase speed; assemble with a "
         "constant-depth DynamicalCore"),
    ):
        if name not in view:
            raise ValueError(
                f"shallow-water eigenmodes need {why}: no {name!r} "
                "provider on this model")
    f0 = resolve_at(view[fr.params.CORIOLIS_F0], at_time)
    csqr = resolve_at(view[sw_params.CSQR], at_time)
    return Eigenmodes(model.grid, f0=f0, csqr=csqr)
