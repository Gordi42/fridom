r"""Operator-sourced analytic eigenmodes of the discrete C-grid.

Description
-----------
The analytic closed-form eigenmodes of the discrete shallow-water
C-grid linear operator, assembled **from the grid's own operator
symbols** instead of the continuous formulas: a
``fr.spatial.GridSymbols`` kit names the staggered component spaces
(``u``, ``v``, ``p``) once, and every derivative / interpolation
diagonal (``k``, ``kb``, ``a``, ``ab``) is the corresponding
operator's ``eigenvalues`` query on the matching coefficient space.
The dispersion relation is the symbol algebra
(``magnitude ** 2`` quantities), the eigenvector column ``q^s`` is a
tag-checked composition of the same symbols, and the biorthonormal
dual is the derived ``fr.spatial.rayleigh_dual`` under the
shallow-water energy metric — no hand-written left vector and no
caller-side masking.

Two strata need explicit patches beyond the symbol-composed
formulas. The ``k = 0`` mean: every symbol-composed wave entry
vanishes there while the physical inertial pair
``omega = +/- f_0`` survives, so the ``s != 0`` columns carry an
explicit inertial patch ``(u, v, p) = (-i s, 1, 0)`` at the mean
mode — the ``k = 0`` triple ``{geostrophic mean pressure, +f_0,
-f_0}`` stays complete and M-orthogonal. And the
interpolation-Nyquist planes of an even grid (where the staggering
average ``cos(k dx / 2)`` hits its exact structural zero): rotation
decouples there and the composed geostrophic column vanishes, while
the plane operator still carries one steady mode — the
discrete-divergence-free velocity ``(u, v, p) = (conj(k_y),
-conj(k_x), 0)`` with ``p = 0`` — which the ``s = 0`` column
carries explicitly, so the discrete ``{vortical, +, -}`` family is
complete at **every** mode of the lattice (odd grids have no
Nyquist stratum and are bitwise unaffected).

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
import numpy as np

import fridom as fr
from fridom.model.eigenstates import (
    coefficient_index,
    describe_nonfinite_branch,
    envelope_scale,
    evaluate_frequency_function,
    hermitian_mode_data,
    resolve_mode_branches,
)
from fridom.model.time_dependent import resolve_at
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.channel_eigenmodes import ChannelEigenmodes
from fridom.shallowwater2.energy import shallowwater_energy_weights
from fridom.shallowwater2.state import State
from fridom.spatial.operators.symbol import Symbol
from fridom.spatial.symbols import GridSymbols, rayleigh_dual

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Mapping

    import jax

    from fridom.model.model import Model
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid


class Eigenmodes:

    r"""Discrete inertia-gravity / geostrophic eigenmodes on a grid.

    Description
    -----------
    Binds a :class:`~fridom.spatial.symbols.GridSymbols` kit
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
    grid : fr.spatial.Grid
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
            "u": fr.spatial.Staggered(x).resolve(grid),
            "v": fr.spatial.Staggered(y).resolve(grid),
            "p": fr.spatial.Collocated().resolve(grid),
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
        per-component coefficient spaces). The composed ``s = 0``
        entries are **exact** structural zeros on the
        interpolation-Nyquist planes of an even grid, where the
        column continues as the rotation-decoupled steady
        divergence-free mode (see :meth:`_extend_nyquist_steady`);
        the ``s != 0`` columns are patched at the ``k = 0`` mean
        with the inertial pair (see :meth:`_patch_mean`).

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
            return self._extend_nyquist_steady({
                "u": -(a[x] @ (ab[y] @ k[y])),
                "v": a[y] @ (ab[x] @ k[x]),
                "p": self.f0 * (a[x].magnitude ** 2
                                * a[y].magnitude ** 2),
            })
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

    def _extend_nyquist_steady(
        self, column: dict[str, Symbol],
    ) -> dict[str, Symbol]:
        r"""Merge the Nyquist steady mode into the geostrophic column.

        Description
        -----------
        On the interpolation-Nyquist planes of an even grid the
        staggering average ``|a| = cos(k dx / 2)`` is an exact
        structural zero, rotation decouples, and every composed
        geostrophic entry vanishes — yet the plane operator still
        has one steady mode: the discrete-divergence-free velocity

        .. math::

            (u, v, p) = (\overline{\hat k_y}, -\overline{\hat k_x},
            0)

        (``kb_x u + kb_y v = 0`` exactly since ``conj(k) = -kb`` in
        the forward/backward difference symbol pair, and the
        Coriolis coupling is structurally zero on the stratum). The
        patch writes that value into the column exactly where the
        interpolation product ``|a_x|^2 |a_y|^2`` is structurally
        zero — disjoint from the composed column's support — so the
        mode family is complete at every mode. On an odd grid the
        stratum is empty and the column is returned untouched
        (bitwise the pre-Nyquist path).

        Parameters
        ----------
        column : dict[str, Symbol]
            The composed geostrophic column.

        Returns
        -------
        dict[str, Symbol]
            The column with the Nyquist steady stratum merged.
        """
        x, y = self._axes
        mask = (self.a[x].magnitude ** 2
                * self.a[y].magnitude ** 2).data == 0
        if not bool(np.any(np.asarray(mask))):
            return column
        supplement = {
            "u": jnp.where(mask, jnp.conj(self.k[y].data), 0.0),
            "v": jnp.where(mask, -jnp.conj(self.k[x].data), 0.0),
        }
        return {
            c: (Symbol(sym.space, sym.data + supplement[c],
                       codomain=sym.codomain)
                if c in supplement else sym)
            for c, sym in column.items()}

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
        metric (``fr.spatial.rayleigh_dual`` + the ``diag(1, 1, 1/c^2)``
        weights): idempotent by biorthonormality, exactly zero on
        structurally degenerate modes (none on the standard
        ``f_0 != 0``, ``c^2 != 0`` system — the family is complete,
        the even-grid Nyquist strata included).

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
        diagonal — ``f = 1`` on a selection reproduces the summed
        projectors exactly. ``s`` is a single branch or an iterable
        of distinct branches from ``{0, +1, -1}``. ``f`` is
        evaluated once, host-side, on the **real** frequencies of
        the branch's *represented* modes only (structural zeros of
        the column never reach ``f``; complex return values
        allowed): ``f = lambda w: 1 / (1j * w)`` builds
        :math:`L^{-1}` on the selection, ``f = lambda w: 1j * w``
        the forward operator. A singular ``f`` meeting a
        structurally represented zero frequency (the geostrophic
        branch ``s = 0``) is a taught ``ValueError``, never a
        floored division.

        Real-safety: for the conjugation-closed wave selection
        ``s = (1, -1)`` and ``f`` satisfying
        :math:`f(-\omega) = \overline{f(\omega)}` (true for
        :math:`1/(i\omega)` and :math:`i\omega`) the map sends
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
        metric = self._energy_weights()
        shape = self._templates["p"].data.shape
        terms = []
        for b in branches:
            q = self._vec_q(b)
            p = rayleigh_dual(q, metric)
            omega = np.broadcast_to(
                np.real(np.asarray(self.omega(b).data)), shape)
            norm = sum(
                metric[c] * np.abs(np.broadcast_to(
                    np.asarray(q[c].data), shape)) ** 2
                for c in q)
            weights = jnp.asarray(evaluate_frequency_function(
                f, omega, norm != 0,
                lambda bad, b=b, om=omega:
                describe_nonfinite_branch(b, om, bad)))
            terms.append((q, p, weights))

        def apply(z: State) -> State:
            """Apply ``sum_s P^s f(omega^s)`` (pointwise per mode)."""
            out: dict[str, ScalarField] | None = None
            for q, p, w in terms:
                amp = w * sum(jnp.conj(p[c].data) * z[c].data
                              for c in p)
                part = {c: self._wrap(c, q[c].data * amp)
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
        ``indices`` is an axis-keyed mapping of integer wavenumber
        indices (e.g. ``{"x": 3, "y": 0}``) — the half-spectrum
        axis runs ``0..n//2``, full-spectrum axes take any integer
        modulo ``n``. The state is the real Hermitian-closed
        physical mode
        :math:`\mathrm{Re}(q^s(k)\,e^{i(k\cdot x + \mathrm{phase})})`
        satisfying the eigen-relation ``d/dt state(phase) =
        omega * state(phase + pi/2)`` under the linearized
        tendency, normalized so the largest horizontal-velocity
        amplitude (the pointwise oscillation envelope over the
        ``u`` and ``v`` nodes) is one; a mode without horizontal
        velocity (the geostrophic ``k = 0`` mean) is left
        unnormalized. On the self-conjugate planes of the
        half-spectrum axis a wave branch synthesizes the standing
        (conjugate-mixed) real mode.

        Parameters
        ----------
        s : int
            The mode branch: 0, +1 or -1.
        indices : Mapping[str, int]
            Axis-keyed integer wavenumber indices, one per grid
            axis.
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
            (only on degenerate-parameter systems, e.g. the
            ``f_0 = 0`` geostrophic mean: the standard family is
            complete, Nyquist strata included).
        """
        components = ("u", "v", "p")
        q = self.q(s)
        slots = {c: coefficient_index(q[c].function_space, indices)
                 for c in components}
        amps = {c: q[c].data[slots[c]] for c in components}
        if all(float(jnp.abs(a)) == 0.0 for a in amps.values()):
            raise ValueError(
                f"mode s={s} at {dict(indices)!r} is structurally "
                "unrepresented on the discrete lattice (the mode "
                "family is complete on the standard f0 != 0, "
                "csqr != 0 system; degenerate parameters drop "
                "strata, e.g. f0 = 0 empties the geostrophic mean)")

        def synth(shift: float) -> dict[str, ScalarField]:
            out = {}
            for c in components:
                value = amps[c] * jnp.exp(1j * (float(phase)
                                                + shift))
                data = hermitian_mode_data(
                    q[c].function_space, slots[c], value)
                out[c] = self._kit.backward(c)(
                    q[c].with_data(data)).real
            return out

        z0 = synth(0.0)
        z1 = synth(jnp.pi / 2.0)
        scale = envelope_scale(z0, z1, ("u", "v"))
        state = State({c: z0[c] / scale for c in components})
        shape = self._templates["p"].data.shape
        omega = float(jnp.broadcast_to(
            self.omega(s).data, shape)[slots["p"]])
        return omega, state

    # ================================================================
    #  Internals
    # ================================================================
    def _wrap(self, name: str, data: jax.Array) -> ScalarField:
        """Broadcast a diagonal onto the component's coefficient field."""
        template = self._templates[name]
        full = jnp.broadcast_to(data, template.data.shape)
        return template.with_data(full.astype(template.data.dtype))

    def _energy_weights(self) -> dict[str, float]:
        r"""Per-component energy weights (the ``fr.model.EnergyMetric`` diag).

        Description
        -----------
        ``diag(1, 1, 1/c^2)`` on ``(u, v, p)`` -- the canonical
        shallow-water energy metric ``M`` (a single source of truth
        with ``fr.model.EnergyMetric.from_model``). The ``1/c^2``
        reciprocal falls back to ``1`` for the degenerate ``c^2 = 0``
        (no-gravity) grid the constructor permits -- the metric
        proper (and ``fr.model.EnergyMetric``) needs ``c^2 != 0``.
        """
        inv_csqr = 1.0 / self.csqr if self.csqr != 0.0 else 1.0
        return shallowwater_energy_weights(inv_csqr)


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
    :class:`~fridom.shallowwater2.channel_eigenmodes.ChannelEigenmodes`
    of a model with exactly one bounded (walled) axis — ``eb.omega``
    / ``eb.q`` / ``eb.labels`` per ``rfft`` plane, the ``families``
    vocabulary, the segment ``slices``, and ``eb.projector(sel)``
    for family / predicate projections on physical states. Works on
    the beta plane (coefficients may vary along the bounded axis).

    A fully periodic grid has no numeric channel basis — its
    eigenmodes are analytic; the taught error points at
    :func:`from_model` and the ``sw.transforms`` projections. A
    multi-walled box has no periodic axis left to diagonalize over
    and is rejected the same way :func:`from_model` rejects it.

    Parameters
    ----------
    model : Model
        The assembled shallow-water channel model.
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
        On a fully periodic or multi-walled grid.
    """
    bounded = _bounded_names(model.grid)
    if not bounded:
        raise ValueError(
            "sw.eigenbasis is the numeric labeled eigenbasis of the "
            "walled channel; this grid is fully periodic — use the "
            "analytic eigenmodes instead "
            "(sw.eigenmodes.from_model(model)) and the "
            "sw.transforms projections")
    if len(bounded) > 1:
        raise ValueError(
            "sw.eigenbasis serves the single-walled channel; this "
            f"grid bounds {bounded!r} — a multi-walled box has no "
            "periodic axis left to diagonalize over")
    return ChannelEigenmodes(model, at_time=at_time)


def from_model(
    model: Model, *, at_time: float = 0.0,
) -> Eigenmodes | ChannelEigenmodes:
    r"""
    Build the eigenmodes of an assembled shallow-water model.

    Description
    -----------
    Dispatches on the grid topology. A fully periodic grid gets the
    analytic operator-sourced :class:`Eigenmodes`; the parameters
    are read through ``model.parameters`` with structural
    validation — a beta-plane core provides no ``coriolis.f0`` (its
    ``f`` is a field, not Fourier-diagonalizable) and raises here.
    A grid with exactly one bounded (walled) axis gets the numeric
    :class:`~fridom.shallowwater2.channel_eigenmodes.ChannelEigenmodes`
    (the labeled dense-column channel eigenbasis, beta-plane
    included). A multi-walled box has no periodic axis left to
    diagonalize over and is rejected. Ramp-valued parameters demand
    an explicit ``at_time`` (a fixed-time snapshot).

    Parameters
    ----------
    model : Model
        The assembled shallow-water model.
    at_time : float, optional
        The clock time at which to freeze time-dependent parameters
        (default: 0.0).

    Returns
    -------
    Eigenmodes | ChannelEigenmodes
        The analytic eigenmodes (fully periodic) or the labeled
        channel eigenmodes (one bounded axis).

    Raises
    ------
    ValueError
        On a multi-walled grid, or — on the fully periodic path —
        if ``coriolis.f0`` or ``shallowwater.csqr`` is not provided
        (a non-constant-coefficient system).
    """
    bounded = _bounded_names(model.grid)
    if len(bounded) > 1:
        raise ValueError(
            "shallow-water eigenmodes serve the fully periodic grid "
            "(analytic) or the single-walled channel (numeric); "
            f"this grid bounds {bounded!r} — a multi-walled box has "
            "no periodic axis left to diagonalize over")
    if bounded:
        return ChannelEigenmodes(model, at_time=at_time)
    view = model.parameters
    for name, why in (
        (fr.model.params.CORIOLIS_F0,
         "a constant Coriolis parameter (a beta-plane f(y) is not "
         "Fourier-diagonalizable); assemble with "
         "sw.modules.FPlaneCoriolis"),
        (sw_params.CSQR,
         "a constant squared phase speed (a variable-depth "
         "csqr(y) is not Fourier-diagonalizable; it is served "
         "only on the single-walled channel, by sw.eigenbasis); "
         "assemble with a constant-depth DynamicalCore"),
    ):
        if name not in view:
            raise ValueError(
                f"shallow-water eigenmodes need {why}: no {name!r} "
                "provider on this model")
    f0 = resolve_at(view[fr.model.params.CORIOLIS_F0], at_time)
    csqr = resolve_at(view[sw_params.CSQR], at_time)
    return Eigenmodes(model.grid, f0=f0, csqr=csqr)
