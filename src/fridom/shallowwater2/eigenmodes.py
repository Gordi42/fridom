r"""
Linear eigenmodes of the shallow-water system (framework2 port).

Description
-----------
The successor of the old ``omega`` / ``vec_q`` / ``vec_p`` machinery
(07_open_threads item 5). Linearizing the shallow-water equations
(:math:`\mathrm{Ro}\to 0`) and Fourier transforming yields the
system matrix

.. math::
    \mathbf{A} = \begin{pmatrix}
        0 & if & k_x \\ -if & 0 & k_y \\ c^2 k_x & c^2 k_y & 0
    \end{pmatrix}

with the geostrophic mode :math:`\omega^0 = 0` and the two
inertia-gravity modes :math:`\omega^\pm = \pm\sqrt{f^2 + c^2 k^2}`.
This module exposes the dispersion relation, the State-valued
eigen/projection vectors, and a projector callable onto a chosen
mode.

Scope note (wave 6): the eigenvectors are built on a **collocated**
spectral basis (all three components on the centre-derived
coefficient space) using the *continuous* formulas — enough for the
dispersion and orthonormality/idempotency smoke tests. Wrapping the
projector as a wave-7 ``fr.StateTransform`` acting on the model's
*staggered* state (which needs the discrete staggered spectral
operators) is deferred.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework2.model.time_dependent import resolve_at
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.model import Model

#: norm floor below which a projection vector is treated as zero
_NORM_FLOOR = 1e-10


class Eigenmodes:

    r"""
    Continuous shallow-water eigenmodes on a periodic grid.

    Parameters
    ----------
    grid : Grid
        A periodic (Fourier-diagonalizable) grid.
    f0 : float
        The constant Coriolis parameter :math:`f`.
    csqr : float
        The squared phase speed :math:`c^2`.
    """

    def __init__(self, grid: Grid, *, f0: float, csqr: float) -> None:
        """Build the coefficient template and wavenumber arrays."""
        self.grid = grid
        self.f0 = float(f0)
        self.csqr = float(csqr)
        if self.f0 == 0.0 and self.csqr == 0.0:
            raise ValueError(
                "degenerate eigenmode system: f0 == csqr == 0")
        center = fr.Collocated().resolve(grid)
        self._transform = grid.dispatch.resolve("transform", center)
        self._template = self._transform.forward(
            grid.create_field(center))
        coeff = self._template.function_space
        names = grid.names
        self._kx = grid.wavenumbers(coeff, names[0]).data
        self._ky = grid.wavenumbers(coeff, names[1]).data

    # ================================================================
    #  Dispersion relation
    # ================================================================
    def omega(
        self, s: int, kx: object = None, ky: object = None,
    ) -> jnp.ndarray:
        r"""
        Return the eigenvalue :math:`\omega^s` (0, +, -).

        Parameters
        ----------
        s : int
            The mode: 0 (geostrophic), +1 / -1 (inertia-gravity).
        kx, ky : array-like, optional
            Wavenumbers; default to the grid's spectral wavenumbers.

        Returns
        -------
        jnp.ndarray
            The eigenvalue(s).
        """
        kx = self._kx if kx is None else jnp.asarray(kx)
        ky = self._ky if ky is None else jnp.asarray(ky)
        if s == 0:
            return jnp.zeros_like(kx * ky * 1.0)
        return s * jnp.sqrt(
            self.f0 ** 2 + self.csqr * (kx ** 2 + ky ** 2))

    # ================================================================
    #  Eigen- and projection vectors (State-valued)
    # ================================================================
    def q(self, s: int) -> State:
        """Return the eigenvector State for mode ``s`` (coeff space)."""
        qu, qv, qp = self._q_arrays(s)
        return self._state(qu, qv, qp)

    def p(self, s: int) -> State:
        """Return the projection-vector State for mode ``s``."""
        pu, pv, pp = self._p_arrays(s)
        return self._state(pu, pv, pp)

    def projector(self, s: int) -> Callable[[State], State]:
        r"""
        Return the spectral projector onto mode ``s``.

        Description
        -----------
        :math:`P_s \boldsymbol{z} = \boldsymbol{q}^s\,
        (\boldsymbol{p}^s{}^* \cdot \boldsymbol{z})`, diagonal in
        wavenumber, acting on a State on this object's collocated
        coefficient space. Idempotent by biorthonormality
        (:math:`\boldsymbol{p}^s{}^* \cdot \boldsymbol{q}^s = 1`).
        """
        qu, qv, qp = self._q_arrays(s)
        pu, pv, pp = self._p_arrays(s)

        def project(z: State) -> State:
            """Project ``z`` onto mode ``s`` (pointwise per mode)."""
            coeff = (jnp.conj(pu) * z["u"].data
                     + jnp.conj(pv) * z["v"].data
                     + jnp.conj(pp) * z["p"].data)
            return self._state(qu * coeff, qv * coeff, qp * coeff)

        return project

    # ================================================================
    #  Internals
    # ================================================================
    def _state(self, u_arr, v_arr, p_arr) -> State:  # noqa: ANN001
        """Wrap three coefficient arrays into a State."""
        return State({
            "u": self._template.with_data(u_arr).with_metadata(
                name="u"),
            "v": self._template.with_data(v_arr).with_metadata(
                name="v"),
            "p": self._template.with_data(p_arr).with_metadata(
                name="p")})

    def _q_arrays(
        self, s: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Continuous eigenvector arrays (u, v, p) for mode ``s``."""
        kx, ky, f = self._kx, self._ky, self.f0
        om = self.omega(s, kx, ky)
        qu = om * kx - 1j * f * ky
        qv = om * ky + 1j * f * kx
        qp = f ** 2 - om ** 2
        # inertial modes at k = 0
        qu0, qv0, qp0 = -1j * s, s ** 2 * 1.0, 1.0 - s ** 2
        nonzero = (kx ** 2 + ky ** 2) != 0
        qu = jnp.where(nonzero, qu, qu0)
        qv = jnp.where(nonzero, qv, qv0)
        qp = jnp.where(nonzero, qp, qp0)
        return qu, qv, qp

    def _energy_weights(self) -> dict[str, float]:
        r"""Per-component energy weights (the ``fr.EnergyMetric`` diag).

        Description
        -----------
        ``diag(1, 1, 1/c^2)`` on ``(u, v, p)`` -- the canonical
        shallow-water energy metric ``M`` (a single source of truth
        with ``fr.EnergyMetric.from_model``); ``p = M q`` scales ``q``
        by these. The ``1/c^2`` reciprocal falls back to ``1`` for the
        degenerate ``c^2 = 0`` (no-gravity) case the constructor permits
        -- the metric proper (and ``fr.EnergyMetric``) needs
        ``c^2 != 0``.
        """
        inv_csqr = 1.0 / self.csqr if self.csqr != 0.0 else 1.0
        return {"u": 1.0, "v": 1.0, "p": inv_csqr}

    def _p_arrays(
        self, s: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        r"""Projection-vector arrays: derived ``p = M q`` normalized.

        Description
        -----------
        ``p^s = M q^s / <q^s, q^s>_M`` with ``M`` the energy metric
        (see :meth:`_energy_weights`); no hand-written formula. The
        per-mode energy norm ``<q, q>_M = sum_c w_c |q_c|^2`` is a
        keepdims component contraction, not the global
        ``fr.EnergyMetric.inner``.
        """
        qu, qv, qp = self._q_arrays(s)
        w = self._energy_weights()
        mqu, mqv, mqp = w["u"] * qu, w["v"] * qv, w["p"] * qp
        qq_m = jnp.real(jnp.conj(qu) * mqu + jnp.conj(qv) * mqv
                        + jnp.conj(qp) * mqp)
        good = qq_m > _NORM_FLOOR
        scale = jnp.where(good, 1.0 / jnp.where(good, qq_m, 1.0), 0.0)
        return mqu * scale, mqv * scale, mqp * scale


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
