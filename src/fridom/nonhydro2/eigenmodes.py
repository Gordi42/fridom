"""Discrete-dispersion eigenmodes (the vec_q / vec_p / omega successor).

Description
-----------
The successor of the old ``eigenvectors.py`` (``vec_q``, ``vec_p``,
``omega``). Framework2 lands no ``Symbol`` cluster
(``operators/symbol.py`` is a stub), and the model-layer eigenmode
surface is an open thread (07_open_threads item 5), so the eigenmodes
are assembled here directly from the grid's discrete wavenumbers.

This wave exposes the eigenmode **data** (``em.q(s)`` / ``em.p(s)`` as
component arrays, ``em.omega(s)`` / ``em.omega_at(k, s)``) and a
``em.projector(s)`` **callable** (spectral state -> spectral state).
Wrapping the projector as an ``fr.StateTransform`` (``nh.transforms``)
is deferred to wave 7 with the transform algebra.

Discrete operators (C-grid, ``use_discrete=True``), per axis with
spacing ``dx``:
``one_hat2 = (1 + cos k dx)/2``,
``k_hat(+/-) = -/+ i (1 - e^{+/- i k dx}) / dx``,
``k_hat2 = 2 (1 - cos k dx) / dx^2``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.model.params import (
    CORIOLIS_F0,
    STRATIFICATION_N2,
)
from fridom.framework2.model.time_dependent import TimeDependent
from fridom.nonhydro2.params import DSQR

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.model import Model


# ================================================================
#  Discrete spectral operators
# ================================================================
def _one_hat(k: jax.Array, dx: float, sign: int) -> jax.Array:
    """``(1 + e^{+/- i k dx}) / 2`` — the averaging symbol (sign +/-1)."""
    return 0.5 * (1.0 + jnp.exp(1j * sign * k * dx))


def _one_hat2(k: jax.Array, dx: float) -> jax.Array:
    """``(1 + cos k dx) / 2`` — the squared averaging symbol."""
    return 0.5 * (1.0 + jnp.cos(k * dx))


def _k_hat(k: jax.Array, dx: float, sign: int) -> jax.Array:
    """``-/+ i (1 - e^{+/- i k dx}) / dx`` (sign = +1 / -1)."""
    return -1j * sign * (1.0 - jnp.exp(1j * sign * k * dx)) / dx


def _k_hat2(k: jax.Array, dx: float) -> jax.Array:
    """``2 (1 - cos k dx) / dx^2`` — the squared derivative symbol."""
    return 2.0 * (1.0 - jnp.cos(k * dx)) / dx**2


class Eigenmodes:

    """Discrete inertia-gravity / geostrophic eigenmodes on a grid.

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
        """Precompute the discrete wavenumber symbols on the grid."""
        if float(f0) == 0.0 and float(n2) == 0.0:
            raise ValueError(
                "an eigenmode set needs f0 != 0 or n2 != 0 "
                "(both zero has no inertia-gravity structure)")
        self.f0 = float(f0)
        self.n2 = float(n2)
        self.dsqr = float(dsqr)
        self._vertical = vertical

        template = grid.create_field()
        bare = template.function_space.bare
        names = bare.names
        if vertical not in names or len(names) != 3:  # noqa: PLR2004
            raise ValueError(
                "eigenmodes need a 3-D grid with the vertical "
                f"coordinate {vertical!r}; got names {names}")
        self.names = names

        khatp: dict[str, jax.Array] = {}
        khatm: dict[str, jax.Array] = {}
        khat2: dict[str, jax.Array] = {}
        one_p: dict[str, jax.Array] = {}
        one_m: dict[str, jax.Array] = {}
        one2: dict[str, jax.Array] = {}
        dx_map: dict[str, float] = {}
        for index, name in enumerate(names):
            factor = bare.factor(name)
            n = factor.shape[0]
            length = factor.mesh.extent[1] - factor.mesh.extent[0]
            dx = length / n
            modes = jnp.fft.fftfreq(n, d=1.0 / n)
            k1d = ((2.0 * jnp.pi / length) * modes).reshape(
                tuple(n if i == index else 1 for i in range(3)))
            khatp[name] = _k_hat(k1d, dx, +1)
            khatm[name] = _k_hat(k1d, dx, -1)
            khat2[name] = _k_hat2(k1d, dx)
            one_p[name] = _one_hat(k1d, dx, +1)
            one_m[name] = _one_hat(k1d, dx, -1)
            one2[name] = _one_hat2(k1d, dx)
            dx_map[name] = dx
        self._khatp = khatp
        self._khatm = khatm
        self._khat2 = khat2
        self._one_p = one_p
        self._one_m = one_m
        self._one2 = one2
        self._dx_map = dx_map

    # ================================================================
    #  Dispersion
    # ================================================================
    def omega(self, s: int = 1) -> jax.Array:
        """Discrete frequency field ``omega^s`` over the spectral grid.

        Description
        -----------
        ``s = 0`` is geostrophic (zero frequency); ``s = +/-1`` are the
        inertia-gravity branches.
        """
        if s == 0:
            return jnp.zeros_like(jnp.real(self._sum_kh2()))
        x, y, z = self.names
        kh2 = self._sum_kh2()
        coriolis = (self._one2[x] * self._one2[y] * self.f0**2
                    * self._khat2[z])
        buoyancy = self._one2[z] * self.n2 * kh2
        denom = self.dsqr * kh2 + self._khat2[z]
        nonzero = self._nonzero_mask()
        safe = jnp.where(denom == 0, 1.0, denom)
        om = jnp.sqrt((coriolis + buoyancy) / safe)
        om = jnp.where(nonzero, om, 0.0)
        return s * om

    def omega_at(self, k: tuple[float, float, float], s: int = 1,
                 ) -> complex:
        """Scalar frequency at a physical wavevector ``k = (kx,ky,kz)``.

        Description
        -----------
        Evaluates the discrete dispersion relation at one wavevector
        (the continuous-symbol convenience accessor).
        """
        if s == 0:
            return 0.0
        x, y, z = self.names
        dx = {name: self._dx(name) for name in self.names}
        kmap = dict(zip(self.names, k, strict=True))
        one2 = {n: _one_hat2(jnp.asarray(kmap[n]), dx[n])
                for n in self.names}
        khat2 = {n: _k_hat2(jnp.asarray(kmap[n]), dx[n])
                 for n in self.names}
        kh2 = khat2[x] + khat2[y]
        coriolis = one2[x] * one2[y] * self.f0**2 * khat2[z]
        buoyancy = one2[z] * self.n2 * kh2
        denom = self.dsqr * kh2 + khat2[z]
        if float(jnp.real(denom)) == 0.0:
            return 0.0
        return complex(s * jnp.sqrt((coriolis + buoyancy) / denom))

    # ================================================================
    #  Eigenvectors
    # ================================================================
    def q(self, s: int = 1) -> dict[str, jax.Array]:
        """Return the ``s``-mode eigenvector ``q^s`` (component arrays)."""
        return self._vec_q(s)

    def p(self, s: int = 1) -> dict[str, jax.Array]:
        """Return the ``s``-mode projection vector ``p^s`` (arrays).

        Description
        -----------
        Normalized so ``<p^s, q^s> = 1`` (inner product = sum over the
        nonzero modes of ``conj(p) . q``), giving the biorthogonal
        projector.
        """
        raw = self._vec_p(s)
        q = self._vec_q(s)
        norm = self._pair(raw, q)
        # guard the degenerate modes (kh = 0 for s != 0, and the mean
        # mode): where <p, q> ~ 0 the mode has no representative in this
        # family, so the projector maps it to zero.
        good = jnp.abs(norm) > 1e-9  # noqa: PLR2004
        safe = jnp.where(good, norm, 1.0)
        return {c: jnp.where(good, raw[c] / safe, 0.0) for c in raw}

    def projector(self, s: int = 1):  # noqa: ANN201 — a closure
        """Return the spectral projector ``P^s`` (a state -> state map).

        Description
        -----------
        The returned callable takes a mapping ``{u,v,w,b: array}`` of
        **spectral** component arrays and returns ``q^s <p^s, .>`` —
        the projection onto the ``s`` eigenspace. Wrapping this as an
        ``fr.StateTransform`` (with the forward/inverse transforms) is
        the deferred wave-7 ``nh.transforms`` surface.
        """
        p = self.p(s)
        q = self.q(s)

        def project(
            fields: dict[str, jax.Array],
        ) -> dict[str, jax.Array]:
            amp = sum(jnp.conj(p[c]) * fields[c] for c in p)
            return {c: q[c] * amp for c in q}

        return project

    # ================================================================
    #  Internals
    # ================================================================
    def _dx(self, name: str) -> float:
        return self._dx_map[name]

    def _sum_kh2(self) -> jax.Array:
        x, y, _ = self.names
        return self._khat2[x] + self._khat2[y]

    def _nonzero_mask(self) -> jax.Array:
        x, y, z = self.names
        total = self._khat2[x] + self._khat2[y] + self._khat2[z]
        return total != 0.0

    def _vec_q(self, s: int) -> dict[str, jax.Array]:
        x, y, z = self.names
        op, om_, o2 = self._one_p, self._one_m, self._one2
        kp, km = self._khatp, self._khatm
        if s == 0:
            u = -(op[x] * om_[y] * op[z] * kp[y])
            v = om_[x] * op[y] * op[z] * kp[x]
            w = jnp.zeros_like(u)
            b = o2[x] * o2[y] * self.f0 * kp[z]
            return self._named(u, v, w, b)
        kh2 = self._sum_kh2()
        omega = self.omega(s)
        u = km[z] * (-1j * omega * kp[x]
                     + op[x] * om_[y] * self.f0 * kp[y])
        v = km[z] * (-1j * omega * kp[y]
                     - om_[x] * op[y] * self.f0 * kp[x])
        w = 1j * omega * kh2
        b = om_[z] * self.n2 * kh2
        return self._named(u, v, w, b)

    def _vec_p(self, s: int) -> dict[str, jax.Array]:
        x, y, z = self.names
        op, om_, o2 = self._one_p, self._one_m, self._one2
        kp, km = self._khatp, self._khatm
        if s == 0:
            u = -(op[x] * om_[y] * op[z] * self.n2 * kp[y])
            v = om_[x] * op[y] * op[z] * self.n2 * kp[x]
            w = jnp.zeros_like(u)
            b = o2[x] * o2[y] * self.f0 * kp[z]
            return self._named(u, v, w, b)
        kh2 = self._sum_kh2()
        omega = self.omega(s)
        gamma = ((kh2 + self._khat2[z])
                 / (self.dsqr * kh2 + self._khat2[z]))
        gamma = jnp.where(self._nonzero_mask(), gamma, 0.0)
        u = km[z] * (-1j * omega * kp[x]
                     + op[x] * om_[y] * self.f0 * gamma * kp[y])
        v = km[z] * (-1j * omega * kp[y]
                     - om_[x] * op[y] * self.f0 * gamma * kp[x])
        w = 1j * omega * kh2
        b = om_[z] * gamma * kh2
        return self._named(u, v, w, b)

    def _named(self, u, v, w, b) -> dict[str, jax.Array]:  # noqa: ANN001
        return {"u": u, "v": v, "w": w, "b": b}

    def _pair(
        self, p: dict[str, jax.Array], q: dict[str, jax.Array],
    ) -> jax.Array:
        return sum(jnp.conj(p[c]) * q[c] for c in p)


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
        if isinstance(value, TimeDependent):
            return float(value.at_time(at_time))
        return float(value)

    return Eigenmodes(
        model.grid,
        f0=_read(CORIOLIS_F0),
        n2=_read(STRATIFICATION_N2),
        dsqr=_read(DSQR))
