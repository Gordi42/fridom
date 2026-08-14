"""Row-by-row consistency of the mapped flux divergence (A0 diagnostic).

Manufactured solution on ``zp = z H(x)``: the stream function
``psi = f(z)`` with ``f(0) = f(1) = 0`` gives a physically
divergence-free flow whose contravariant column flux
``Omega = w - Z_x u`` vanishes IDENTICALLY, so both walls are exact
streamlines.  Advecting ``q = zp`` then has the closed-form tendency
``-w``.  Comparing the module's tendency against it row by row shows
where the discretization is consistent and where it is not.
"""
from __future__ import annotations

import sys

import numpy as np

import fridom as fr
from fridom.model.modules.advection import CenteredAdvection

from clean_repro import make_mapped_model

PI = np.pi


def fields(n, ny, amp):
    """Analytic (u, v, w, b) samples on the C-grid staggering."""
    dx = 2 * PI / n
    xc = (np.arange(n) + 0.5) * dx
    xr = (np.arange(n) + 1.0) * dx
    zc = (np.arange(n) + 0.5) / n
    zi = (np.arange(1, n)) / n          # Inner(z): interior faces

    def h(x):
        return 1.0 + amp * np.sin(x)

    def hp(x):
        return amp * np.cos(x)

    # u on Right(x) (x) Center(y) (x) Center(z)
    X, Z = np.meshgrid(xr, zc, indexing="ij")
    u = (PI * np.cos(PI * Z) / h(X))[:, None, :] * np.ones((1, ny, 1))
    # w on Center(x) (x) Center(y) (x) Inner(z)
    X, Z = np.meshgrid(xc, zi, indexing="ij")
    w = (PI * np.cos(PI * Z) * Z * hp(X) / h(X))[:, None, :] \
        * np.ones((1, ny, 1))
    # b = zp on Center(x) (x) Center(y) (x) Center(z)
    X, Z = np.meshgrid(xc, zc, indexing="ij")
    b = (Z * h(X))[:, None, :] * np.ones((1, ny, 1))
    # exact tendency -w, on the CELL CENTERS
    wc = (PI * np.cos(PI * Z) * Z * hp(X) / h(X))[:, None, :] \
        * np.ones((1, ny, 1))
    return u, w, b, -wc


def probe(n, *, ny=4, amp=0.2):
    model = make_mapped_model(n, 0.01, ny=ny, amp=amp)
    u, w, b, exact = fields(n, ny, amp)
    model.set_fields(u=u, v=0.0 * b, w=w, b=b)
    tau = model.tendency(
        model.state, constraints=False,
        filter=fr.model.term_predicates.owned_by(CenteredAdvection))
    got = np.asarray(tau["b"].data)
    err = np.abs(got - exact).max(axis=(0, 1))
    return err


def main():
    print("row-wise max |tendency(b) - exact| ; b = zp, Omega == 0 flow")
    for n in (16, 32, 64):
        err = probe(n)
        print(f"\nn = {n}")
        print(f"  bottom row  iz=0     : {err[0]:.4e}")
        print(f"  interior    iz=n/2   : {err[n // 2]:.4e}")
        print(f"  top row     iz=n-1   : {err[-1]:.4e}")
        print(f"  interior max (1..n-2): {err[1:-1].max():.4e}")
    print("\nflat-metric control (amp = 0):")
    for n in (16, 32):
        err = probe(n, amp=0.0)
        print(f"  n={n}: bottom {err[0]:.3e} top {err[-1]:.3e} "
              f"interior {err[1:-1].max():.3e}")


if __name__ == "__main__":
    sys.exit(main())
