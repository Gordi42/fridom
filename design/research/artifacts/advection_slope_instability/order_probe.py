"""Row-wise order of the mapped transport, impermeable vs not (A0)."""
from __future__ import annotations

import sys

import numpy as np

import fridom as fr
from fridom.model.modules.advection import CenteredAdvection

from clean_repro import make_mapped_model

L = 2 * np.pi


def tendency(model):
    return model.tendency(
        model.state, constraints=False,
        filter=fr.model.term_predicates.owned_by(CenteredAdvection))


def case(n, *, impermeable, ny=4, amp=0.2, U=0.4):
    """u = U uniform, b = cos(x) zp; w chosen to set Omega = 0 or 0."""
    m = make_mapped_model(n, 0.01, ny=ny, amp=amp)
    dx = L / n
    xc = (np.arange(n) + 0.5) * dx
    xr = (np.arange(n) + 1.0) * dx
    zc = (np.arange(n) + 0.5) / n
    zi = np.arange(1, n) / n
    ones = np.ones((1, ny, 1))

    def h(x):
        return 1.0 + amp * np.sin(x)

    def hp(x):
        return amp * np.cos(x)

    x, z = np.meshgrid(xc, zc, indexing="ij")
    zp = z * h(x)
    b = (np.cos(x) * zp)[:, None, :] * ones
    exact = (U * np.sin(x) * zp)[:, None, :] * ones
    fields = {"u": U + 0.0 * b, "b": b}
    if impermeable:
        xw, zw = np.meshgrid(xc, zi, indexing="ij")
        fields["w"] = (zw * hp(xw) * U)[:, None, :] * ones
        # -div(F) = U sin(x) zp - 2 U cos(x) H' zp / H
        exact = exact - (2 * U * np.cos(x) * hp(x) * zp / h(x))[:, None, :] \
            * ones
    m.set_fields(**fields)
    got = np.asarray(tendency(m)["b"].data)
    err = np.abs(got - exact)
    dv = (1.0 / n) * dx
    return (err.max(axis=(0, 1)),
            float(np.sqrt((err[:, 0, :] ** 2).sum() * dv)))


def report(tag, impermeable):
    print(f"\n--- {tag} ---")
    prev = None
    for n in (16, 32, 64):
        rows, l2 = case(n, impermeable=impermeable)
        line = (f"n={n:3d}  all-max {rows.max():.4e}  "
                f"interior-max {rows[1:-1].max():.4e}  "
                f"bottom {rows[0]:.4e}  top {rows[-1]:.4e}  L2 {l2:.4e}")
        if prev is not None:
            line += (f"   orders: all {np.log2(prev[0] / rows.max()):+.2f}"
                     f" int {np.log2(prev[1] / rows[1:-1].max()):+.2f}"
                     f" top {np.log2(prev[2] / rows[-1]):+.2f}"
                     f" L2 {np.log2(prev[3] / l2):+.2f}")
        prev = (rows.max(), rows[1:-1].max(), rows[-1], l2)
        print(line)


def main():
    report("u = U, w = 0  (violates impermeability at the sloping lid)",
           False)
    report("u = U, w = z H' U  (Omega == 0, impermeable)", True)


if __name__ == "__main__":
    sys.exit(main())
