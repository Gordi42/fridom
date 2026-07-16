"""Correctness oracle for the branchless-upwind-tax lowerings (CPU).

Validates, host-side against fridom's own coefficient tables and array
kernels, the two algebraic reformulations of upwind-biased advection:

  (A) SELECTED-INPUT MIRROR TRICK
      recon_right(window) == recon_left(reversed(window)) for the
      mirror-symmetric biased schemes (linear order 3/5, WENO 3/5), so
      the per-face upwind select can be pushed to the *inputs* (one
      cheap `where` per stencil tap on shifted slices of the union
      window) and a SINGLE left-biased reconstruction run.

  (B) DISSIPATION FORM (linear upwind only)
      upwind_flux == u * (c_sym . q_U) - |u| * (c_diss . q_U), exact,
      where q_U is the (order+1)-cell union window straddling the face,
      c_sym = (L_U + R_U)/2 is the centered order+1 row and
      c_diss = (R_U - L_U)/2 the difference row.

Run standalone on CPU:

    JAX_PLATFORMS=cpu .venv/bin/python \
        design/research/stencil_lowering/microbench/phase3_prep/\
validate_spellings.py

Exits nonzero on any failed assertion.
"""
from __future__ import annotations

import os
from fractions import Fraction

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

# fridom coefficient tables + array kernels (host-side, exact rows)
from fridom.nonhydro2.modules.advection import (
    _centered_row,
    _linear_row,
)
from fridom.spatial.operators.weno import (
    _shu_row,
    weno_reconstruct,
    weno_tables,
)

RNG = np.random.default_rng(20260716)
ORDERS = (3, 5)
FTOL = 1e-13  # float64 relative/absolute tolerance for kernel identities


# ================================================================
#  Exact rational rows (Fraction reconstruction of the float tables)
# ================================================================
def _linear_row_exact(order: int, bias: str) -> tuple[Fraction, ...]:
    """Rebuild `_linear_row` in exact Fraction arithmetic.

    Mirrors advection._linear_row / weno.weno_tables but keeps
    rationals so the dissipation-form solve is exact.
    """
    r = (order + 1) // 2
    offsets = tuple(range(r))
    coeffs = tuple(_shu_row(r, r - m) for m in range(r))          # Fraction
    optimal = {2: (Fraction(1, 3), Fraction(2, 3)),
               3: (Fraction(1, 10), Fraction(3, 5),
                   Fraction(3, 10))}[r]
    if bias == "right":
        offsets = tuple(r - 1 - m for m in range(r))
        coeffs = tuple(tuple(reversed(row)) for row in coeffs)
    row = [Fraction(0)] * order
    for m, offset in enumerate(offsets):
        for i, c in enumerate(coeffs[m]):
            row[offset + i] += optimal[m] * c
    return tuple(row)


def _fmt(frac_row: tuple[Fraction, ...]) -> str:
    return "[" + ", ".join(str(f) for f in frac_row) + "]"


def _w1(window: np.ndarray, order: int, bias: str) -> float:
    """One reconstructed face value from a length-``order`` window."""
    out = np.asarray(weno_reconstruct(window.copy(), 0, order=order,
                                      bias=bias))
    return float(out.reshape(-1)[0])


# ================================================================
#  Task 1 — mirror symmetry of the biased rows
# ================================================================
def check_linear_mirror() -> None:
    print("=" * 64)
    print(" (1) LINEAR ROW MIRROR SYMMETRY")
    print("=" * 64)
    for order in ORDERS:
        left = _linear_row(order, "left")
        right = _linear_row(order, "right")
        rev_left = tuple(reversed(left))
        maxdiff = max(abs(a - b) for a, b in zip(right, rev_left))
        print(f"order {order}:")
        print(f"  left  = {left}")
        print(f"  right = {right}")
        print(f"  max|right - reversed(left)| = {maxdiff:.3e}")
        assert maxdiff == 0.0, (order, maxdiff)

        # numeric: right row on window == left row on reversed window
        w = RNG.standard_normal(order)
        v_right = float(np.dot(right, w))
        v_left_rev = float(np.dot(left, w[::-1]))
        assert abs(v_right - v_left_rev) < FTOL, order
    print("  VERDICT: linear order-3/5 right row == exact reversal of "
          "left row (bitwise, floats reversed).\n")


def check_weno_mirror() -> None:
    print("=" * 64)
    print(" (1) WENO TABLE + RECONSTRUCTION MIRROR SYMMETRY")
    print("=" * 64)
    for order in ORDERS:
        tl = weno_tables(order, "left")
        tr = weno_tables(order, "right")
        r = len(tl.optimal)
        # offsets: right = r-1-m ; coeffs/beta rows reversed per candidate
        assert tr.offsets == tuple(r - 1 - m for m in range(r))
        assert tr.optimal == tl.optimal  # d_m stay attached to candidate m
        for m in range(r):
            assert tr.coeffs[m] == tuple(reversed(tl.coeffs[m]))
            for d in range(len(tl.beta_rows[m])):
                assert tr.beta_rows[m][d] == tuple(
                    reversed(tl.beta_rows[m][d]))
        assert tr.beta_scale == tl.beta_scale

        # numeric nonlinear check: right-recon(w) == left-recon(rev(w))
        # over many random windows (single output point, axis length =
        # order so exactly one window).
        maxrel = 0.0
        for _ in range(4000):
            w = RNG.standard_normal(order)
            vr = _w1(w, order, "right")
            vl = _w1(w[::-1], order, "left")
            denom = max(1.0, abs(vr))
            maxrel = max(maxrel, abs(vr - vl) / denom)
        print(f"order {order}: offsets/coeffs/beta mirror OK; "
              f"max rel |right(w) - left(rev w)| = {maxrel:.3e}")
        assert maxrel < FTOL, (order, maxrel)
    print("  VERDICT: WENO-3/5 right kernel == left kernel on the "
          "reversed window (nonlinear machinery mirrors).\n")


# ================================================================
#  Task 2 — dissipation form (linear upwind)
# ================================================================
def _union_embeddings(order: int) -> tuple[
        tuple[Fraction, ...], tuple[Fraction, ...]]:
    """Embed left/right rows into the (order+1)-cell union window.

    Union U = cells [F - order//2 .. F + order//2] (order+1 cells,
    index k = 0..order). Left recon reads U[0..order-1] (drops the
    rightmost cell); right recon reads U[1..order] (drops the
    leftmost). Returns (L_U, R_U), each length order+1.
    """
    left = _linear_row_exact(order, "left")
    right = _linear_row_exact(order, "right")
    lu = (*left, Fraction(0))          # pad a zero on the right
    ru = (Fraction(0), *right)         # pad a zero on the left
    return lu, ru


def derive_dissipation() -> dict[int, tuple]:
    print("=" * 64)
    print(" (2) DISSIPATION FORM  flux = u*(c_sym.q) - |u|*(c_diss.q)")
    print("=" * 64)
    rows = {}
    for order in ORDERS:
        lu, ru = _union_embeddings(order)
        c_sym = tuple((a + b) / 2 for a, b in zip(lu, ru))
        c_diss = tuple((b - a) / 2 for a, b in zip(lu, ru))
        # fridom's own centered order+1 row (its _centered_row table)
        centered = tuple(Fraction(c).limit_denominator(10**9)
                         for c in _centered_row(order + 1))
        centered_exact = _shu_row(order + 1, (order + 1) // 2)

        print(f"order {order}  (union width {order + 1}):")
        print(f"  c_sym  = {_fmt(c_sym)}")
        print(f"  c_diss = {_fmt(c_diss)}")
        print(f"  _centered_row({order + 1}) exact = "
              f"{_fmt(centered_exact)}")
        print(f"  c_sym == _centered_row({order + 1}) : "
              f"{c_sym == centered_exact}")

        # numeric validation over random windows and both signs
        maxerr = 0.0
        for _ in range(5000):
            q = RNG.standard_normal(order + 1)
            u = RNG.standard_normal()
            # reference: fridom's Where(positive, left, right) * u
            left = _linear_row(order, "left")
            right = _linear_row(order, "right")
            fv_left = float(np.dot(left, q[:order]))
            fv_right = float(np.dot(right, q[1:]))
            positive = u + abs(u)
            fv = fv_left if positive > 0 else fv_right
            flux_ref = u * fv
            # dissipation form
            cs = np.array([float(c) for c in c_sym])
            cd = np.array([float(c) for c in c_diss])
            flux_diss = u * float(np.dot(cs, q)) - abs(u) * float(
                np.dot(cd, q))
            maxerr = max(maxerr, abs(flux_ref - flux_diss))
        print(f"  max|flux_ref - flux_diss| over 5000 (both signs) = "
              f"{maxerr:.3e}")
        assert maxerr < FTOL, (order, maxerr)
        assert c_sym == centered_exact, order
        rows[order] = (c_sym, c_diss)
        print()
    print("  VERDICT: exact dissipation split; c_sym reuses fridom's "
          "own _centered_row(order+1) table.\n")
    return rows


# ================================================================
#  Task 4 — end-to-end oracle: both-then-select vs the lowerings
# ================================================================
def _linear_face_both_select(q_u: np.ndarray, u: np.ndarray,
                             order: int) -> np.ndarray:
    """Reference spelling: compute both rows, select by sign of u.

    q_u has the union window along axis 0 (length order+1 + extra for
    multiple output points); u is the face velocity per output point.
    Reproduces fridom's Where(v+|v|, left, right).
    """
    left = np.array(_linear_row(order, "left"))
    right = np.array(_linear_row(order, "right"))
    npts = q_u.shape[0] - order            # union width order+1 per point
    fv_left = np.zeros(npts)
    fv_right = np.zeros(npts)
    for t in range(npts):
        win = q_u[t:t + order + 1]         # order+1 union cells
        fv_left[t] = np.dot(left, win[:order])
        fv_right[t] = np.dot(right, win[1:])
    positive = u + np.abs(u)
    return np.where(positive > 0, fv_left, fv_right)


def _linear_face_selected_input(q_u: np.ndarray, u: np.ndarray,
                                order: int) -> np.ndarray:
    """Lowering (A): per-tap select then ONE left reconstruction.

    tap i (i=0..order-1) = where(u>0, U[i], U[order - i]).
    """
    left = np.array(_linear_row(order, "left"))
    npts = q_u.shape[0] - order
    positive = (u + np.abs(u)) > 0
    out = np.zeros(npts)
    for t in range(npts):
        U = q_u[t:t + order + 1]
        taps = np.where(positive[t], U[:order],
                        U[order - np.arange(order)])
        out[t] = np.dot(left, taps)
    return out


def _linear_face_dissipation(q_u: np.ndarray, u: np.ndarray,
                             order: int,
                             c_sym: np.ndarray,
                             c_diss: np.ndarray) -> np.ndarray:
    """Lowering (B): flux/u face value = c_sym.U - sign(u)*c_diss.U.

    Returned as a face *value* (flux without the leading u factor) so
    it is directly comparable to the both-then-select face value; the
    module multiplies by u afterwards.
    """
    npts = q_u.shape[0] - order
    s = np.sign(u)
    s = np.where(s == 0, -1.0, s)          # v=0 tie -> right branch
    out = np.zeros(npts)
    for t in range(npts):
        U = q_u[t:t + order + 1]
        out[t] = np.dot(c_sym, U) - s[t] * np.dot(c_diss, U)
    return out


def _weno_face_both_select(q_u: np.ndarray, u: np.ndarray,
                           order: int) -> np.ndarray:
    """WENO reference: both biased recons, select by sign of u."""
    npts = q_u.shape[0] - order
    out = np.zeros(npts)
    positive = (u + np.abs(u)) > 0
    for t in range(npts):
        U = q_u[t:t + order + 1]
        vl = _w1(U[:order], order, "left")
        vr = _w1(U[1:], order, "right")
        out[t] = vl if positive[t] else vr
    return out


def _weno_face_selected_input(q_u: np.ndarray, u: np.ndarray,
                              order: int) -> np.ndarray:
    """WENO lowering (A): per-tap select then ONE left reconstruction."""
    npts = q_u.shape[0] - order
    out = np.zeros(npts)
    positive = (u + np.abs(u)) > 0
    for t in range(npts):
        U = q_u[t:t + order + 1]
        taps = np.where(positive[t], U[:order],
                        U[order - np.arange(order)])
        out[t] = _w1(taps, order, "left")
    return out


def end_to_end_oracle(diss_rows: dict[int, tuple]) -> None:
    print("=" * 64)
    print(" (4) END-TO-END ORACLE (multi-point windows, sign changes)")
    print("=" * 64)
    n = 32
    for order in ORDERS:
        # union field long enough for n output faces + a sign-changing u
        q_u = RNG.standard_normal(n + order)
        u = RNG.standard_normal(n)
        u[n // 3] = 0.0                     # exercise the v=0 tie
        u[: n // 2] = np.abs(u[: n // 2])   # forced-positive block
        u[n // 2:] = -np.abs(u[n // 2:])    # forced-negative block

        ref = _linear_face_both_select(q_u, u, order)
        sel = _linear_face_selected_input(q_u, u, order)
        c_sym = np.array([float(c) for c in diss_rows[order][0]])
        c_diss = np.array([float(c) for c in diss_rows[order][1]])
        dis = _linear_face_dissipation(q_u, u, order, c_sym, c_diss)
        e_sel = float(np.max(np.abs(ref - sel)))
        e_dis = float(np.max(np.abs(ref - dis)))
        print(f"linear order {order}: "
              f"max|ref - selected_input| = {e_sel:.3e} ; "
              f"max|ref - dissipation| = {e_dis:.3e}")
        assert e_sel < FTOL, (order, e_sel)
        assert e_dis < FTOL, (order, e_dis)

        wref = _weno_face_both_select(q_u, u, order)
        wsel = _weno_face_selected_input(q_u, u, order)
        e_wsel = float(np.max(np.abs(wref - wsel)))
        print(f"weno   order {order}: "
              f"max|ref - selected_input| = {e_wsel:.3e}")
        assert e_wsel < FTOL, (order, e_wsel)
    print("\nALL ORACLE ASSERTIONS PASSED.")


def main() -> None:
    check_linear_mirror()
    check_weno_mirror()
    diss_rows = derive_dissipation()
    end_to_end_oracle(diss_rows)
    print("\n" + "=" * 64)
    print(" ALL CHECKS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
