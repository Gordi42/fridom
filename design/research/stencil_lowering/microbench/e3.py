"""E3/E4: C-grid flux-form advection tendency and reconstructions.

tendency = sum_axis diff_axis( interp_axis(vel) * reconstruct_axis(b) )
with valid-mode windows off ONE halo-padded input per field (H=4).

All fields (u,v,w,b) are passed pre-padded, shape (N+2H,)*3.  The
tendency has interior size N per axis.  Face arrays have size N+1 along
the working axis and N along the others; diff along the axis brings them
back to N.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax

H = 4  # halo per side


def face_win(fp, a, off, n):
    """b at cell (face+off): size (n+1 along a, n along others)."""
    idx = [slice(H, H + n), slice(H, H + n), slice(H, H + n)]
    idx[a] = slice(H + off, H + off + n + 1)
    return fp[tuple(idx)]


def diff_along(face, a):
    """flux[1:]-flux[:-1] along axis a: (n+1)->n along a."""
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[a] = slice(0, face.shape[a] - 1)
    hi[a] = slice(1, face.shape[a])
    return face[tuple(hi)] - face[tuple(lo)]


# ================================================================
#  Reconstructions (return face value, size n+1 along a)
# ================================================================


def interp_vel(velp, a, n):
    return 0.5 * (face_win(velp, a, 0, n) + face_win(velp, a, 1, n))


def recon_centered(bp, a, n, vel):  # noqa: ARG001
    return 0.5 * (face_win(bp, a, 0, n) + face_win(bp, a, 1, n))


def recon_upwind3(bp, a, n, vel):
    wm1 = face_win(bp, a, -1, n)
    w0 = face_win(bp, a, 0, n)
    w1 = face_win(bp, a, 1, n)
    w2 = face_win(bp, a, 2, n)
    left = (-wm1 + 5.0 * w0 + 2.0 * w1) / 6.0
    right = (2.0 * w0 + 5.0 * w1 - w2) / 6.0
    return jnp.where(vel > 0.0, left, right)


def recon_upwind5(bp, a, n, vel):
    wm2 = face_win(bp, a, -2, n)
    wm1 = face_win(bp, a, -1, n)
    w0 = face_win(bp, a, 0, n)
    w1 = face_win(bp, a, 1, n)
    w2 = face_win(bp, a, 2, n)
    w3 = face_win(bp, a, 3, n)
    left = (2.0 * wm2 - 13.0 * wm1 + 47.0 * w0
            + 27.0 * w1 - 3.0 * w2) / 60.0
    right = (2.0 * w3 - 13.0 * w2 + 47.0 * w1
             + 27.0 * w0 - 3.0 * wm1) / 60.0
    return jnp.where(vel > 0.0, left, right)


def _weno_js(b):
    """WENO-JS reconstruction from 5 windows [b_{-2..+2}] -> face value."""
    bm2, bm1, b0, bp1, bp2 = b
    eps = 1e-6
    c13 = 13.0 / 12.0
    p0 = (2.0 * bm2 - 7.0 * bm1 + 11.0 * b0) / 6.0
    p1 = (-bm1 + 5.0 * b0 + 2.0 * bp1) / 6.0
    p2 = (2.0 * b0 + 5.0 * bp1 - bp2) / 6.0
    beta0 = c13 * (bm2 - 2.0 * bm1 + b0) ** 2 \
        + 0.25 * (bm2 - 4.0 * bm1 + 3.0 * b0) ** 2
    beta1 = c13 * (bm1 - 2.0 * b0 + bp1) ** 2 \
        + 0.25 * (bm1 - bp1) ** 2
    beta2 = c13 * (b0 - 2.0 * bp1 + bp2) ** 2 \
        + 0.25 * (3.0 * b0 - 4.0 * bp1 + bp2) ** 2
    a0 = 0.1 / (eps + beta0) ** 2
    a1 = 0.6 / (eps + beta1) ** 2
    a2 = 0.3 / (eps + beta2) ** 2
    asum = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / asum


def recon_weno5(bp, a, n, vel):
    wm2 = face_win(bp, a, -2, n)
    wm1 = face_win(bp, a, -1, n)
    w0 = face_win(bp, a, 0, n)
    w1 = face_win(bp, a, 1, n)
    w2 = face_win(bp, a, 2, n)
    w3 = face_win(bp, a, 3, n)
    left = _weno_js([wm2, wm1, w0, w1, w2])
    right = _weno_js([w3, w2, w1, w0, wm1])
    return jnp.where(vel > 0.0, left, right)


RECON = {
    "centered2": recon_centered,
    "upwind3": recon_upwind3,
    "upwind5": recon_upwind5,
    "weno5": recon_weno5,
}


# ================================================================
#  Tendency spellings
# ================================================================


def tendency_composed(up, vp, wp, bp, recon, n):
    """(a) composed: per-op functions, one jit over the whole tendency."""
    vels = [up, vp, wp]
    tend = None
    for a in range(3):
        vel = interp_vel(vels[a], a, n)
        rb = recon(bp, a, n, vel)
        flux = vel * rb
        term = -diff_along(flux, a)
        tend = term if tend is None else tend + term
    return tend


def tendency_barrier(up, vp, wp, bp, recon, n):
    """(c) composed with optimization_barrier after every operator."""
    vels = [up, vp, wp]
    tend = None
    for a in range(3):
        vel = lax.optimization_barrier(interp_vel(vels[a], a, n))
        rb = lax.optimization_barrier(recon(bp, a, n, vel))
        flux = lax.optimization_barrier(vel * rb)
        term = lax.optimization_barrier(-diff_along(flux, a))
        tend = term if tend is None else tend + term
    return tend


def tendency_handfused(up, vp, wp, bp, recon, n):
    """(b) hand-fused: single expression, no named per-op intermediates.

    Structurally the same graph as (a); differs only in how it is
    spelled (one nested expression).  Included to show XLA fuses (a)
    to the same optimized HLO.
    """
    vels = [up, vp, wp]
    return sum(
        -diff_along(
            interp_vel(vels[a], a, n)
            * recon(bp, a, n, interp_vel(vels[a], a, n)),
            a,
        )
        for a in range(3)
    )


# single-axis weno term (E4: flops vs 3x)
def tendency_weno_axis0(up, vp, wp, bp, n):  # noqa: ARG001
    vel = interp_vel(up, 0, n)
    rb = recon_weno5(bp, 0, n, vel)
    return -diff_along(vel * rb, 0)


# E4 (iv): hoist the sign-split reconstruction behind a barrier so the
# left/right weno branches are each materialized once (forced sharing).
def tendency_weno_hoist(up, vp, wp, bp, n):
    vels = [up, vp, wp]
    tend = None
    for a in range(3):
        vel = interp_vel(vels[a], a, n)
        rb = lax.optimization_barrier(recon_weno5(bp, a, n, vel))
        term = -diff_along(vel * rb, a)
        tend = term if tend is None else tend + term
    return tend
