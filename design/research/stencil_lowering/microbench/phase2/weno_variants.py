"""Phase-2 WENO-JS divide-reduction spellings (micro-benchmark side).

Mirrors ``e3.py``'s ``_weno_js`` (eps = 1e-6, r = 3) but prices the
divide count of the nonlinear weights:

- ``baseline``   : standard  a_i = d_i/(eps+beta_i)**2 ; sum(a*p)/sum(a)
                   -> 4 f64 divides / reconstruction (r + 1).
- ``singlediv``  : s_i = (eps+beta_i)**2 ; n_i = d_i * prod_{k!=i} s_k ;
                   sum(n*p)/sum(n) -> 1 f64 divide, extra multiplies
                   (exact algebra modulo FP).
- ``f32w``       : nonlinear weights (beta, alpha, normalize) in f32,
                   normalized w_i cast back to f64, applied to f64
                   candidates.  Divides move f64 -> f32.
- ``combined``   : single-divide numerators in f32 (one f32 reciprocal),
                   applied to f64 candidates.

All keep the flux/tendency in f64; only the weight sub-computation of
``f32w``/``combined`` is f32.
"""
from __future__ import annotations

import jax.numpy as jnp

import e3 as E  # parent microbench dir on sys.path

EPS = 1e-6
C13 = 13.0 / 12.0
F32 = jnp.float32
F64 = jnp.float64


# ================================================================
#  Shared candidate / smoothness computation (no divides)
# ================================================================
def _candidates(bm2, bm1, b0, bp1, bp2):
    """The three r=3 Shu candidate reconstructions."""
    p0 = (2.0 * bm2 - 7.0 * bm1 + 11.0 * b0) / 6.0
    p1 = (-bm1 + 5.0 * b0 + 2.0 * bp1) / 6.0
    p2 = (2.0 * b0 + 5.0 * bp1 - bp2) / 6.0
    return p0, p1, p2


def _betas(bm2, bm1, b0, bp1, bp2):
    """The three Jiang-Shu smoothness indicators."""
    beta0 = C13 * (bm2 - 2.0 * bm1 + b0) ** 2 \
        + 0.25 * (bm2 - 4.0 * bm1 + 3.0 * b0) ** 2
    beta1 = C13 * (bm1 - 2.0 * b0 + bp1) ** 2 \
        + 0.25 * (bm1 - bp1) ** 2
    beta2 = C13 * (b0 - 2.0 * bp1 + bp2) ** 2 \
        + 0.25 * (3.0 * b0 - 4.0 * bp1 + bp2) ** 2
    return beta0, beta1, beta2


# ================================================================
#  WENO-JS spellings
# ================================================================
def _weno_js_baseline(b):
    """Standard spelling: r + 1 = 4 f64 divides."""
    bm2, bm1, b0, bp1, bp2 = b
    p0, p1, p2 = _candidates(bm2, bm1, b0, bp1, bp2)
    beta0, beta1, beta2 = _betas(bm2, bm1, b0, bp1, bp2)
    a0 = 0.1 / (EPS + beta0) ** 2
    a1 = 0.6 / (EPS + beta1) ** 2
    a2 = 0.3 / (EPS + beta2) ** 2
    asum = a0 + a1 + a2
    return (a0 * p0 + a1 * p1 + a2 * p2) / asum


def _weno_js_singlediv(b):
    """Single-divide spelling: 1 f64 divide, exact algebra."""
    bm2, bm1, b0, bp1, bp2 = b
    p0, p1, p2 = _candidates(bm2, bm1, b0, bp1, bp2)
    beta0, beta1, beta2 = _betas(bm2, bm1, b0, bp1, bp2)
    s0 = (EPS + beta0) ** 2
    s1 = (EPS + beta1) ** 2
    s2 = (EPS + beta2) ** 2
    n0 = 0.1 * (s1 * s2)
    n1 = 0.6 * (s0 * s2)
    n2 = 0.3 * (s0 * s1)
    return (n0 * p0 + n1 * p1 + n2 * p2) / (n0 + n1 + n2)


def _weno_js_f32w(b):
    """Standard-spelling weights in f32; f64 candidates/application."""
    bm2, bm1, b0, bp1, bp2 = b
    p0, p1, p2 = _candidates(bm2, bm1, b0, bp1, bp2)  # f64
    # weight sub-computation in f32
    cm2, cm1, c0, c1, c2 = (bm2.astype(F32), bm1.astype(F32),
                            b0.astype(F32), bp1.astype(F32),
                            bp2.astype(F32))
    beta0, beta1, beta2 = _betas(cm2, cm1, c0, c1, c2)
    eps = F32(EPS)
    a0 = F32(0.1) / (eps + beta0) ** 2
    a1 = F32(0.6) / (eps + beta1) ** 2
    a2 = F32(0.3) / (eps + beta2) ** 2
    inv = F32(1.0) / (a0 + a1 + a2)
    w0 = (a0 * inv).astype(F64)
    w1 = (a1 * inv).astype(F64)
    w2 = (a2 * inv).astype(F64)
    return w0 * p0 + w1 * p1 + w2 * p2


def _weno_js_combined(b):
    """Single-divide numerators in f32 (one f32 reciprocal)."""
    bm2, bm1, b0, bp1, bp2 = b
    p0, p1, p2 = _candidates(bm2, bm1, b0, bp1, bp2)  # f64
    cm2, cm1, c0, c1, c2 = (bm2.astype(F32), bm1.astype(F32),
                            b0.astype(F32), bp1.astype(F32),
                            bp2.astype(F32))
    beta0, beta1, beta2 = _betas(cm2, cm1, c0, c1, c2)
    eps = F32(EPS)
    s0 = (eps + beta0) ** 2
    s1 = (eps + beta1) ** 2
    s2 = (eps + beta2) ** 2
    n0 = F32(0.1) * (s1 * s2)
    n1 = F32(0.6) * (s0 * s2)
    n2 = F32(0.3) * (s0 * s1)
    inv = F32(1.0) / (n0 + n1 + n2)
    w0 = (n0 * inv).astype(F64)
    w1 = (n1 * inv).astype(F64)
    w2 = (n2 * inv).astype(F64)
    return w0 * p0 + w1 * p1 + w2 * p2


_WENO = {
    "baseline": _weno_js_baseline,
    "singlediv": _weno_js_singlediv,
    "f32w": _weno_js_f32w,
    "combined": _weno_js_combined,
}


def _make_recon(js):
    def recon(bp, a, n, vel):
        wm2 = E.face_win(bp, a, -2, n)
        wm1 = E.face_win(bp, a, -1, n)
        w0 = E.face_win(bp, a, 0, n)
        w1 = E.face_win(bp, a, 1, n)
        w2 = E.face_win(bp, a, 2, n)
        w3 = E.face_win(bp, a, 3, n)
        left = js([wm2, wm1, w0, w1, w2])
        right = js([w3, w2, w1, w0, wm1])
        return jnp.where(vel > 0.0, left, right)
    return recon


RECON = {name: _make_recon(js) for name, js in _WENO.items()}
