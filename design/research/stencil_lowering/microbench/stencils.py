"""Stencil spellings for E1-E4.  Pure jax/lax.

Conventions:
- valid-mode spellings take a pre-padded array `fp` of size N+(k-1) along
  the stencil axis and return size N.
- shape-preserving spellings take size-N input and return size N.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from common import sl

# distinct, non-zero, non-unity weights so no tap folds away
E1_W = [0.3, 0.7, 1.3, 1.9, 2.3, 2.9, 3.1]


def w_for(k):
    return E1_W[:k]


# ================================================================
#  E1 spellings
# ================================================================


def valid_sum(fp, k, axis, w):
    """(a) fridom's valid-mode slice-window weighted sum."""
    n = fp.shape[axis] - (k - 1)
    out = w[0] * sl(fp, 0, n, axis)
    for j in range(1, k):
        out = out + w[j] * sl(fp, j, j + n, axis)
    return out


def roll_sum(f, k, axis, w):
    """(b) periodic roll sum (shape preserving)."""
    out = w[0] * f
    for j in range(1, k):
        out = out + w[j] * jnp.roll(f, -j, axis=axis)
    return out


def pad_inside(f, k, axis, w):
    """(c) pad(mode='wrap') inside jit, then valid windows."""
    padw = [(0, 0)] * f.ndim
    padw[axis] = (0, k - 1)
    fp = jnp.pad(f, padw, mode="wrap")
    return valid_sum(fp, k, axis, w)


def conv_stencil(fp, k, axis, w):
    """(d) lax.conv_general_dilated with a degenerate 3-D kernel.

    Input fp is pre-padded (size N+k-1 along axis); VALID conv -> N.
    """
    lhs = fp[None, None]  # (1,1, d0,d1,d2)
    kshape = [1, 1, 1, 1, 1]
    kshape[2 + axis] = k
    kernel = jnp.zeros(kshape, dtype=fp.dtype)
    for j in range(k):
        idx = [0, 0, 0, 0, 0]
        idx[2 + axis] = j
        kernel = kernel.at[tuple(idx)].set(w[j])
    out = lax.conv_general_dilated(
        lhs, kernel, window_strides=(1, 1, 1), padding="VALID",
    )
    return out[0, 0]


def stack_tensordot(fp, k, axis, w):
    """(e) stack k shifted windows then tensordot (materialize strawman)."""
    n = fp.shape[axis] - (k - 1)
    windows = [sl(fp, j, j + n, axis) for j in range(k)]
    stacked = jnp.stack(windows, axis=0)  # (k, ...)
    wv = jnp.asarray(w, dtype=fp.dtype)
    return jnp.tensordot(wv, stacked, axes=([0], [0]))


def tap_loop(fp, k, axis, w):
    """(f) fori_loop over taps with dynamic_slice (no unroll)."""
    n = fp.shape[axis] - (k - 1)
    wv = jnp.asarray(w, dtype=fp.dtype)
    size = list(fp.shape)
    size[axis] = n
    out_shape = tuple(size)

    def body(j, acc):
        start = [0] * fp.ndim
        start[axis] = j
        window = lax.dynamic_slice(fp, start, size)
        return acc + wv[j] * window

    return lax.fori_loop(0, k, body, jnp.zeros(out_shape, dtype=fp.dtype))


# ---- shape-preserving wrappers for the per-iter loop probe ----


def valid_preserve(f, k, axis, w):
    """pad+valid windows, size N -> N (loop probe form of a/c/e)."""
    return pad_inside(f, k, axis, w)


def conv_preserve(f, k, axis, w):
    padw = [(0, 0)] * f.ndim
    padw[axis] = (0, k - 1)
    fp = jnp.pad(f, padw, mode="wrap")
    return conv_stencil(fp, k, axis, w)


def stack_preserve(f, k, axis, w):
    padw = [(0, 0)] * f.ndim
    padw[axis] = (0, k - 1)
    fp = jnp.pad(f, padw, mode="wrap")
    return stack_tensordot(fp, k, axis, w)


def taploop_preserve(f, k, axis, w):
    padw = [(0, 0)] * f.ndim
    padw[axis] = (0, k - 1)
    fp = jnp.pad(f, padw, mode="wrap")
    return tap_loop(fp, k, axis, w)


# ================================================================
#  E2: derivatives (staggered two-point diff building block)
# ================================================================
#  Two-point staggered difference on a padded array (valid mode).
#  d1[i] = f[i+1] - f[i]  (Center->Face), size N given input N+1.


def diff1(fp, axis):
    n = fp.shape[axis] - 1
    return sl(fp, 1, 1 + n, axis) - sl(fp, 0, n, axis)


def second_deriv_composed(fp, axis):
    """Apply diff1 twice: input size N+2 -> N.  Intermediate size N+1."""
    d = diff1(fp, axis)          # size N+1
    return diff1(d, axis)        # size N


def second_deriv_composed_barrier(fp, axis):
    d = diff1(fp, axis)
    d = lax.optimization_barrier(d)
    return diff1(d, axis)


def second_deriv_direct(fp, axis):
    """Direct 3-point: f[i+2]-2f[i+1]+f[i], input N+2 -> N."""
    n = fp.shape[axis] - 2
    return (sl(fp, 2, 2 + n, axis)
            - 2.0 * sl(fp, 1, 1 + n, axis)
            + sl(fp, 0, n, axis))


def laplacian_composed(fp):
    """3-D Laplacian via per-axis composed second derivatives, sum.

    fp is padded by 1 on both sides of every axis (size N+2 per axis).
    Output size N per axis.  Trim the non-differentiated axes to N.
    """
    out = None
    for axis in range(3):
        term = second_deriv_composed(fp, axis)  # N along axis, N+2 others
        # trim the other two axes from N+2 to N (drop 1 each side)
        for other in range(3):
            if other != axis:
                term = sl(term, 1, term.shape[other] - 1, other)
        out = term if out is None else out + term
    return out


def laplacian_composed_barrier(fp):
    out = None
    for axis in range(3):
        d = diff1(fp, axis)
        d = lax.optimization_barrier(d)
        term = diff1(d, axis)
        for other in range(3):
            if other != axis:
                term = sl(term, 1, term.shape[other] - 1, other)
        out = term if out is None else out + term
    return out


def laplacian_fused(fp):
    """Single fused 7-point Laplacian expression.  fp padded by 1 each side.

    Output size N per axis; interior slice [1:N+1] per axis is the center.
    """
    n0, n1, n2 = (s - 2 for s in fp.shape)
    c = fp[1:1 + n0, 1:1 + n1, 1:1 + n2]
    xm = fp[0:n0, 1:1 + n1, 1:1 + n2]
    xp = fp[2:2 + n0, 1:1 + n1, 1:1 + n2]
    ym = fp[1:1 + n0, 0:n1, 1:1 + n2]
    yp = fp[1:1 + n0, 2:2 + n1, 1:1 + n2]
    zm = fp[1:1 + n0, 1:1 + n1, 0:n2]
    zp = fp[1:1 + n0, 1:1 + n1, 2:2 + n2]
    return xm + xp + ym + yp + zm + zp - 6.0 * c


def biharmonic_composed(fp):
    """Laplacian of Laplacian.  fp padded by 2 each side (size N+4/axis).

    Inner laplacian_fused on the N+2 interior -> size N+2 per axis, then
    outer laplacian_fused -> size N.
    """
    lap1 = laplacian_fused(fp)  # size N+2 per axis (fp is N+4)
    return laplacian_fused(lap1)  # size N


def biharmonic_composed_barrier(fp):
    lap1 = laplacian_fused(fp)
    lap1 = lax.optimization_barrier(lap1)
    return laplacian_fused(lap1)


def biharmonic_direct(fp):
    """Direct 3-D biharmonic 13-point (per-axis 4th deriv + cross terms).

    Biharmonic = d4/dx4 + d4/dy4 + d4/dz4 + 2 d2/dx2 d2/dy2 + ... .
    fp padded by 2 each side (size N+4).  Build directly from slices.
    """
    n0, n1, n2 = (s - 4 for s in fp.shape)

    def win(o0, o1, o2):
        return fp[o0:o0 + n0, o1:o1 + n1, o2:o2 + n2]

    c = win(2, 2, 2)
    # pure fourth derivatives: [1,-4,6,-4,1]
    d4 = None
    for ax in range(3):
        offs = [[2, 2, 2] for _ in range(5)]
        for t in range(5):
            offs[t][ax] = t
        term = (win(*offs[0]) - 4.0 * win(*offs[1]) + 6.0 * win(*offs[2])
                - 4.0 * win(*offs[3]) + win(*offs[4]))
        d4 = term if d4 is None else d4 + term
    # cross terms 2 * d2x d2y etc: d2 along a = [1,-2,1]
    cross = None
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        acc = 0.0
        for ca, wa in ((1, 1.0), (2, -2.0), (3, 1.0)):
            for cb, wb in ((1, 1.0), (2, -2.0), (3, 1.0)):
                o = [2, 2, 2]
                o[a] = ca
                o[b] = cb
                acc = acc + wa * wb * win(*o)
        cross = acc if cross is None else cross + acc
    return d4 + 2.0 * cross + 0.0 * c


# shape-preserving loop forms for E2 (re-pad inside)
def _pad_sym(f, width):
    padw = [(width, width)] * f.ndim
    return jnp.pad(f, padw, mode="wrap")


def lap_preserve(f):
    return laplacian_fused(_pad_sym(f, 1))


def lap_composed_preserve(f):
    return laplacian_composed(_pad_sym(f, 1))


def biharm_preserve(f):
    return biharmonic_composed(_pad_sym(f, 2))


def biharm_direct_preserve(f):
    return biharmonic_direct(_pad_sym(f, 2))
