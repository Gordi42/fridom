"""Numerical/shape validation of all spellings (small sizes, cheap)."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import stencils as S
import e3 as E
from common import check_device

check_device()
rng = np.random.default_rng(0)


def arr(shape):
    return jnp.asarray(rng.standard_normal(shape))


# ---------------- E1 ----------------
print("== E1 equivalence (a=c=e=f, conv order) ==")
N = 16
for axis in (0, 2):
    for k in (2, 3, 5, 7):
        w = S.w_for(k)
        shp = [N, N, N]
        shp[axis] = N + (k - 1)
        fp = arr(tuple(shp))
        a = S.valid_sum(fp, k, axis, w)
        # pad_inside works on size-N input; build the N-input whose wrap-pad
        # reproduces fp? Not generally. Instead test c/e/f on the SAME fp
        # by calling their inner valid parts:
        e = S.stack_tensordot(fp, k, axis, w)
        f = S.tap_loop(fp, k, axis, w)
        d = S.conv_stencil(fp, k, axis, w)
        assert a.shape == e.shape == f.shape == d.shape, (
            a.shape, e.shape, f.shape, d.shape)
        assert jnp.allclose(a, e), ("e", k, axis, float(jnp.abs(a - e).max()))
        assert jnp.allclose(a, f), ("f", k, axis, float(jnp.abs(a - f).max()))
        assert jnp.allclose(a, d), ("d conv", k, axis,
                                    float(jnp.abs(a - d).max()))
    print(f"  axis {axis}: a==e==f==d OK for k in 2,3,5,7")

# roll interior match (periodic vs valid): build size-N f, pad wrap -> fp
for axis in (0, 2):
    for k in (3, 5):
        w = S.w_for(k)
        f = arr((N, N, N))
        rs = S.roll_sum(f, k, axis, w)
        pc = S.pad_inside(f, k, axis, w)
        assert rs.shape == pc.shape == f.shape
        # interior (away from wrap boundary) should match
        core = [slice(None)] * 3
        core[axis] = slice(0, N - (k - 1))
        assert jnp.allclose(rs[tuple(core)], pc[tuple(core)]), (k, axis)
    print(f"  axis {axis}: roll interior == pad_inside OK")

# ---------------- E2 ----------------
print("== E2 equivalence ==")
for axis in (0, 2):
    shp = [N, N, N]
    shp[axis] = N + 2
    fp = arr(tuple(shp))
    c = S.second_deriv_composed(fp, axis)
    cb = S.second_deriv_composed_barrier(fp, axis)
    dd = S.second_deriv_direct(fp, axis)
    assert c.shape == dd.shape == cb.shape
    assert jnp.allclose(c, dd) and jnp.allclose(c, cb), axis
print("  2nd deriv composed==direct==barrier OK")

fp1 = arr((N + 2, N + 2, N + 2))
lc = S.laplacian_composed(fp1)
lf = S.laplacian_fused(fp1)
lb = S.laplacian_composed_barrier(fp1)
assert lc.shape == lf.shape == lb.shape == (N, N, N), lc.shape
assert jnp.allclose(lc, lf) and jnp.allclose(lc, lb)
print("  laplacian composed==fused==barrier OK", lc.shape)

fp2 = arr((N + 4, N + 4, N + 4))
bc = S.biharmonic_composed(fp2)
bd = S.biharmonic_direct(fp2)
bb = S.biharmonic_composed_barrier(fp2)
assert bc.shape == bd.shape == bb.shape == (N, N, N), bc.shape
assert jnp.allclose(bc, bb)
assert jnp.allclose(bc, bd), float(jnp.abs(bc - bd).max())
print("  biharmonic composed==direct==barrier OK", bc.shape)

# ---------------- E3 ----------------
print("== E3 equivalence (composed==handfused==barrier) ==")
n = 12
pad = n + 2 * E.H
up, vp, wp, bp = (arr((pad, pad, pad)) for _ in range(4))
for name, recon in E.RECON.items():
    ta = E.tendency_composed(up, vp, wp, bp, recon, n)
    tb = E.tendency_handfused(up, vp, wp, bp, recon, n)
    tc = E.tendency_barrier(up, vp, wp, bp, recon, n)
    assert ta.shape == tb.shape == tc.shape == (n, n, n), (name, ta.shape)
    assert jnp.allclose(ta, tb), (name, "handfused",
                                  float(jnp.abs(ta - tb).max()))
    assert jnp.allclose(ta, tc), (name, "barrier",
                                  float(jnp.abs(ta - tc).max()))
    print(f"  {name}: composed==handfused==barrier OK")

wax0 = E.tendency_weno_axis0(up, vp, wp, bp, n)
whoist = E.tendency_weno_hoist(up, vp, wp, bp, n)
tw = E.tendency_composed(up, vp, wp, bp, E.recon_weno5, n)
assert wax0.shape == (n, n, n)
assert jnp.allclose(tw, whoist), float(jnp.abs(tw - whoist).max())
print("  weno hoist==composed OK; single-axis shape", wax0.shape)

print("ALL VALIDATION PASSED")
