r"""nonhydro2 mapped energy-leak n-scaling probe (energy_metric_asymmetry
sec 1.6 cousin leak).

Mirrors the hydrostatic gate `_broadband`/`_phys_skew`
(tests/hydrostatic/test_core_terrain.py): broadband divergence-free
random states X, Y under the model's own Leray projection, evaluate the
linear tendency L, and measure the physical (J-weighted) energy-metric
skew.  The metric is diag(1, 1, 1, 1/N^2) on (u, v, w, b), physical
(J-weighted) via field.integrate().  dsqr = 1 makes the unit-weight-w
metric energy-consistent; coriolis f0 = 0 isolates the buoyancy w<->b
pair (satisfies the linear-model coverage lint, contributes exactly 0).

READ-ONLY probe.  No repo edits.
"""
import sys

import numpy as np
import jax.numpy as jnp

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver

IM = fr.spatial.meshes.IntervalMesh
N2 = 2.0
DSQR = 1.0
FIELDS = ("u", "v", "w", "b")


# ================================================================
#  Grid / model builders (match sec 1.6 + probe_nh.py)
# ================================================================
def _depth(a):
    return lambda x, y: 1.0 + a * jnp.sin(2 * jnp.pi * x) * jnp.cos(
        2 * jnp.pi * y)


def _grid(n, a):
    mapping = CoordinateMapping(maps={"zp": lambda z, H: z * H},
                               params={"H": _depth(a)})
    return fr.spatial.Grid(
        (IM(n, (0.0, 1.0), periodic=True, name="x"),
         IM(n, (0.0, 1.0), periodic=True, name="y"),
         IM(n, (-1.0, 0.0), periodic=False, name="z")), mapping=mapping)


def _model(n, a, *, iters=80, tol=1e-12, family=None):
    grid = _grid(n, a)
    return nh.Model(grid=grid, dsqr=DSQR, dt=1e-3, advection=False,
                    family=family,
                    coriolis=nh.FPlaneCoriolis(f0=0.0),
                    stratification=nh.ConstantStratification(n2=N2),
                    pressure_iterations=iters, pressure_tolerance=tol)


# ================================================================
#  Broadband random field (hydrostatic _broadband pattern)
# ================================================================
def _nodes(grid, space, ax):
    return np.asarray(grid.evaluation_nodes(space.bare, ax).data)


def _broadband(grid, space, seed):
    xs = {a: _nodes(grid, space, a) for a in ("x", "y", "z")}
    rng = np.random.default_rng(seed)
    out = 0.0
    for kx in range(1, 5):
        for kz in range(1, 5):
            amp = rng.standard_normal(2) / (kx * kz)
            out = (out
                   + amp[0] * np.sin(2 * np.pi * kx * xs["x"])
                   * np.cos(kz * np.pi * xs["z"])
                   + amp[1] * np.cos(2 * np.pi * kx * xs["y"])
                   * np.sin(kz * np.pi * (xs["z"] + 1.0)))
    return np.broadcast_to(
        out, np.broadcast_shapes(*(v.shape for v in xs.values())))


# ================================================================
#  Physical-metric machinery
# ================================================================
def _jint(f):
    return float(f.integrate().data.ravel()[0])


def _metric_weight(name, f):
    """b leg carries 1/N^2; u, v, w carry unit weight."""
    return (f / N2) if name == "b" else f


def _pairing(a, db):
    """<a, db>_M: physical (J-weighted) energy inner product."""
    return sum(_jint(_metric_weight(name, a[name]) * db[name])
               for name in FIELDS)


def _norm_M(f):
    return np.sqrt(sum(_jint(_metric_weight(name, f[name]) * f[name])
                       for name in FIELDS))


def _divfree_state(model, seeds):
    """Set broadband ICs, project divergence-free, snapshot + div norm."""
    grid = model.state["u"].grid
    model.set_fields(**{
        name: _broadband(grid, model.state[name].function_space, s)
        for name, s in zip(FIELDS, seeds)})
    pre = model.state
    df = model.constrain(pre)
    solver = MappedPressureSolver(grid, model.state["p"].function_space,
                                  iterations=1)
    d0 = float(jnp.abs(solver.divergence(
        {"x": pre["u"], "y": pre["v"], "z": pre["w"]}).data).max())
    d1 = float(jnp.abs(solver.divergence(
        {"x": df["u"], "y": df["v"], "z": df["w"]}).data).max())
    snap = fr.spatial.fields.vector_field.VectorField(
        {name: df[name].with_data(jnp.asarray(df[name].data))
         for name in FIELDS})
    return snap, d0, d1


def _skews(model, seeds_x, seeds_y):
    """Return dict of bilinear + quadratic skews (raw and projected)."""
    X, d0x, d1x = _divfree_state(model, seeds_x)
    Y, d0y, d1y = _divfree_state(model, seeds_y)
    out = {"div_pre": max(d0x, d0y), "div_post": max(d1x, d1y)}
    for tag, constraints in (("raw", False), ("proj", True)):
        lx = model.tendency(X, constraints=constraints)
        ly = model.tendency(Y, constraints=constraints)
        xy, yx = _pairing(X, ly), _pairing(Y, lx)
        out[f"bilin_{tag}"] = abs(xy + yx) / (abs(xy) + abs(yx))
        # quadratic single-state diagnostic on X (the recorded -6.5e-3)
        qx = _pairing(X, lx)
        out[f"quad_{tag}"] = qx / (_norm_M(X) * _norm_M(lx))
    return out


def _orders(vals):
    vals = np.asarray(vals)
    return np.log2(vals[:-1] / vals[1:])


# ================================================================
#  Runs
# ================================================================
SX = (1, 2, 3, 4)
SY = (5, 6, 7, 8)


def run_flat():
    print("=== FLAT SANITY (a = 0, mapped J=1) — expect ~machine zero ===")
    m = _model(16, 0.0)
    s = _skews(m, SX, SY)
    print(f"  div pre {s['div_pre']:.3e} -> post {s['div_post']:.3e}")
    print(f"  bilinear raw {s['bilin_raw']:.3e}  proj {s['bilin_proj']:.3e}")
    print(f"  quadratic raw {s['quad_raw']:+.3e}  proj "
          f"{s['quad_proj']:+.3e}")
    return s


def run_terrain(a=0.2, ns=(16, 32, 64, 128)):
    print(f"\n=== TERRAIN n-scaling (a = {a}) ===")
    rows = []
    for n in ns:
        m = _model(n, a)
        s = _skews(m, SX, SY)
        rows.append((n, s))
        print(f"  n={n:>3}  div {s['div_pre']:.2e}->{s['div_post']:.2e}  "
              f"bilin_raw {s['bilin_raw']:.3e}  bilin_proj "
              f"{s['bilin_proj']:.3e}  quad_raw {s['quad_raw']:+.3e}  "
              f"quad_proj {s['quad_proj']:+.3e}")
    for key in ("bilin_raw", "bilin_proj", "quad_raw", "quad_proj"):
        vals = [abs(s[key]) for _n, s in rows]
        print(f"  orders[{key}] = "
              + ", ".join(f"{o:.2f}" for o in _orders(vals)))
    return rows


def run_cg_check(a=0.2, n=32):
    print(f"\n=== CG-INDEPENDENCE (a = {a}, n = {n}) ===")
    for iters, tol in ((200, 1e-12), (60, 1e-6), (30, None)):
        m = _model(n, a, iters=iters, tol=tol)
        s = _skews(m, SX, SY)
        print(f"  iters={iters:>3} tol={tol}  div_post {s['div_post']:.2e}"
              f"  bilin_raw {s['bilin_raw']:.4e}  bilin_proj "
              f"{s['bilin_proj']:.4e}  quad_raw {s['quad_raw']:+.4e}")


def run_family(a=0.2, n=16):
    print(f"\n=== nodal vs fv (a = {a}, n = {n}) ===")
    for fam in ("fv", "nodal"):
        m = _model(n, a, family=fam)
        s = _skews(m, SX, SY)
        print(f"  {fam:>5}  bilin_raw {s['bilin_raw']:.6e}  quad_raw "
              f"{s['quad_raw']:+.6e}  quad_proj {s['quad_proj']:+.6e}")


def run_mechanism(a=0.2, n=32):
    """Confirm the leak is the physical-vs-computational measure defect
    in the projection's velocity legs (not the buoyancy coupling)."""
    from fridom.spatial.operators.krylov import _computational_integral
    print(f"\n=== MECHANISM (a = {a}, n = {n}) ===")
    m = _model(n, a)
    X, *_ = _divfree_state(m, SX)
    Y, *_ = _divfree_state(m, SY)

    def cint(f):
        return float(jnp.sum(_computational_integral(f).data))

    # corr = the pressure gradient the projection subtracts: L_raw - L_proj
    # (identically 0 on the b leg; removes the buoyancy force from w).
    def corr(state):
        raw = m.tendency(state, constraints=False)
        proj = m.tendency(state, constraints=True)
        return {nm: raw[nm] - proj[nm] for nm in ("u", "v", "w")}

    cx, cy = corr(X), corr(Y)

    def cross(jw, a_, c):
        """<a, corr>  over the velocity legs u, v, w (raw, unnormalized)."""
        return sum(jw(a_[nm] * c[nm]) for nm in ("u", "v", "w"))

    # <X, corr_Y> : vanishes under the COMPUTATIONAL measure (div_solver
    # X = 0 => <X, grad phi>_comp = <div X, phi>_comp = 0); survives under
    # the PHYSICAL measure -> the leak.
    print("  <X, gradP_Y>  comp {:+.3e}  phys {:+.3e}".format(
        cross(cint, X, cy), cross(_jint, X, cy)))
    print("  <Y, gradP_X>  comp {:+.3e}  phys {:+.3e}".format(
        cross(cint, Y, cx), cross(_jint, Y, cx)))
    leak = cross(_jint, X, cy) + cross(_jint, Y, cx)
    # denominator of the full-operator bilinear skew, for scale
    lpx = m.tendency(X, constraints=True)
    lpy = m.tendency(Y, constraints=True)
    denom = abs(_pairing(X, lpy)) + abs(_pairing(Y, lpx))
    print(f"  physical leak numerator (sum): {leak:+.3e}   "
          f"/ full-op scale {denom:.3e}  = {abs(leak)/denom:.3e}")


def run_ascale(n=32, avals=(0.1, 0.2, 0.4)):
    print(f"\n=== a-scaling at n = {n} ===")
    for a in avals:
        m = _model(n, a)
        s = _skews(m, SX, SY)
        bp, qp = s["bilin_proj"], s["quad_proj"]
        print(f"  a={a}  bilin_proj {bp:.3e}  quad_proj {qp:+.3e}  "
              f"bilin/a {bp/a:.3e}  |quad|/a {abs(qp)/a:.3e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "flat"):
        run_flat()
    if which in ("all", "terrain"):
        run_terrain()
    if which in ("all", "cg"):
        run_cg_check()
    if which in ("all", "family"):
        run_family()
    if which in ("all", "ascale"):
        run_ascale()
