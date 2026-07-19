"""Genuine-chart (J!=1) column-equivalence twin feasibility probe.

Domain A: terrain chart zp = z*H(x) over z in (-1, 0), surface flat at
zp=0, plus an immersed cut at physical height zeta(x) (genuine partial
cells). Wet region [zeta(x), 0].

Domain B: all-wet terrain chart zp = z*D(x), D = -zeta = wet depth, so B
covers exactly the physical wet region of A with nz full cells (no cut).

Both are genuine charts (J = H, D vary in x). Same PHYSICAL IC. Compare
physical J-weighted functionals; measure order under n -> 2n refinement.
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

import fridom.nonhydro2 as nh
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi


# ---- geometry (physical) -------------------------------------------
def H_full(x, y=None):
    """Full chart depth of A: surface 0, bottom -H."""
    return 1.0 + 0.3 * jnp.sin(x)


def zeta(x, y=None):
    """Immersed bottom physical height (negative), well inside (-H,0)."""
    return -0.5 + 0.12 * jnp.sin(x + 0.7)


def D_wet(x, y=None):
    """Wet depth of the sub-chart = -zeta > 0."""
    return -zeta(x)


def make_cut(nz, width_cells):
    """Sharp-ish immersed cut at physical zp=zeta(x).

    The transition band is ``width_cells`` physical A-cells wide
    (physical cell height = H/nz), so a linear GL quadrature is exact
    and the boundary sharpens as nz -> 2nz (targets B's sharp bottom).
    ``width_cells`` small (e.g. 0.2) approaches a geometric step.
    """
    def cut(x, y, z):  # z is PHYSICAL zp on the chart-quadrature path
        if width_cells < 0:                    # FIXED physical width
            width = -width_cells
        else:                                  # width ~ dz (sharpens)
            width = width_cells * H_full(x) / nz
        return jnp.clip((z - zeta(x)) / width + 0.5, 0.0, 1.0)
    return cut


def allwet(x, y, z):  # noqa: ARG001
    return x * 0.0 + y * 0.0 + z * 0.0 + 1.0


# ---- physical initial condition ------------------------------------
def phys_field(name):
    """Smooth physical IC f(x, y, zp) for each prognostic field."""
    def f(x, y, zp):
        base = jnp.cos(x) * jnp.cos(y) * jnp.cos(np.pi * (zp + 0.25))
        if name == "u":
            return 0.2 * base
        if name == "v":
            return 0.15 * jnp.sin(x) * jnp.cos(y) * (zp + 0.5)
        if name == "w":
            return 0.0 * base
        if name == "b":
            return 0.1 * jnp.cos(2 * x) * (zp + 0.3)
        return 0.0 * base
    return f


def phys_modes():
    """Fixed smooth physical test modes phi_k(x, y, zp) for functionals."""
    return {
        "one": lambda x, y, zp: 1.0 + 0.0 * (x + y + zp),
        "cosx": lambda x, y, zp: jnp.cos(x) + 0.0 * (y + zp),
        "z": lambda x, y, zp: zp + 0.0 * (x + y),
        "cosx_z": lambda x, y, zp: jnp.cos(x) * (zp + 0.4) + 0.0 * y,
        "cos2x_z2": lambda x, y, zp: jnp.cos(2 * x) * (zp * zp) + 0.0 * y,
    }


# ---- grids ---------------------------------------------------------
def grid_A(nx, ny, nz, order=4, min_fraction=0.0, width_cells=1.0):
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(nz, (-1.0, 0.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": H_full})
    return Grid((mx, my, mz), mapping=mapping,
                immersed=ImmersedDomain(make_cut(nz, width_cells),
                                        order=order,
                                        min_fraction=min_fraction))


def grid_B(nx, ny, nz):
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(nz, (-1.0, 0.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": D_wet})
    return Grid((mx, my, mz), mapping=mapping)


def model(grid, dt, pressure_iterations=200, pressure_tolerance=1e-12,
          **kw):
    return nh.Model(
        grid=grid, dt=dt, advection=False,
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        dsqr=0.5, pressure_iterations=pressure_iterations,
        pressure_tolerance=pressure_tolerance,
        pressure_preconditioner="spectral", **kw)


def depth_of(grid):
    """Return the depth callable H_full (A) or D_wet (B) by mapping."""
    # both use z*H; distinguish by the param default identity
    return grid.mapping._params["H"]  # noqa: SLF001


def _phys_init(pfun, dfun):
    """Build an (x,y,z)-signature init sampling pfun at physical zp=z*D."""
    def init(x, y, z):
        return pfun(x, y, z * dfun(x))
    return init


def set_physical_ic(m):
    """Set each prognostic field from its physical IC f(x,y,zp)."""
    dfun = depth_of(m.grid)
    fields = {}
    for name in ("u", "v", "w", "b"):
        init = _phys_init(phys_field(name), dfun)
        fields[name] = m.grid.create_field(
            m.state[name].function_space, init=init).data
    m.set_fields(**fields)


def functionals(m):
    """Physical J-weighted inner products <q, phi_k> over the wet region."""
    dfun = depth_of(m.grid)
    imm = m.grid.immersed
    modes = phys_modes()
    out = {}
    for qname in ("u", "v", "w", "b"):
        q = m.state[qname]
        fs = q.function_space
        if imm is not None:
            theta = imm.fraction(fs)
            qw = theta * q
        else:
            qw = q
        for mk, mf in modes.items():
            phi = m.grid.create_field(fs, init=_phys_init(mf, dfun))
            val = float(jnp.sum((qw * phi).integrate().data))
            out[f"{qname}:{mk}"] = val
    return out


def run_case(nx, ny, nz, dt, steps, width_cells=1.0, tol=1e-12,
             iters=200):
    gA = grid_A(nx, ny, nz, width_cells=width_cells)
    gB = grid_B(nx, ny, nz)
    mA = model(gA, dt, pressure_iterations=iters, pressure_tolerance=tol)
    mB = model(gB, dt, pressure_iterations=iters, pressure_tolerance=tol)
    set_physical_ic(mA)
    set_physical_ic(mB)
    f0 = {k: abs(functionals(mA)[k] - functionals(mB)[k])
          for k in functionals(mA)}
    mA.advance(steps)
    mB.advance(steps)
    if mA.panicked or mB.panicked:
        print(f"  PANIC nz={nz}: A={mA.panicked} B={mB.panicked}")
    fA = functionals(mA)
    fB = functionals(mB)
    diffs = {k: abs(fA[k] - fB[k]) for k in fA}
    scale = {k: max(abs(fA[k]), abs(fB[k]), 1e-30) for k in fA}
    reld = {k: diffs[k] / scale[k] for k in fA}
    return fA, fB, diffs, reld, f0


def _ord(v, ns):
    """Convergence orders for values v at resolutions ns (any ratio)."""
    return [(np.log(v[i] / v[i + 1]) / np.log(ns[i + 1] / ns[i]))
            if v[i + 1] > 0 and v[i] > 0 else float("nan")
            for i in range(len(v) - 1)]


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.004
    width_cells = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    resolutions = ([int(x) for x in sys.argv[4].split(",")]
                   if len(sys.argv) > 4 else [8, 16, 32])
    tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-12
    print(f"steps={steps} dt={dt} FULL-refine n=nx=ny=nz min_fraction=0 "
          f"width_cells={width_cells} res={resolutions} tol={tol}")
    records = []
    for n in resolutions:
        fA, fB, diffs, reld, f0 = run_case(
            n, n, n, dt, steps, width_cells, tol=tol)
        records.append((n, diffs, reld, f0))
    print("\nPer-functional |A-B| (t=0 baseline // after run) + order:")
    keys = sorted(records[0][1])
    ns = resolutions
    for k in keys:
        vals = [rec[1][k] for rec in records]
        v0 = [rec[3][k] for rec in records]
        vs = " ".join(f"{v:.2e}" for v in vals)
        os = ",".join(f"{o:.2f}" for o in _ord(vals, ns))
        oo = ",".join(f"{o:.2f}" for o in _ord(v0, ns))
        print(f"  {k:12s} t0[{oo}] // run[{vs}] ord[{os}]")
    print("\nMax dynamic (non-'one') |A-B| and order:")
    md = [max(v for k, v in rec[1].items() if not k.endswith(":one"))
          for rec in records]
    for i, n in enumerate(ns):
        line = f"  n={n:3d} max_dyn={md[i]:.3e}"
        if i > 0:
            line += (f" order="
                     f"{np.log(md[i-1]/md[i])/np.log(ns[i]/ns[i-1]):.2f}")
        print(line)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
