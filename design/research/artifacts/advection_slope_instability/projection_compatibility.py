"""Is the advection divergence the divergence the projection zeroes?

The adjudication probe (2026-08-14). The A0 fix replaced the nodal
mapped coupled-axis term ``D_i F_i - (Z_i/J) I_b(D_b F_i)`` by the
J-weighted flux form ``(1/J)[D_i(J F_i) - D_b(Z_i I_b(F_i))]``, which
made the nodal mapped tendency coincide with the FV one to round-off.
This asks whether the two spellings were ever interchangeable.

They were not. A tracer's flux divergence must be the SAME discrete
operator the pressure projection drives to zero, or a discretely
non-divergent velocity injects a spurious source into every constant
field. ``tau(b == 1)`` is exactly ``-Div_adv(v)``, so it is directly
comparable with ``MappedPressureSolver.divergence``.

Run it once on the fix and once on ``git show HEAD~1`` of
``src/fridom/model/modules/advection.py``.
"""
from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.model.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver

from clean_repro import make_mapped_model

L = 2 * np.pi
NY = 4


def tendency(model):
    """Return the advection-owned tendency of the model state."""
    return model.tendency(
        model.state, constraints=False,
        filter=fr.model.term_predicates.owned_by(CenteredAdvection))


def projection_divergence(model):
    """Return ``(1/J) Div_proj(v)`` — the operator the projection zeroes."""
    state = model.state
    grid = state["u"].grid
    solver = MappedPressureSolver(
        grid, state["p"].function_space, weights={"z": 1.0},
        iterations=1, params=None)
    div = solver.divergence({"x": state["u"], "y": state["v"],
                             "z": state["w"]})
    inv_j = grid.metric(div.function_space, "dz_dzp", params=None)
    return np.asarray((div * inv_j).data)


def omega_free(n):
    """Smooth analytic (u, w) whose contravariant column flux vanishes."""
    dx = L / n
    xr = (np.arange(n) + 1.0) * dx
    xc = (np.arange(n) + 0.5) * dx
    zc = (np.arange(n) + 0.5) / n
    zi = np.arange(1, n) / n
    ones = np.ones((1, NY, 1))

    def h(x):
        return 1.0 + 0.2 * np.sin(x)

    def hp(x):
        return 0.2 * np.cos(x)

    x, z = np.meshgrid(xr, zc, indexing="ij")
    u = (np.pi * np.cos(np.pi * z) / h(x))[:, None, :] * ones
    x, z = np.meshgrid(xc, zi, indexing="ij")
    w = (np.pi * np.cos(np.pi * z) * z * hp(x) / h(x))[:, None, :] * ones
    return u, w


def operator_gap(n):
    """Row-wise |Div_adv - Div_proj| on a smooth flow (b == 1)."""
    model = make_mapped_model(n, 0.005, ny=NY)
    u, w = omega_free(n)
    model.set_fields(u=u, v=0.0 * u, w=w, b=1.0)
    tau = np.asarray(tendency(model)["b"].data)
    gap = np.abs(projection_divergence(model) + tau).max(axis=(0, 1))
    return gap[0], gap[1:-1].max(), gap[-1]


def constant_source(n):
    """|tau(b == 1)| after one projected step (should be the CG residual)."""
    model = make_mapped_model(n, 0.005, ny=NY)
    rng = np.random.default_rng(5)
    model.set_fields(**{c: rng.standard_normal(model.state[c].data.shape)
                        for c in ("u", "v", "w")})
    model.advance(steps=1)
    model.set_fields(b=1.0)
    return np.abs(np.asarray(tendency(model)["b"].data)).max()


def free_stream(n):
    """Row-wise |tau(b == 1)| for a uniform u (the metric identity)."""
    model = make_mapped_model(n, 0.005, ny=NY)
    model.set_fields(u=0.4, v=0.0, w=0.0, b=1.0)
    rows = np.abs(np.asarray(tendency(model)["b"].data)).max(axis=(0, 1))
    return rows[1:-1].max(), rows[-1]


def main():
    print("1. |Div_adv - Div_proj| row-wise, smooth Omega-free flow")
    for n in (16, 32, 64):
        bottom, interior, top = operator_gap(n)
        print(f"   n={n:3d}  bottom {bottom:.4e}  interior {interior:.4e}"
              f"  top {top:.4e}")
    print("\n2. |tau(b == 1)| on a PROJECTED velocity (spurious source)")
    for n in (16, 32):
        print(f"   n={n:3d}  {constant_source(n):.4e}")
    print("\n3. |tau(b == 1)| for a uniform u = 0.4 (metric identity)")
    for n in (16, 32, 64):
        interior, top = free_stream(n)
        print(f"   n={n:3d}  interior {interior:.4e}  top {top:.4e}")


if __name__ == "__main__":
    sys.exit(main())
