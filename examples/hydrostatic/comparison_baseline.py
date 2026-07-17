r"""
Geostrophic Adjustment (Comparison Baseline)
============================================

The hydrostatic *comparison preset* (:func:`hy.comparison_model`) is the
matched-numerics configuration in which FRIDOM is meant to agree, in the
shared numerical limit, with pyOM, Veros, and Oceananigans: quasi-AB2
time stepping, a backward-Euler implicit linear free surface, centered
second-order flux-form advection, an f-plane, and a linear buoyancy. Here
we drive it with the classic **geostrophic (Rossby) adjustment** problem
— a released surface-pressure ridge that radiates inertia-gravity waves
and settles into a balanced jet — and check the outcome against the
analytic adjusted state.
"""

# %%
# Experiment Settings
# -------------------
# One nondimensional number sets the scene: the Rossby radius of
# deformation :math:`L_d = \sqrt{c^2}/f_0`, the horizontal scale at which
# rotation balances the surface-pressure gradient. We take
# :math:`c^2 = 1` and :math:`f_0 = 4`, so :math:`L_d = 1/4` — a quarter of
# the doubly-periodic domain. The initial pressure ridge is narrower than
# :math:`L_d`, so adjustment is vigorous: most of the released energy
# leaves as waves and only a fraction is retained in the balanced jet.
import os

import matplotlib.pyplot as plt
import numpy as np

import fridom as fr
import fridom.hydrostatic as hy
from fridom.spatial.meshes import IntervalMesh

# sphinx_gallery_thumbnail_number = 2
f0 = 4.0                       # f-plane Coriolis parameter
csqr = 1.0                     # squared barotropic phase speed c^2 = g H
ld = np.sqrt(csqr) / f0        # deformation radius = 1/4 of the domain
ridge_width = 0.6 * ld         # the ridge is narrower than L_d

fast = "FRIDOM_EXAMPLES_FAST" in os.environ
nx = ny = 48 if fast else 96
nz = 4
# the implicit free surface is unconditionally stable in the (fast)
# gravity waves, so dt is set by the explicit Coriolis (f0*dt << 1) and
# is deliberately independent of the resolution: the backward-Euler wave
# damping is per-step, so a resolution-independent dt keeps the number of
# steps that settle the balance the same at every nx.
dt = 0.008 / np.sqrt(csqr)     # f0 * dt = 0.032
n_frames, steps_per_frame = 40, 8

# %%
# Grid and Model
# --------------
# The domain is a doubly-periodic square with a bounded (flat-bottom)
# vertical — the ``(P, P, bounded-z)`` box the preset targets. We switch
# the stratification off (``n2=0``) so the problem is purely barotropic:
# the buoyancy decouples and the adjustment plays out in the
# surface-pressure and horizontal-velocity fields alone, which is exactly
# where the analytic Rossby-adjustment solution lives.
grid = fr.spatial.Grid((
    IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x"),
    IntervalMesh(ny, (0.0, 1.0), periodic=True, name="y"),
    IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z")))

model = hy.comparison_model(grid, dt, csqr=csqr, coriolis_f0=f0, n2=0.0)

# %%
# Initial Condition
# -----------------
# A surface-pressure ridge, Gaussian in :math:`x` and uniform in
# :math:`y`, released from rest (:math:`u = v = 0`). It is not in
# geostrophic balance, so the model must adjust: an inertia-gravity wave
# front runs off to either side, and a balanced meridional jet is left
# straddling the ridge.
amplitude = 1.0e-2


def ridge(x, y):
    """Return a Gaussian surface-pressure ridge, uniform in y."""
    xc = x - 0.5
    return amplitude * np.exp(-(xc ** 2) / (2.0 * ridge_width ** 2)) \
        + 0.0 * y


ps0 = model.grid.create_field(
    model.state["ps"].function_space, init=ridge, name="ps")
model.set_state(model.state.replace(ps=ps0))

xg = np.linspace(0.0, 1.0, nx, endpoint=False) + 0.5 / nx
ps_initial = np.asarray(model.state["ps"].data)[:, 0, 0]

fig, ax = plt.subplots()
ax.plot(xg, ps_initial, color="tab:blue")
ax.set(xlabel="x", ylabel="surface pressure  $p_s$",
       title="Initial pressure ridge (released from rest)")

# %%
# Running and Tracking the Energy
# -------------------------------
# The implicit free surface is unconditionally stable and *damps* the
# gravity waves it cannot resolve in time, so the radiated energy leaves
# the box while the zero-frequency geostrophic mode is untouched. We
# advance in short chunks and record the total (kinetic + available
# potential) energy, which falls and levels off at the retained,
# geostrophic value.


def total_energy(m):
    """Discrete total energy 0.5 int (u^2 + v^2 + ps^2/c^2)."""
    s = m.state
    u = np.asarray(s["u"].data)
    v = np.asarray(s["v"].data)
    ps = np.asarray(s["ps"].data)
    return 0.5 * ((u ** 2).sum() + (v ** 2).sum()
                  + nz * (ps ** 2).sum() / csqr)


e0 = total_energy(model)
times = [0.0]
energy = [1.0]
for frame in range(n_frames):
    model.advance(steps_per_frame)
    times.append((frame + 1) * steps_per_frame * dt)
    energy.append(total_energy(model) / e0)

# %%
# The Adjusted State
# ------------------
# By the end of the run the waves have radiated away and the flow is in
# geostrophic balance: the pressure ridge has broadened and weakened, and
# a pair of counter-flowing meridional jets — :math:`f v = \partial_x p_s`
# — straddles it. The jets are the geostrophic signature the balanced
# state retains.
ps_final = np.asarray(model.state["ps"].data)[:, 0, 0]
# the meridional velocity lives at the x-cell centres already (uniform in
# y for this ridge), so a surface slice is the jet profile v(x)
v_cell = np.asarray(model.state["v"].data)[:, 0, 0]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.0, 5.0))
ax1.plot(xg, ps_initial, "--", color="0.6", label="initial")
ax1.plot(xg, ps_final, color="tab:blue", label="adjusted")
ax1.set(ylabel="surface pressure  $p_s$")
ax1.legend()
ax2.plot(xg, v_cell, color="tab:red")
ax2.set(xlabel="x", ylabel="meridional velocity  $v$",
        title="Geostrophic jet")
fig.tight_layout()

# %%
# Checking Against the Analytic Adjusted State
# --------------------------------------------
# For a single Fourier mode the retained energy fraction has a closed
# form. Discrete potential-vorticity inversion — with the
# energy-conserving C-grid Coriolis interpolation factor
# :math:`\gamma = \prod_a \cos(k_a \Delta x_a / 2)` — gives, per mode,
#
# .. math::
#
#     \frac{E_\mathrm{retained}}{E_\mathrm{initial}}
#       = \frac{\gamma^2}{\gamma^2 + k_d^2 L_d^2},
#
# with :math:`k_d` the discrete wavenumber and :math:`L_d^2 = c^2/f^2`.
# Summing this over the ridge's spectrum predicts the fraction of energy
# the balanced state keeps; the measured plateau settles onto it within a
# couple of percent (the small excess is the last, slowly-damped
# long-wave energy still in flight). The precise, machine-accuracy
# version of this check lives in the physics-validation suite
# (``tests/hydrostatic/test_comparison.py``).
ps_hat = np.fft.rfft(ps_initial)
power = np.abs(ps_hat) ** 2
kd2 = (2.0 * np.sin(np.pi * np.arange(power.size) / nx) * nx) ** 2
gamma2 = np.cos(np.pi * np.arange(power.size) / nx) ** 2
retained = gamma2 / (gamma2 + kd2 * (csqr / f0 ** 2))
predicted = float((power * retained).sum() / power.sum())

fig, ax = plt.subplots()
ax.plot(times, energy, color="tab:blue", label="model")
ax.axhline(predicted, color="tab:green", ls="--",
           label=f"analytic retained fraction = {predicted:.3f}")
ax.set(xlabel="time", ylabel="total energy  $E / E_0$",
       title="Energy released to (damped) gravity waves",
       ylim=(0.0, 1.05))
ax.legend()

print(f"deformation radius L_d = {ld:.3f}")
print(f"measured retained energy fraction = {energy[-1]:.4f}")
print(f"analytic retained energy fraction = {predicted:.4f}")
