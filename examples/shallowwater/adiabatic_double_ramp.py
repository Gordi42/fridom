r"""
Adiabatic Double Ramp
=====================

A balanced flow lives on the *slow manifold* of a rotating model: the
subspace of vortical, geostrophically adjusted motion, as opposed to
the fast inertia-gravity waves. On the equatorial beta-plane no
reference-frame spectral decomposition isolates that subspace, because
the Coriolis parameter varies with latitude and the linear operator
does not diagonalize in closed form. The adiabatic ramping method of
Rosenau et al. sidesteps the difficulty: it *deforms* the model
between two operator configurations slowly enough that a balanced state
follows the deformation without radiating waves.

We run the staggered double ramp of that method. Starting from a
balanced state of an f-plane reference, where the slow manifold is the
familiar geostrophic subspace, we ramp the beta effect on to reach the
equatorial target, ramp the nonlinearity on, let the flow evolve
freely, and then retrace both ramps back to the reference. If the round
trip were perfectly adiabatic the flow would return exactly to the slow
manifold. The residual that does not is the *diabatic leakage*, and we
measure it as the relative imbalance.
"""

# %%
# Experiment Settings
# -------------------
# The domain is a channel that is periodic in the zonal direction
# :math:`x` and walled in the meridional direction :math:`y`, with the
# equator at its center. The Coriolis parameter of the target is
# :math:`f(y) = f_0 + \beta y` with :math:`f_0 = -\beta/2`, so
# :math:`f` is antisymmetric about :math:`y = 1/2` and vanishes at the
# equator. The reference is the same channel with :math:`\beta = 0`, an
# f-plane of uniform rotation :math:`f_0`. A small Rossby number keeps
# the target weakly nonlinear:
import os

import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.shallowwater2 as sw
from fridom.model import params
from fridom.model import term_predicates as terms
from fridom.spatial.meshes import IntervalMesh

csqr = 1.0                   # squared gravity-wave speed
beta = 8.0                   # target beta; reference is beta = 0
f0 = -0.5 * beta             # equator centered in the channel
rossby = 0.2                 # target Rossby number

fast = "FRIDOM_EXAMPLES_FAST" in os.environ
nx = ny = 32 if fast else 48
dt = 0.15 / nx               # inside the AB3 gravity-wave stability limit
tau = 10.0                   # ramp period of the protocol legs
t_free = 2.0                 # free-evolution time between the ramps

# %%
# Reference System and the Slow Projector
# ---------------------------------------
# We assemble the nonlinear target model on the channel and derive the
# f-plane reference as a ``beta = 0`` variant of it. The reference
# operator diagonalizes into geostrophic and inertia-gravity modes, so
# ``sw.eigenbasis`` resolves it and ``VorticalProjection`` projects
# onto the geostrophic slow manifold. That projector defines what
# "balanced" means for the whole protocol:
mesh_x = IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x")
mesh_y = IntervalMesh(ny, (0.0, 1.0), periodic=False, name="y")
grid = fr.spatial.Grid((mesh_x, mesh_y))

model = sw.Model(
    grid=grid, csqr=csqr, rossby_number=rossby,
    coriolis=sw.modules.BetaPlaneCoriolis(f0=f0, beta=beta),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

reference = model.variant(updates={"coriolis.beta": 0.0})
slow = sw.transforms.VorticalProjection(sw.eigenbasis(reference))

# %%
# The initial condition is a random geostrophic state of the reference,
# projected once onto the slow manifold so that its imbalance starts at
# machine zero. Its largest velocity is normalized to one:
z0 = slow(sw.initial_conditions.random_vortical(reference, seed=5))
eta0 = fr.model.transforms.relative_imbalance(z0, slow)
print(f"initial imbalance: {eta0:.1e}")

# %%
# The Staggered Double Ramp
# -------------------------
# Each leg is an :class:`~fridom.model.transforms.AdiabaticRamping`. The
# beta ramp deforms the *linear* system, so we filter it to the linear
# operator; the Rossby ramp turns the nonlinearity on at fixed beta.
# The constructor always describes the up leg (reference to target); the
# ``.down`` accessor gives the endpoint-swapped return leg:
lin_up = fr.model.transforms.AdiabaticRamping(
    model, ramps={"coriolis.beta": (0.0, beta)},
    ramp_period=tau, curve="exp", term_filter=terms.linear)
nl_up = fr.model.transforms.AdiabaticRamping(
    model, ramps={params.SCALING_ROSSBY: (0.0, rossby)},
    ramp_period=tau, curve="exp")
free = fr.model.transforms.Propagator(model, steps=round(t_free / dt))

# %%
# The two ramps are staggered in time: the beta effect switches on
# first, then the nonlinearity, and they switch off in the reverse
# order, so that at every instant the flow sits on the slow manifold of
# the operator it currently experiences. The schedule is the two
# blend weights :math:`\lambda_\beta(t)` and
# :math:`\lambda_{\mathrm{Ro}}(t)`, each rising from zero to one along
# the exponential (Gevrey-2) ramp curve and falling back:
# A leg's blend weight rises from zero along the up-ramp and stays
# there; multiplying by a down-ramp that starts later carves the
# plateau and the fall, giving the staggered trapezoid exactly.
up = fr.model.time_dependent.Ramp(0.0, 1.0, period=tau, curve="exp")
down = fr.model.time_dependent.Ramp(1.0, 0.0, period=tau, curve="exp")
edges = np.array([0.0, tau, 2 * tau, 2 * tau + t_free,
                  3 * tau + t_free, 4 * tau + t_free])
t = np.linspace(0.0, edges[-1], 400)

lam_beta = np.asarray(up(t)) * np.asarray(down(t - (3 * tau + t_free)))
lam_ro = (np.asarray(up(t - tau))
          * np.asarray(down(t - (2 * tau + t_free))))

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.plot(t, lam_beta, label=r"$\lambda_\beta$ (beta effect)")
ax.plot(t, lam_ro, label=r"$\lambda_\mathrm{Ro}$ (nonlinearity)")
for edge in edges[1:-1]:
    ax.axvline(edge, color="0.7", lw=0.8, ls="--")
ax.set_xlabel("time")
ax.set_ylabel("blend weight")
ax.set_title("Staggered double-ramp schedule")
ax.legend(loc="center right", frameon=False)
_ = ax.set_ylim(-0.05, 1.15)

# %%
# The vertical guides mark the phase boundaries: linear up-ramp,
# nonlinear up-ramp, free evolution, nonlinear down-ramp, and linear
# down-ramp. The protocol is the composition of the five legs, applied
# right to left, which we here apply one at a time so we can snapshot
# the flow at each boundary:
#
# Running the Protocol
# --------------------
z1 = lin_up(z0)          # end of the linear up-ramp (on the beta-plane)
z2 = nl_up(z1)           # end of the nonlinear up-ramp (full target)
z3 = free(z2)            # after free evolution
z4 = nl_up.down(z3)      # end of the nonlinear down-ramp
z5 = lin_up.down(z4)     # end of the linear down-ramp (back at reference)


# %%
# We view each stage as the layer-thickness anomaly (the pressure
# field, shaded) with the horizontal velocity streamlines on top. The
# reference and returned states are both smooth, balanced flows, but
# they are not identical: every leg runs forward in time, so the slow
# modes accumulate phase over the protocol and the flow evolves. The
# double ramp preserves the balanced *subspace*, not the individual
# flow, which is why its fidelity is a norm (the imbalance below) rather
# than the eye. The middle panels show the flow deformed onto the
# equatorial target and evolved freely there:
def _panel(ax, state, title, lim):
    """Shade the thickness anomaly and overlay velocity streamlines."""
    center = model.state.p.function_space
    p = state["p"].xr
    u = state["u"].to(center).xr
    v = state["v"].to(center).xr
    x, y = p.coords["x"].values, p.coords["y"].values
    mesh = ax.pcolormesh(x, y, p.values.T, cmap="RdBu_r", vmin=-lim,
                         vmax=lim, shading="auto")
    ax.streamplot(x, y, u.values.T, v.values.T, color="0.2",
                  density=0.7, linewidth=0.6, arrowsize=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


stages = [
    (z0, "reference (balanced)"),
    (z2, "target (deformed)"),
    (z3, "after free evolution"),
    (z5, "returned to reference"),
]
# a shared color scale (set by the reference) lets the panels be
# compared directly: the returned flow carries the same balanced
# thickness amplitude, not the fine-scale checkerboard of fast waves
lim = float(np.abs(z0["p"].xr.values).max())
fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0), constrained_layout=True)
for ax, (state, title) in zip(axes, stages, strict=True):
    mesh = _panel(ax, state, title, lim)
fig.colorbar(mesh, ax=axes, shrink=0.8, label="thickness anomaly")
_ = fig.suptitle("Flow through the double-ramp protocol", fontsize=12)

# %%
# The Relative Imbalance
# ----------------------
# The relative imbalance :math:`\eta(z) = \lVert (I - P) z\rVert /
# \lVert z\rVert` is the fraction of a state that the slow projector
# :math:`P` does not capture. For the returned state it measures the
# diabatic leakage of the whole round trip, against the reference slow
# projector:
eta_round_trip = fr.model.transforms.relative_imbalance(z5, slow)
print(f"round-trip imbalance at tau={tau}: {eta_round_trip:.3e}")

# %%
# At :math:`\tau = 10` the round trip leaks a few tenths of a percent.
# It is a projection-like cycle, though, and such cycles floor at their
# own reversibility residual rather than improving without limit as
# :math:`\tau` grows. To expose the true adiabatic scaling we use a
# single leg: we ramp the linearized channel from the f-plane reference
# to the equatorial target once, and measure how far the transported
# balanced state has left the target slow manifold.
#
# The Exponential Scaling of a Single Leg
# ---------------------------------------
# A scaling study needs long ramps, so we run it on a smaller linear
# channel to keep the cost down. The reference-end balanced state is
# ramped up over a range of ramp periods, with both the smooth
# exponential (Gevrey) curve and a plain linear curve:
sn = 16 if fast else 24
small = sw.Model(
    grid=fr.spatial.Grid((
        IntervalMesh(sn, (0.0, 1.0), periodic=True, name="x"),
        IntervalMesh(sn, (0.0, 1.0), periodic=False, name="y"))),
    csqr=csqr, rossby_number=rossby, advection=False,
    coriolis=sw.modules.BetaPlaneCoriolis(f0=f0, beta=beta),
    time_stepper=fr.model.time_steppers.AdamBashforth(0.15 / sn, order=3))
small_ref = small.variant(updates={"coriolis.beta": 0.0})
slow_ref = sw.transforms.VorticalProjection(sw.eigenbasis(small_ref))
slow_tgt = sw.transforms.VorticalProjection(sw.eigenbasis(small))
z_bal = slow_ref(sw.initial_conditions.random_vortical(small_ref, seed=1))

scaling_taus = np.array([5.0, 10.0, 20.0, 40.0])
leakage = {}
for curve in ("exp", "linear"):
    eta_c = []
    for tau_i in scaling_taus:
        leg = fr.model.transforms.AdiabaticRamping(
            small, ramps={"coriolis.beta": (0.0, beta)},
            ramp_period=float(tau_i), curve=curve,
            term_filter=terms.linear)
        eta_c.append(
            fr.model.transforms.relative_imbalance(leg(z_bal), slow_tgt))
    leakage[curve] = np.array(eta_c)
    print(f"{curve:6s}: " + "  ".join(
        f"tau={ti:.0f}: {e:.1e}"
        for ti, e in zip(scaling_taus, eta_c, strict=True)))

# fit the exponential leg to log(eta) = slope * sqrt(tau) + const
root = np.sqrt(scaling_taus)
slope, const = np.polyfit(root, np.log(leakage["exp"]), 1)

# %%
# Plotted against :math:`\sqrt{\tau}` on a logarithmic axis, the
# exponential-ramp leakage falls on a straight line, the signature of
# Gevrey-class adiabatic decay :math:`\eta \sim \exp(-c\sqrt{\tau})`.
# The linear-ramp leakage is far shallower:
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.semilogy(root, leakage["exp"], "o", label="exp (Gevrey) ramp")
ax.semilogy(root, np.exp(slope * root + const), "-", color="C0",
            label=fr"fit: $\ln\eta = {slope:.2f}\sqrt{{\tau}}{const:+.2f}$")
ax.semilogy(root, leakage["linear"], "s--", color="C1",
            label="linear ramp")
ax.set_xlabel(r"$\sqrt{\tau}$")
ax.set_ylabel(r"single-leg imbalance $\eta$")
_ = ax.legend(frameon=False)

# %%
# The two curves are comparable at the shortest ramp and diverge by
# orders of magnitude as the ramp lengthens: the smooth ramp only pays
# off for ramp periods beyond a few gravity-wave periods
# (:math:`\tau \gtrsim 5` here), and below that the flow is in a
# pre-asymptotic regime where the ramp shape barely matters. A balancing
# study therefore picks a ramp long enough to sit in the exponential
# regime, and reads its convergence from the single-leg diagnostic
# rather than from a projection cycle that would floor at its own
# reversibility residual.
