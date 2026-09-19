r"""
Barotropic Instability
======================

A narrow zonal jet rolls up into a street of vortices.
"""

# %%
# Experiment Settings
# -------------------
# We work in nondimensional *advective* units, so the jet has unit
# width and unit velocity and one time unit is one eddy turnover. Two
# numbers set the regime. The Rossby number
# :math:`\mathrm{Ro} = U / (f L)` measures the advection against the
# rotation and the Froude number :math:`\mathrm{Fr} = U / c` against
# the gravity waves. Together they fix the deformation radius
# :math:`L_d = (\mathrm{Ro} / \mathrm{Fr})\,L`, here two jet widths.
import cdfviewer as cv
import jax.numpy as jnp
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.shallowwater2 as sw

rossby_number = 0.7       # Ro = U / (f L): advection vs. rotation
froude_number = 0.35      # Fr = U / c: advection vs. gravity waves
scaling = fr.scaling.Advective()   # time unit prop to eddy turnover times
domain_size = 10.0        # square domain, ten jet widths on a side
seed_amplitude = 1e-2     # weak vortical mode that seeds the instability

nx = ny = 128             # grid cells per side
runlen = 60.0             # eddy turnover times

# %%
# Grid and Model
# --------------
# A doubly periodic square with the shallow-water core, the f-plane
# rotation, and the Sadourny advection. A weak biharmonic friction
# absorbs the enstrophy that the roll-up cascades to the grid scale.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny),
    extent=(domain_size, domain_size),
    periodic=(True, True))

dx = grid.factor("x").dx
# gravity-wave Courant number 0.2, fitted so that the run is a whole
# number of steps
dt = fr.model.fit_dt(runlen, 0.2 * froude_number * dx)
# set dissipation (nu k^4) to advection (Uk) ratio at the grid scale to 0.3.
k_max = jnp.pi / dx                # grid-scale (Nyquist) wavenumber
hyperviscosity = 0.3 / k_max ** 3

model = sw.Model(
    grid=grid,
    core=sw.Core(froude_number=froude_number),
    scaling=scaling,
    coriolis=sw.modules.FPlaneCoriolis(rossby_number=rossby_number),
    advection=sw.SadournyAdvection(),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3),
    modules_extra=fr.model.closures.BiharmonicFriction(nu=hyperviscosity))

# %%
# Initial Condition
# -----------------
# A Gaussian zonal jet in geostrophic balance, seeded with a weak
# vortical mode of zonal wavenumber two.

# a Gaussian zonal jet of unit width, centered in the domain
jet = model.blank_state(
    u=lambda x, y: jnp.exp(-((y - 0.5 * domain_size) ** 2)))

# project onto the vortical subspace to attach the geostrophic pressure
eigenmodes = sw.eigenbasis(model)
project_vortical = sw.transforms.VorticalProjection(eigenmodes)
balanced = project_vortical(jet)
balanced /= balanced.u.max()   # the projection lowers the peak velocity

# seed the instability and set the initial condition
_, perturbation = eigenmodes.mode("vortical", mode_number={"x": 2, "y": 0})
model.set_state(balanced + seed_amplitude * perturbation)

# plot the initial condition
fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
model.state.u.xr.plot(x="x", ax=axs[0])
model.state.v.xr.plot(x="x", ax=axs[1])
_ = model.state.p.xr.plot(x="x", ax=axs[2])

# %%
# The jet sits in geostrophic balance, :math:`u` against its pressure
# :math:`p`, while :math:`v` carries the wavenumber-two seed.
#
# Running and Writing Output
# --------------------------
# We write the pressure and the relative vorticity, interpolated to the
# cell centers, every 0.5 time units to a zarr store.
center = model.state.p.function_space
writer = fr.io.Writer(
    "barotropic_instability.zarr",
    fields="p",
    derived={"rel_vort": lambda ms: ms.state.rel_vort},
    space=center,
    trigger=fr.io.every(time_units=0.5),
    mode="w")

model.run(runlen=runlen, outputs=writer)

# %%
# By the end of the run the jet has broken up into a wavenumber-two
# street of vortices connected by filaments of vorticity.
_ = model.state.rel_vort.xr.plot(x="x")

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# vorticity animation from the store.
_ = cv.record(
    "barotropic_instability.zarr", var="rel_vort", x="x", y="y",
    plot_type="heatmap", ani_dim="time",
    kwargs={"animlabel": "t = {rawvalue}", "animlabelnumfmt": "%.1f",
            "colormap": "balance", "colorrange": (-1.2, 1.2),
            "title": "Barotropic instability"},
    filename="barotropic_instability.mp4", framerate=24)
