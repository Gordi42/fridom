r"""
Barotropic Jet
==============

A single zonal jet in a triply periodic box rolls up into vortices.
"""

# %%
# Experiment Settings
# -------------------
# We work in nondimensional *advective* units, so the jet has unit
# width and unit velocity and one time unit is one eddy turnover. Two
# numbers set the regime. The Rossby number
# :math:`\mathrm{Ro} = U / (f L)` measures the advection against the
# rotation and the Froude number :math:`\mathrm{Fr} = U / (N H)`
# against the stratification. Together they fix the deformation radius
# :math:`L_d = (\mathrm{Ro} / \mathrm{Fr})\,H`, here two jet widths.
import subprocess

import jax.numpy as jnp
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

rossby_number = 0.7       # Ro = U / (f L): advection vs. rotation
froude_number = 0.35      # Fr = U / (N H): advection vs. stratification
scaling = fr.scaling.Advective()   # time unit prop to eddy turnover times
domain_size = 10.0        # square domain, ten jet widths on a side
box_height = 1.0          # H in the Froude number, one jet width
seed_amplitude = 1e-2     # weak vortical mode that seeds the instability
seed_wavenumber = 2       # zonal mode number of that seed

nx = ny = 128
nz = 1                    # the flow is 2D, so one layer is enough
runlen = 50.0
frames = 120

# %%
# Grid and Model
# --------------
# The roll-up cascades enstrophy to the grid scale and something has
# to absorb it. Instead of a closure we let the advection scheme do
# it. Fifth-order WENO reconstruction weights its candidate stencils
# by smoothness, so it leaves smooth regions alone and damps the
# oscillations a centered scheme would build at a front. That is the
# only dissipation in this run.
#
# The jet has no vertical structure and nothing in the periodic box
# can give it any, so the flow stays two-dimensional. Rotation then
# drops out of the vorticity budget and the buoyancy stays at zero,
# which leaves the shear as the only source of growth. One cell in
# the vertical is therefore all the run needs.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(domain_size, domain_size, box_height),
    periodic=(True, True, True))

dx = grid.factor("x").dx
# advective Courant number 0.2, fitted so that the run and each of its
# frames are whole numbers of steps
dt = fr.model.fit_dt(runlen, 0.2 * dx, parts=frames)

model = nh.Model(
    grid=grid,
    scaling=scaling,
    coriolis=nh.FPlaneCoriolis(rossby_number=rossby_number),
    buoyancy=nh.ConstantStratification(froude_number=froude_number),
    advection=nh.WENOAdvection(order=5),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# A Gaussian zonal jet in geostrophic balance, seeded with a weak
# vortical mode of zonal wavenumber two.

# a Gaussian zonal jet of unit width, centered in the domain
jet = model.blank_state(
    u=lambda x, y, z: jnp.exp(-((y - 0.5 * domain_size) ** 2)))

# project onto the vortical subspace, which also drops the domain
# mean that would otherwise ring at the inertial frequency
eigenmodes = nh.eigenbasis(model)
project_vortical = nh.transforms.VorticalProjection(eigenmodes)
balanced = project_vortical(jet)
balanced /= balanced.u.max()   # the projection lowers the peak velocity

# seed the instability and set the initial condition
_, perturbation = eigenmodes.mode(
    "vortical", mode_number={"x": seed_wavenumber, "y": 0, "z": 0})
model.set_state(balanced + seed_amplitude * perturbation)

# plot the initial condition in the lowest layer
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
model.state.u.xr.isel(z=0, drop=True).plot(x="x", ax=axs[0])
_ = model.state.v.xr.isel(z=0, drop=True).plot(x="x", ax=axs[1])

# %%
# Dropping the domain mean leaves the jet riding on a weak return
# flow, while :math:`v` carries the wavenumber-two seed alone.
#
# Running and Writing Output
# --------------------------
# We write the vertical vorticity, interpolated to the cell centers,
# once per frame to a zarr store.
center = model.state.b.function_space
writer = fr.io.Writer(
    "barotropic_jet.zarr",
    fields=[],
    derived={"rel_vort_z": lambda ms: ms.state.rel_vort_z},
    space=center,
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=writer)

# %%
# Each flank of the jet has rolled up into two vortices, so the
# saturated state is a wavenumber-two street with filaments of
# vorticity drawn out between the cores.
_ = model.state.rel_vort_z.to(center).xr.isel(z=0, drop=True).plot(
    x="x", size=3.2, aspect=1.2)

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# vorticity animation from the store.
command = (
    "cdfviewer barotropic_jet.zarr"
    " -v rel_vort_z -x x -y y --dims=z=0 -p heatmap -a time"
    # the time axis is nondimensional, so the label drops its unit
    " --kwargs='animlabel=\"t = {rawvalue}\", animlabelnumfmt=\"%.1f\","
    " colormap=:balance, colorrange=(-1.1, 1.1),"
    ' title="Barotropic jet"'
    "' --record -s 'filename=\"barotropic_jet.mp4\", framerate=24'"
)
_ = subprocess.run(command, shell=True, check=True)
