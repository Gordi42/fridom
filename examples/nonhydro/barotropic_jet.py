r"""
Barotropic Jet
==============

Two opposing zonal jets roll up into a street of vortices.
"""

# %%
# Experiment Settings
# -------------------
# Two zonal jets of opposite sign sit in a triply periodic cube, with
# a weak meridional perturbation on top. The shear between them is
# unstable, so the perturbation grows and folds the jets into
# vortices. We work in nondimensional advective units, where one time
# unit is one eddy turnover. The Rossby number
# :math:`\mathrm{Ro} = U / (f L)` measures the advection against the
# rotation and the Froude number :math:`\mathrm{Fr} = U / (N H)`
# against the stratification. Both are 0.5 here, which puts the
# deformation radius :math:`L_d = (\mathrm{Ro} / \mathrm{Fr}) L` at
# one domain width.
import os
import subprocess

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

ROSSBY_NUMBER = 0.5       # Ro = U / (f L): advection vs. rotation
FROUDE_NUMBER = 0.5       # Fr = U / (N H): advection vs. stratification

WAVE_NUMBER = 3           # zonal wavenumber of the perturbation
WAVE_AMPLITUDE = 0.1      # perturbation amplitude
JET_WIDTH = 0.01          # jet width, relative to the domain

fast = "FRIDOM_EXAMPLES_FAST" in os.environ
nx = ny = 96 if fast else 128
nz = 4                    # the flow is barotropic, so z is cheap
runlen = 1.0 if fast else 2.0
frames = 200

# %%
# Grid and Model
# --------------
# The jets are sharp and they cascade enstrophy to the grid scale, so
# something has to absorb it. Rather than adding a closure we let the
# advection scheme do it: fifth-order WENO reconstruction weights its
# candidate stencils by smoothness, which leaves smooth regions alone
# and damps the oscillations that a centered scheme would build at a
# front. The dissipation is a property of the scheme, not a module,
# so no friction or diffusion is assembled here.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(1.0, 1.0, 1.0),
    periodic=(True, True, True))

# the jets peak at 2.5 in the initial condition, so the advective
# Courant number sets the step
dt = 0.4 * grid.factor("x").dx / 2.5

model = nh.Model(
    grid=grid,
    scaling=fr.scaling.Advective(),
    coriolis=nh.modules.FPlaneCoriolis(rossby_number=ROSSBY_NUMBER),
    buoyancy=nh.modules.ConstantStratification(
        froude_number=FROUDE_NUMBER),
    advection=nh.modules.WENOAdvection(order=5),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# The two jets are narrow Gaussians of opposite sign at one and three
# quarters of the domain, carrying a meridional perturbation of zonal
# wavenumber three. Projecting the sum onto the vortical subspace
# attaches the pressure that balances the jets, so the run starts in
# geostrophic balance and the instability is the only thing that
# grows. The jet width matters: the band of unstable wavelengths
# scales with it, and a jet several times wider than this one leaves
# the seeded wavenumber outside that band, where it only shears over.
model.set_state(nh.initial_conditions.barotropic_jet(
    model,
    wavenum=WAVE_NUMBER,
    waveamp=WAVE_AMPLITUDE,
    jet_width=JET_WIDTH,
    geo_proj=True))

# the vorticity lives on the staggered corner, so it is interpolated
# onto the cell center that the buoyancy already sits on
center = model.state.b.function_space

# plot the initial vorticity in a horizontal slice
_ = model.state.rel_vort_z.to(center).xr.isel(z=0).plot(
    x="x", size=3.2, aspect=1.2)

# %%
# Each jet shows as a pair of vorticity bands, one for each flank,
# already rippled by the seed.
#
# Running and Writing Output
# --------------------------
# Two time units are enough for the ripple to grow, break the jets and
# let the vortices settle. We write the vertical vorticity in a
# horizontal slice once per frame.
writer = fr.io.Writer(
    "barotropic_jet.zarr",
    fields=[],
    derived={"rel_vort_z": lambda ms: ms.state.rel_vort_z.to(center)},
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=(writer,), progress=False)

# %%
# Each jet has rolled up into three vortices, one per wavelength of
# the seed, with filaments of vorticity drawn out between them. A
# zonal transform of the vorticity along the jet keeps its peak at
# wavenumber three throughout, and the harmonics that grow alongside
# it are the sharpened vortex cores rather than a competing mode.
_ = model.state.rel_vort_z.to(center).xr.isel(z=0).plot(
    x="x", size=3.2, aspect=1.2)

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# vorticity animation from the store.
_ = subprocess.run(
    "cdfviewer barotropic_jet.zarr -v rel_vort_z -x x -y y --dims=z=0"
    " -p heatmap -a time"
    " --kwargs='colormap=:balance, colorrange=(-60, 60),"
    " figsize=(700, 640),"
    " titlesize=28, xlabelsize=24, ylabelsize=24,"
    ' animlabel="{value}",'
    " title=\"Barotropic jet\"'"
    " --record -s 'filename=\"barotropic_jet.mp4\", framerate=24'",
    shell=True, check=True)
