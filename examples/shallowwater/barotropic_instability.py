r"""
Barotropic Instability
======================

A narrow zonal jet is barotropically unstable: small perturbations
feed on the horizontal shear and grow until the jet rolls up into a
street of vortices. We integrate this classic experiment with the
shallow-water model on a doubly periodic f-plane, write the run to a
zarr store, and render the vorticity animation from that store with
CDFViewer.
"""
# sphinx_gallery_thumbnail_number = 2

# %%
# Experiment Settings
# -------------------
# Two nondimensional numbers characterize the jet. The Rossby number
# :math:`\mathrm{Ro} = U / (f_0 L_\mathrm{jet})` measures the strength
# of the nonlinear advection relative to the rotation, and the Burger
# number :math:`\mathrm{Bu} = c^2 / (f_0 L)^2` relates the deformation
# radius to the domain size. We choose the jet velocity
# :math:`U = f_0 L_\mathrm{jet}`, so the Rossby number is one, and a
# small Burger number, so the deformation radius is one tenth of the
# domain:
import os
import subprocess

import fridom as fr
import fridom.shallowwater2 as sw

f0 = 1.0                     # Coriolis parameter
L = 1.0                      # square domain of size L x L
jet_width = L / 20
u_jet = f0 * jet_width       # Rossby number Ro = 1
csqr = (0.1 * f0 * L) ** 2   # Burger number Bu = 1/100

# %%
# Documentation builds on pull requests smoke-run every example at
# reduced cost; the ``FRIDOM_EXAMPLES_FAST`` environment variable
# selects a coarser grid and a shorter run:
fast = "FRIDOM_EXAMPLES_FAST" in os.environ
nx = 96 if fast else 192     # grid points per direction
runlen = 30.0 if fast else 120.0

# %%
# Grid and Model
# --------------
# The domain is a doubly periodic square, built from one
# ``IntervalMesh`` per axis. The ``sw.Model`` preset assembles the
# shallow-water dynamical core, the rotation, and the Sadourny
# advection scheme on that grid. Moreover, we add a weak biharmonic
# friction that dissipates the enstrophy the roll-up cascades to the
# grid scale:
mesh_x = fr.spatial.meshes.IntervalMesh(nx, (0.0, L),
                                        periodic=True, name="x")
mesh_y = fr.spatial.meshes.IntervalMesh(nx, (0.0, L),
                                        periodic=True, name="y")
grid = fr.spatial.Grid((mesh_x, mesh_y))

model = sw.Model(
    grid=grid,
    csqr=csqr,
    rossby_number=1.0,
    coriolis=sw.modules.FPlaneCoriolis(f0=f0),
    time_stepper=fr.model.time_steppers.AdamBashforth(2.0 / nx, order=3),
    modules_extra=(fr.model.closures.BiharmonicFriction(
        nu=0.01 * u_jet * (L / nx) ** 3),))

# %%
# Initial Condition
# -----------------
# ``sw.initial_conditions.jet`` samples a Gaussian zonal jet, projects
# it onto the geostrophic subspace, and adds a small single-mode
# perturbation of zonal wavenumber two that seeds the instability. The
# factory normalizes the largest velocity to one; therefore we scale
# the state to the jet velocity before assigning it:
z = sw.initial_conditions.jet(
    model, width=jet_width / L, wavenum=2, waveamp=1e-2)
model.set_state(u_jet * z)

model.state.u.xr.plot(x="x_right")

# %%
# The plot shows the unperturbed picture: a narrow band of eastward
# velocity centered at :math:`y = L/2` (``u`` lives on the staggered
# x faces of the C-grid, hence its coordinate name ``x_right``). The
# perturbation is far too weak to be visible at this stage.
#
# Running and Writing Output
# --------------------------
# A ``Writer`` streams selected fields to a zarr store while the model
# runs; the store opens in xarray with no post-processing. We store
# the pressure and, as a derived output evaluated at write time, the
# relative vorticity; the trigger fires once per model time unit:
writer = fr.model.io.Writer(
    "barotropic_instability.zarr",
    fields=["p"],
    derived={"rel_vort": lambda ms: ms.state.rel_vort},
    trigger=fr.model.io.every(seconds=1.0),
    mode="w")

model.run(runlen=runlen, outputs=(writer,), progress=False)

# %%
# By the end of the run the instability has saturated: the jet has
# broken up into a wavenumber-two street of coherent vortices,
# connected by filaments of vorticity:
model.state.rel_vort.xr.plot(x="x_right")

# %%
# Rendering the Animation
# -----------------------
# For a quick look at the store, or a video of it, there is
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_, which plots
# NetCDF files and zarr stores straight from the command line. The
# same one-liner that explores the store interactively records the
# vorticity animation when we pass ``--record``:
command = (
    "cdfviewer barotropic_instability.zarr"
    " -v rel_vort -x x_right -y y_right -p heatmap -a time"
    " --kwargs='colormap=:balance, colorrange=(-1.2, 1.2)'"
    " --record -s 'filename=\"barotropic_instability.mp4\", framerate=24'"
)
_ = subprocess.run(command, shell=True, check=True)

# %%
# The animation shows the full life cycle: the shear instability grows
# out of an imperceptible perturbation, overturns, and settles into a
# street of coherent eddies.
