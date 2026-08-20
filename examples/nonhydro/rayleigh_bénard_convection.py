r"""
Rayleigh-Bénard Convection
==========================

Heat a box from below, cool it from above.
"""

# %%
# Experiment Settings
# -------------------
# A layer of fluid held warm at the bottom and cool at the top cannot
# stay still. Conduction alone would give a straight buoyancy profile
# with the dense fluid on top, and that arrangement is unstable, so the
# layer breaks into plumes that carry warm fluid up and cool fluid
# down.
import subprocess

import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.nonhydro2 as nh

box_length = 2.0          # metres across
box_depth = 1.0           # metres deep
contrast = 1.0            # m/s^2 held at each plate, warm below, cool above
plate_depth = 0.02        # metres, the layer the plates act on
plate_time = 0.5          # seconds, how fast a plate imposes its value

nz = 128
nx = 2 * nz               # square cells on a box twice as wide as deep
ny = 1                    # the flow is two-dimensional, in x and z
runlen = 12.0
frames = 480

# %%
# Grid and Model
# --------------
# The box is periodic across and walled top and bottom.
# :class:`~fridom.nonhydro2.modules.buoyancy_tracer.BuoyancyTracer`
# registers the buoyancy and contributes its force.
#
# Fifth-order WENO reconstruction is the only dissipation. That is
# worth being plain about: with no explicit viscosity or diffusivity
# there is no Rayleigh number to quote, since the smallest scales are
# set by the grid rather than by a physical parameter.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(box_length, box_depth, box_depth),
    periodic=(True, True, False))

dz = grid.factor("z").dx
# the fastest flow is the free fall of a plume across the layer, about
# one metre per second for this contrast and depth
free_fall = (2 * contrast * box_depth) ** 0.5
# a quarter of a cell per step at that speed, fitted so that the run
# and each of its frames are whole numbers of steps
dt = fr.model.fit_dt(runlen, 0.25 * dz / free_fall, parts=frames)

# %%
# The Plates
# ----------
# The two plates are a relaxation term rather than a boundary
# condition. Inside a thin layer next to each wall the buoyancy is
# nudged toward the value that plate holds, on a timescale short
# compared with the overturning but long compared with the time step.
# Everywhere else the mask is zero and the term does nothing.


def plate_value(z):
    """Return the buoyancy each plate holds, warm below and cool above."""
    return jnp.where(z < 0.5 * box_depth, contrast, -contrast)


def plate_mask(z):
    """Return one inside either plate layer and zero between them."""
    return jnp.where(
        (z < plate_depth) | (z > box_depth - plate_depth), 1.0, 0.0)


model = nh.Model(
    grid=grid,
    buoyancy=nh.BuoyancyTracer(),
    advection=nh.WENOAdvection(order=5),
    modules_extra=fr.model.modules.Relaxation(
        "b", rate=1.0 / plate_time,
        target=plate_value, mask=plate_mask),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# The fluid starts at rest and at uniform buoyancy, so the plates have
# to build the unstable profile themselves. A layer that is exactly
# uniform in the horizontal would stay that way, since nothing would
# pick out where the first plume should rise, so we seed it with four
# long waves of a thousandth of the plate contrast. The four modes
# make the run reproducible.


def seed(x, y, z):  # y is named but the flow is two-dimensional
    """Return a faint horizontal ripple, strongest at mid depth."""
    ripple = sum(
        jnp.cos(2 * jnp.pi * wavenumber * x / box_length + phase)
        for wavenumber, phase in ((3, 0.0), (5, 1.7), (8, 3.9), (13, 0.6)))
    return 1e-3 * contrast * ripple * jnp.exp(
        -((z - 0.5 * box_depth) / (0.25 * box_depth)) ** 2)


model.set_fields(b=seed)

# %%
# Running and Writing Output
# --------------------------
# We write the buoyancy once per frame to a zarr store.
writer = fr.io.Writer(
    "rayleigh_benard.zarr",
    fields="b",
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=writer)

plot = model.state.b.xr.isel(y=0, drop=True).plot(
    x="x", size=2.6, aspect=2.0, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
plot.axes.set_aspect("equal")

# %%
# Red is warm fluid and blue is cool. By the end the plumes reach right
# across the layer and the interior is well mixed, with the sharp
# gradients confined to thin skins against the two plates.
#
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# buoyancy animation from the store.
_ = subprocess.run(
    "cdfviewer rayleigh_benard.zarr -v b -x x -y z --dims=y=0"
    " -p heatmap -a time"
    " --kwargs='animlabel=\"t = {rawvalue} s\", animlabelnumfmt=\"%.1f\","
    " colormap=:balance, colorrange=(-1, 1),"
    " title=\"Rayleigh-Benard convection\"'"
    " --record -s 'filename=\"rayleigh_benard.mp4\", framerate=24'",
    shell=True, check=True)

# %%
# The plates build their thin unstable layers first, and for a while
# nothing else happens. Those layers then let go in a burst of small
# plumes along both walls, which merge as they cross the interior into
# the few large overturning cells that carry most of the transport.
