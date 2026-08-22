r"""
Geostrophic Adjustment
======================

A bump on the surface of a rotating channel collapses into an eddy,
and its waves run along the walls.
"""

# %%
# Experiment Settings
# -------------------
# A layer of water forty metres deep fills a channel on an f-plane,
# periodic along x and walled on its two sides. Its surface is raised
# into a Gaussian bump a metre high, and the water is at rest. Nothing
# holds the bump up, so it starts to collapse, and the Coriolis force
# turns the collapsing flow until an anticyclone balances what is
# left of the bump. The rest leaves as a ring of inertia-gravity
# waves, and the ring meets the walls.
#
# Barotropic gravity waves travel at
# :math:`c = \sqrt{gH}`, about 20 m/s here, and the deformation radius
# :math:`L_d = c/f`, about 200 km. The bump is one and a half deformation radii
# in radius.
import subprocess

import jax.numpy as jnp
import numpy as np

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.hydrostatic as hy

gravity = 9.81            # m/s^2
depth = 40.0              # metres
f0 = 1e-4                 # Coriolis parameter [1/s]
channel_length = 6000e3   # metres, periodic
channel_width = 2000e3    # metres, between the walls
bump_height = 1.0         # metres
bump_radius = 300e3       # 1/e radius, one and a half deformation radii

wave_speed = np.sqrt(gravity * depth)
nx, ny = 192, 64
runlen = 72.0 * 3600.0    # seconds
frames = 180

# %%
# Grid and Model
# --------------
# Nothing varies with depth, so one cell in z carries the whole flow.
# The surface waves are the physics of this experiment, so the free
# surface is stepped explicitly and the time step resolves them. We run the
# model without advection.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, 1),
    extent=(channel_length, channel_width, depth),
    periodic=(True, False, False))

dx = grid.factor("x").dx
# the explicit surface waves hold the step to c dt/dx <= 0.2 on the
# two-dimensional grid, and parts=frames fits it to a whole number of
# steps per frame
dt = fr.model.fit_dt(runlen, 0.2 * dx / wave_speed, parts=frames)

model = hy.Model(
    grid=grid,
    core=hy.Core(gravity=gravity),
    free_surface=hy.ExplicitFreeSurface(),
    coriolis=hy.FPlaneCoriolis(f0=f0),
    buoyancy=None,            # constant density: a barotropic flow
    advection=False,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# The model carries the surface as the pressure :math:`p_s = g\eta` it
# exerts on the water below. The bump goes in through that field, in
# the middle of the channel, and everything else starts at rest. For
# the plots and the store the surface is read back as the elevation
# :math:`\eta`, which the model exposes as the ``eta`` diagnostic.
bump_center = (0.5 * channel_length, 0.5 * channel_width)


def bump(x, y):
    """Return the surface pressure of a Gaussian bump."""
    r2 = (x - bump_center[0]) ** 2 + (y - bump_center[1]) ** 2
    return gravity * bump_height * jnp.exp(-r2 / bump_radius**2)


model.set_fields(ps=bump)

# the bump, drawn to scale
_ = model.diagnostics.eta().xr.plot(x="x", size=2.6, aspect=3.0)

# %%
# Running and Writing Output
# --------------------------
# Every frame writes the surface elevation and the two horizontal
# velocities to a zarr store, all moved to the cell centre so one set
# of coordinates carries them.
center = model.state["p_hyd"].function_space
writer = fr.io.Writer(
    "geostrophic_adjustment.zarr",
    fields=["u", "v"],
    derived={"eta": lambda ms: model.diagnostics.eta(ms.state)},
    space=center,
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=writer)

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# animation from the store, the velocities as black arrows over the
# elevation.
_ = subprocess.run(
    "cdfviewer geostrophic_adjustment.zarr -v eta -x x -y y --dims=z=0"
    " -p heatmap -a time"
    " --over u,v --over-plot quiver"
    " --kwargs='animlabel=\"t = {rawvalue} h\", animunit=\"hours\","
    ' animlabelnumfmt="%.1f",'
    " colormap=:balance, colorrange=(-0.6, 0.6),"
    " color=:black, arrows=(75, 25),"
    ' figsize=(1200, 460), xunit="km", yunit="km",'
    " title=\"Geostrophic adjustment\"'"
    " --record -s 'filename=\"geostrophic_adjustment.mp4\", framerate=24'",
    shell=True, check=True)

# %%
# The bump collapses within a few hours but does not go flat. What stays is
# a lower, broader hump with the flow circling it clockwise, an
# anticyclone in balance with the pressure gradient. The rest leaves
# as a ring of inertia-gravity waves.
#
# A wall turns part of what reaches it into Kelvin waves: waves that
# stay within a deformation radius of the wall and run along it at the
# gravity-wave speed, with the wall on their right because :math:`f`
# is positive, their flow parallel to the wall and in balance with the
# slope across it. Along the southern wall they run east, along the
# northern wall west, and since the channel is periodic they keep
# going round, one circuit of the channel taking three and a half
# days. The rest of the ring reflects back and forth between the walls
# as Poincaré waves, and the channel slowly fills with them while the
# eddy in the middle stays.

# %%
# The same surface in three dimensions, the camera drifting slowly
# along the channel: the bump drops to half its height and keeps it,
# and the waves that carry off the rest are a far gentler undulation.
_ = subprocess.run(
    "cdfviewer geostrophic_adjustment.zarr -v eta -x x -y y --dims=z=0"
    " -p surface -a time"
    " --kwargs='animlabel=\"t = {rawvalue} h\", animunit=\"hours\","
    ' animlabelnumfmt="%.1f",'
    " colormap=:balance, colorrange=(-0.6, 0.6),"
    " limits=(0, 6000e3, 0, 2000e3, -0.3, 1.1), aspect=(3, 1, 0.6),"
    " azimuth=4.05, elevation=0.45, perspectiveness=0.4, rotate=2.4,"
    " viewmode=:fitzoom, protrusions=(50, 0, 30, 0),"
    " ylabeloffset=60, zlabeloffset=60,"
    ' figsize=(1200, 470), xunit="km", yunit="km", zlabel="η [m]",'
    " title=\"Geostrophic adjustment\"'"
    " --record -s 'filename=\"geostrophic_adjustment_surface.mp4\","
    " framerate=24'",
    shell=True, check=True)
