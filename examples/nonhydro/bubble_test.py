r"""
Bubble Test
===========

A warm bubble rises through a stratified box, overshoots, and breaks
into a mushroom, filaments and waves.
"""

# %%
# Experiment Settings
# -------------------
# A bubble of warm fluid sits in the middle of a stably stratified
# box. It is lighter than its surroundings by a buoyancy :math:`b_0`
# at its centre, falling off as a Gaussian of radius :math:`R`, and
# it starts at rest. Two numbers size what happens next. The bubble
# rises at about :math:`U = \sqrt{b_0 R}`, a tenth of a metre per
# second here, and it stops being buoyant once it has climbed to where
# the background is as light as it is, a height :math:`b_0/N^2` above
# its start. That height is four radii, so the stratification above
# the bubble is overturned rather than merely displaced. The bubble
# does not settle gently into place; it rolls into a mushroom,
# overshoots, collapses back, and leaves the fluid around it ringing.
# The buoyancy period :math:`2\pi/N` is twenty-one minutes, so the
# hour we run covers about three of them.
#
# The box is periodic in :math:`x`, walled at the top and the bottom,
# and one cell thick in :math:`y`, so the flow is two-dimensional. With
# 256 cells a side a bubble radius spans 25 cells, and the time step
# moves the fluid a quarter of a cell at the rise speed.
import cdfviewer as cv
import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

box_size = 100.0          # metres, square
stratification = 2.5e-5   # N^2 [1/s^2]
bubble_buoyancy = 1e-3    # b_0 [m/s^2] at the centre of the bubble
bubble_radius = 10.0      # R [m]

n = 256                   # cells along x and z
runlen = 3600.0           # seconds, one hour
frames = 240              # number of frames to write

# %%
# Grid and Model
# --------------
# The bubble's own shear rolls it up, and the roll-up cascades
# structure down to the grid scale, where something has to absorb it.
# Instead of a closure we let the advection scheme do it. Fifth-order
# WENO reconstruction weights its candidate stencils by smoothness, so
# it leaves the smooth parts of the flow alone and damps the
# oscillations a centered scheme would build at the sharp edges of the
# mushroom. That is the only dissipation in this run.
dx = box_size / n
grid = fr.spatial.cartesian.Grid(
    shape=(n, 1, n),
    extent=(box_size, dx, box_size),   # one cubic cell thick in y
    periodic=(True, True, False))

rise_speed = (bubble_buoyancy * bubble_radius) ** 0.5
# a quarter of a cell per step at the rise speed, fitted so that the
# run and each of its frames are whole numbers of steps
dt = fr.model.fit_dt(runlen, 0.25 * dx / rise_speed, parts=frames)

model = nh.Model(
    grid=grid,
    buoyancy=nh.ConstantStratification(n2=stratification),
    advection=nh.WENOAdvection(order=5),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# The buoyancy ``b`` is the anomaly against the background
# stratification, so the resting state is zero everywhere and the
# bubble is all there is to set.


def bubble(x, y, z):
    """Return the initial buoyancy, a Gaussian bump at the centre."""
    r2 = (x - 0.5 * box_size) ** 2 + (z - 0.5 * box_size) ** 2
    return bubble_buoyancy * jnp.exp(-r2 / bubble_radius**2)


model.set_fields(b=bubble)

plot = model.state["b"].xr.isel(y=0, drop=True).plot(
    x="x", size=4, vmin=0.0, vmax=bubble_buoyancy)
_ = plot.axes.set_aspect("equal")

# %%
# Running and Writing Output
# --------------------------
# The buoyancy goes to a zarr store once every fifteen seconds of
# model time, 240 frames for the hour.
writer = fr.io.Writer(
    "bubble_test.zarr", fields="b",
    trigger=fr.io.every(seconds=runlen / frames), mode="w")
model.run(runlen=runlen, outputs=writer)

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# animation from the store.
_ = cv.record(
    "bubble_test.zarr", var="b", x="x", y="z", dims={"y": 0},
    plot_type="heatmap", ani_dim="time",
    kwargs={"animlabel": "{duration}", "colormap": "balance",
            "colorrange": (-4e-4, 4e-4),
            "xlabel": "x [m]", "ylabel": "z [m]",
            "title": "Bubble test"},
    filename="bubble_test.mp4", framerate=24)

# %%
# The bubble rolls its own shear into a mushroom within a quarter of
# an hour and overshoots the level where it would be neutrally
# buoyant. The cap breaks up on the way back down, and what spreads
# sideways afterwards is a stack of intrusions trailing filaments a
# few cells wide. The collapse launches internal waves as well, and
# their beams cross the quiet fluid above and below. Every stage of
# that hands finer structure to the grid, and the field at the end of
# the hour carries the whole history of it.
plot = model.state["b"].xr.isel(y=0, drop=True).plot(
    x="x", size=4, cmap="RdBu_r", vmin=-3e-4, vmax=3e-4)
_ = plot.axes.set_aspect("equal")

# %%
# What a model does with that structure once it reaches the grid
# scale is a choice, and the WENO scheme here is one of several.
# :ref:`sphx_glr_auto_examples_nonhydro_advection_and_closures.py`
# runs this same hour eight times, with a different answer each time.
