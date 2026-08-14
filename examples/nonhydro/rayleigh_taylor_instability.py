r"""
Rayleigh-Taylor Instability
===========================

Dense fluid resting on light fluid, leads to overturning and mixing.
"""

# %%
# Experiment Settings
# -------------------
# Two layers of uniform density sit one on the other, the dense one on
# top. Any dimple in the interface grows, because displacing dense
# fluid downward releases potential energy, so the interface folds into
# the mushrooms that give the instability its picture. A buoyancy jump
# of one across a layer one metre deep gives a free-fall speed of one
# metre per second and a time scale of one second.
#
# There is no rotation and no background stratification. The buoyancy
# is the whole of the physics, and the model carries it as a
# prognostic field that the advection scheme transports.
import subprocess

import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

box_length = 2.0          # metres across
box_depth = 1.0           # metres deep
buoyancy_jump = 1.0       # m/s^2 between the two layers
dimple = 0.002            # interface displacement, as a fraction of depth

nz = 192
nx = 2 * nz               # square cells on a box twice as wide as deep
ny = 1                    # the flow is two-dimensional, in x and z
runlen = 6.0
frames = 240

# %%
# Grid and Model
# --------------
# The box is periodic across and walled top and bottom.
# :class:`~fridom.nonhydro2.modules.buoyancy_tracer.BuoyancyTracer` is
# the buoyancy for a problem like this one. It registers ``b`` and
# contributes the buoyancy force, and nothing else, which is what a
# two-layer setup wants: the restoring term of a background
# stratification is absent from the assembly rather than present and
# multiplied by zero.
#
# The overturning cascades buoyancy to the grid scale and something has
# to absorb it. Instead of a closure we let the advection scheme do it.
# Fifth-order WENO reconstruction weights its candidate stencils by
# smoothness, so it leaves the smooth interior of each layer alone and
# damps the oscillations a centered scheme would build at the
# interface. That is the only dissipation in this run, and on the
# finite-volume family its flux form conserves total buoyancy to
# machine precision.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(box_length, box_depth, box_depth),
    periodic=(True, True, False))

dz = grid.factor("z").dx
# the fastest flow is the free fall of the dense fluid, about one
# metre per second for a unit buoyancy jump across a unit depth
dt = 0.15 * dz / (buoyancy_jump * box_depth) ** 0.5

model = nh.Model(
    grid=grid,
    buoyancy=nh.BuoyancyTracer(),
    advection=nh.WENOAdvection(order=5),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# The interface sits halfway up, dimpled by four long waves so the
# instability has something definite to grow from. A perfectly flat
# interface is an equilibrium, unstable but exact, and would sit there
# until rounding error broke it. The four modes make the run
# reproducible instead.
#
# The interface is smeared over a few cells rather than left as a step.
interface_thickness = 4.0 * dz


def interface(x):
    """Return the height of the interface above the bottom."""
    displacement = sum(
        jnp.cos(2 * jnp.pi * wavenumber * x / box_length + phase)
        for wavenumber, phase in ((3, 0.0), (5, 1.7), (8, 3.9), (13, 0.6)))
    return 0.5 * box_depth + dimple * box_depth * displacement


def two_layers(x, y, z):  # y is named but the flow is two-dimensional
    """Return the buoyancy, high below the interface and low above."""
    return 0.5 * buoyancy_jump * (
        1.0 - jnp.tanh((z - interface(x)) / interface_thickness))


model.set_fields(b=two_layers)

plot = model.state.b.xr.isel(y=0, drop=True).plot(
    x="x", size=2.6, aspect=2.0, cmap="Blues_r", vmin=0.0, vmax=1.0)
plot.axes.set_aspect("equal")

# %%
# Dark is the dense fluid resting on top and light is the buoyant
# fluid underneath.
#
# Running and Writing Output
# --------------------------
# We write the buoyancy once per frame to a zarr store.
writer = fr.io.Writer(
    "rayleigh_taylor.zarr",
    fields=["b"],
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=(writer,))

plot = model.state.b.xr.isel(y=0, drop=True).plot(
    x="x", size=2.6, aspect=2.0, cmap="Blues_r", vmin=0.0, vmax=1.0)
plot.axes.set_aspect("equal")

# %%
# By the end the two layers have traded places and what is left is a
# mixed region rather than an interface.
#
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# buoyancy animation from the store.
_ = subprocess.run(
    "cdfviewer rayleigh_taylor.zarr -v b -x x -y z --dims=y=0"
    " -p heatmap -a time"
    " --kwargs='animlabel=\"t = {rawvalue} s\", animlabelnumfmt=\"%.1f\","
    " colormap=:ice, colorrange=(0, 1),"
    " title=\"Rayleigh-Taylor instability\"'"
    " --record -s 'filename=\"rayleigh_taylor.mp4\", framerate=24'",
    shell=True, check=True)

# %%
# The interface first folds into mushrooms of a single size, set by the
# modes it was given. Those roll up, collide with their neighbours, and
# lose their symmetry, and from there the flow coarsens: small
# structures merge into larger ones and the mixed layer grows from the
# middle outward.
