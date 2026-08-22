r"""
Dancing Eddies
==============

Two eddy dipoles collide, swap partners and work the walls, over and
over.
"""

# %%
# Experiment Settings
# -------------------
# A pair of counter-rotating eddies moves. Each one is carried by the
# flow of the other, so the pair travels along the line between them,
# while a single eddy would only sit and spin. This example puts two
# such dipoles in a box, aimed at each other. We work in
# nondimensional advective units scaled on the eddy instead of the
# box, so one core has unit radius and unit peak vorticity, which
# puts the fastest flow in the box near half a unit. The flow stays
# two-dimensional and non-divergent, so a Coriolis force would be a
# pure gradient and the pressure would absorb it. The model therefore
# leaves out the Coriolis and buoyancy modules. The box is periodic in
# y and walled in x, and that combination is what makes the sequence
# come back around.
import cdfviewer as cv

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.nonhydro2 as nh

eddy_radius = 1.0         # Gaussian radius of one eddy core
eddy_vorticity = 1.0      # peak relative vorticity of one core
domain_size = 12.0 * eddy_radius   # square box, walls in x

nx = ny = 192
nz = 1                    # the flow is barotropic, so one layer is enough
runlen = 117.4            # one full cycle, so the animation loops
frames = 120

# %%
# Grid and Model
# --------------
# Nothing varies with depth here, so a single cell in z carries the
# whole flow. The collisions cascade vorticity to the grid scale and
# something has to absorb it. Instead of a closure we let the
# advection scheme do it. Fifth-order WENO reconstruction weights its
# candidate stencils by smoothness, so it leaves the smooth eddy cores
# alone and damps the oscillations a centered scheme would build where
# two dipoles meet. That is the only dissipation in this run.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(domain_size, domain_size, domain_size),
    periodic=(False, True, True))

dx = grid.factor("x").dx
# the fastest flow in the box is the jet between the two cores of a
# dipole, which runs at about 0.6 of the peak vorticity times the
# core radius
dt = fr.model.fit_dt(
    runlen, 0.3 * dx / (0.6 * eddy_vorticity * eddy_radius), parts=frames)

model = nh.Model(
    grid=grid,
    scaling=fr.scaling.Advective(),
    advection=nh.WENOAdvection(order=5),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# :func:`~fridom.nonhydro2.initial_conditions.eddy_dipole` builds a
# dipole from where it starts, how strong its cores are and which way
# it should point. Position and width are given as fractions of the
# box, so the radius is divided by the box size. The angle is a
# compass bearing, so 90 sends the western pair east and 270 sends
# the eastern pair west. Each dipole is a pair of Gaussian
# vorticity blobs whose streamfunction the factory recovers by
# inverting the Laplacian, which on a walled axis is exact, and the
# velocities are the discrete curl of that streamfunction. Adding the
# two states superposes them, and that is exact rather than
# approximate because the whole construction is linear in the
# streamfunction.
dipoles = (
    nh.eddy_dipole(model, pos_x=0.25, pos_y=0.3, angle=90.0,
                   width=eddy_radius / domain_size,
                   amplitude=eddy_vorticity)
    + nh.eddy_dipole(model, pos_x=0.75, pos_y=0.3, angle=270.0,
                     width=eddy_radius / domain_size,
                     amplitude=eddy_vorticity))
model.set_state(dipoles)

# the vorticity lives on the cell corner, so it is moved to the cell
# center the pressure already sits on before plotting
center = model.state["p"].function_space
plot = model.state.rel_vort_z.to(center).xr.isel(z=0, drop=True).plot(
    x="x", size=3.4, aspect=1.3)
_ = plot.axes.set_aspect("equal")

# %%
# Red and blue mark the two rotation senses. Each dipole is one red
# eddy beside one blue one, and the pair moves along the line between
# them, so the two dipoles head toward each other. Their travel speed
# is an outcome of the mutual induction rather than a setting, and it
# scales as the peak vorticity times the core radius.
#
# Running and Writing Output
# --------------------------
# The run covers one full turn of the sequence, so it ends on the
# arrangement it started from and the animation loops. Every frame
# writes the vertical vorticity and the two horizontal velocities to a
# zarr store. The animation draws its arrows from the velocities. All
# three are moved to the cell center the pressure occupies, so one set
# of coordinates carries them.
writer = fr.io.Writer(
    "dancing_eddies.zarr",
    fields=["u", "v"],
    derived={"rel_vort_z": lambda ms: ms.state.rel_vort_z},
    space=center,
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=writer)

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# animation from the store. The velocities go over the vorticity as a
# second layer of arrows.

_ = cv.record(
    "dancing_eddies.zarr", var="rel_vort_z", x="x", y="y", dims={"z": 0},
    plot_type="heatmap", ani_dim="time",
    over=["u,v"], over_plot=["quiver"],
    kwargs={"animlabel": "t = {rawvalue}", "animlabelnumfmt": "%.0f",
            "colormap": "balance", "colorrange": (-1.4, 1.4),
            "color": "black", "arrows": (24, 24),
            "figsize": (800, 750),
            "cbarlabel": "auto", "title": "Dancing eddies"},
    filename="dancing_eddies.mp4", framerate=24)

# %%
# The two dipoles meet in the middle and each eddy leaves with the
# partner it did not arrive with, so the new pairs travel north and
# south instead of east and west. The box is periodic in y, so those
# pairs meet again and swap back, and the pairs that come out of that
# second exchange head for the walls. A dipole cannot cross a wall, so
# it splits, one eddy running along the wall in each direction, until
# each meets its opposite number and pairs up again in the arrangement
# the run started from. That return is close rather than exact, since a
# pair of Gaussian dipoles is not a periodic solution, and the cores
# land about a tenth of a radius from their starting points.
