r"""
Tracers and Eddies
==================

A dye band wound into a spiral by one eddy, then punched through by a
dipole.
"""

# %%
# Experiment Settings
# -------------------
# A passive tracer is carried by the flow and does nothing back to it,
# so it is a way of seeing what the velocity field does rather than a
# part of the dynamics. Both experiments here start from a straight
# band of dye and let an eddy work on it. We use nondimensional
# advective units scaled on the eddy, so one core has unit radius and
# unit peak vorticity. The flow stays two-dimensional and
# non-divergent, so a Coriolis force would be a pure gradient and the
# pressure would absorb it. The model therefore leaves out the
# Coriolis and buoyancy modules, and the dye is the only thing riding
# along with the velocities.
import cdfviewer as cv
import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

eddy_radius = 1.0         # Gaussian radius of one eddy core
eddy_vorticity = 1.0      # peak relative vorticity of one core
band_width = 0.4          # Gaussian half width of the dye band

nx = ny = 128
nz = 1                    # the flow is barotropic, so one layer is enough
frames = 120

# %%
# Model
# -----
# Both experiments want the same model apart from the box size, so we
# build them from one function. Fifth-order WENO reconstruction is the
# only dissipation, and it carries the dye as well as the momentum.
# The tracer is one line: :class:`~fridom.model.modules.Tracer`
# declares a prognostic field with the tracer and advected roles, so
# the assembled advection scheme transports it with no further wiring.
# Each model needs its own tracer instance, since a module binds to
# exactly one model.
#
# The two experiments share one plotting function as well. It draws
# the dye with contours of relative vorticity over it, so the eddies
# that move the dye appear in the same frame. Negative contour levels
# come out dashed, which tells the two senses of rotation apart.
#
# A third helper lists what each run writes to disk. The animations
# colour the dye, and one of them draws arrows from the two horizontal
# velocities, so all three fields go into the store on the cell centre
# the dye already sits on.


def build(domain_size, runlen):
    """Return a model on a periodic box, timed for a run of runlen."""
    grid = fr.spatial.cartesian.Grid(
        shape=(nx, ny, nz),
        extent=(domain_size, domain_size, domain_size),
        periodic=(True, True, True))
    dx = grid.factor("x").dx
    # the fastest flow is the swirl in a core, about 0.6 of the peak
    # vorticity times the core radius. The step is fitted so that the
    # run and each of its frames are whole numbers of steps
    dt = fr.model.fit_dt(
        runlen, 0.3 * dx / (0.6 * eddy_vorticity * eddy_radius),
        parts=frames)
    return nh.Model(
        grid=grid,
        scaling=fr.scaling.Advective(),
        advection=nh.WENOAdvection(order=5),
        modules_extra=fr.model.modules.Tracer(
            "dye", long_name="Dye concentration", units="1"),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


def dye_band(centre):
    """Return a Gaussian dye band centred on a height in the box."""
    def band(x, y, z):  # x and z are named but the band varies only in y
        return jnp.exp(-((y - centre) / band_width) ** 2)
    return band


# three rings per core, as fractions of the peak vorticity
vorticity_levels = [fraction * eddy_vorticity
                    for fraction in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8)]


def plot_dye(model, space):
    """Plot the dye of a model with the vorticity drawn over it."""
    dye = model.state["dye"].to(space).xr.isel(z=0, drop=True)
    vorticity = model.state.rel_vort_z.to(space).xr.isel(z=0, drop=True)
    # the dye range is pinned to the one the animations use. WENO
    # leaves a little undershoot below zero, and xarray answers signed
    # data with a range symmetric about zero
    plot = dye.plot(x="x", size=3.4, aspect=1.2, cmap="Blues",
                    vmin=0, vmax=1)
    vorticity.plot.contour(
        x="x", ax=plot.axes, levels=vorticity_levels, colors="crimson",
        linewidths=0.8, add_colorbar=False)
    plot.axes.set_aspect("equal")


# %%
# The Spiral
# ----------
# One eddy sits at the middle of a box five radii across, with the dye
# band laid straight through it. The eddy turns faster at its centre
# than at its edge, so the band is wound rather than merely rotated,
# and every turn of the spiral is a thinner filament than the one
# before it.
spiral_box = 5.0 * eddy_radius
spiral_runlen = 30.0

spiral = build(spiral_box, spiral_runlen)
spiral.set_state(nh.coherent_eddy(
    spiral, pos_x=0.5, pos_y=0.5,
    width=eddy_radius / spiral_box, amplitude=eddy_vorticity))
spiral.set_fields(dye=dye_band(0.5 * spiral_box))

# the dye sits on the cell center the pressure already occupies
center = spiral.state["p"].function_space
plot_dye(spiral, center)

# %%
# We write the dye and the two velocities, all on the cell centers,
# once per frame to a zarr store and run.
writer = fr.io.Writer(
    "tracers_spiral.zarr",
    fields=["dye", "u", "v"],
    space=center,
    trigger=fr.io.every(time_units=spiral_runlen / frames),
    mode="w")

spiral.run(runlen=spiral_runlen, outputs=writer)

plot_dye(spiral, center)

# %%
# The band has been drawn into a spiral of several turns. The
# filaments thin as they wind, and once one is as narrow as a grid
# cell the advection scheme can no longer hold it, which is why the
# outer turns fade rather than continuing indefinitely.
#
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# animation from the store. ``cbarlabel="auto"`` labels the bar from
# the tracer's own metadata, the ``long_name`` given to
# :class:`~fridom.model.modules.Tracer` above. ``figsize`` gives the
# window the shape of the box, so the heatmap fills the frame and the
# colorbar sits next to it instead of away from it.
#
# There are no arrows over this one. The filaments are the subject,
# and they are narrower than the arrows that would cross them. The
# flow those arrows would report is the rotation the spiral already
# draws.

_ = cv.record(
    "tracers_spiral.zarr", var="dye", x="x", y="y", dims={"z": 0},
    plot_type="heatmap", ani_dim="time",
    kwargs={"animlabel": "t = {rawvalue}", "animlabelnumfmt": "%.0f",
            "colormap": "dense", "colorrange": (0, 1), "figsize": (800, 756),
            "cbarlabel": "auto", "title": "A tracer wound by one eddy"},
    filename="tracers_spiral.mp4", framerate=24)

# %%
# The Collision
# -------------
# The second experiment gives the dye something to be hit by. A dipole
# starts low in a box ten radii across and travels north into a band
# laid across its path. A dipole carries a parcel of fluid with it, so
# rather than sliding past the band it pushes a bulge ahead of itself,
# closes that bulge into a bubble, and drags the bubble along while the
# rest of the band is pulled out into two trailing filaments.
collision_box = 10.0 * eddy_radius
collision_runlen = 55.0

collision = build(collision_box, collision_runlen)
collision.set_state(nh.eddy_dipole(
    collision, pos_x=0.5, pos_y=0.15, angle=0.0,
    width=eddy_radius / collision_box, amplitude=eddy_vorticity))
collision.set_fields(dye=dye_band(0.55 * collision_box))

center = collision.state["p"].function_space
writer = fr.io.Writer(
    "tracers_collision.zarr",
    fields=["dye", "u", "v"],
    space=center,
    trigger=fr.io.every(time_units=collision_runlen / frames),
    mode="w")

collision.run(runlen=collision_runlen, outputs=writer)

plot_dye(collision, center)

# %%
# The rings are the two cores of the dipole. The bubble and its two
# trailing filaments are what a dipole leaves behind after crossing a
# tracer front.
#
# Here the flow is worth drawing, because the parcel the dipole
# carries is not something the dye reports on its own. ``--over u,v``
# adds the velocity as a second layer and ``--over-plot quiver`` draws
# it as arrows. ``minspeed`` drops every arrow slower than 0.15, which
# is about a quarter of the swirl speed in a core. That empties the
# quiet far field and leaves the carried parcel and the flow that
# returns around it. ``arrows`` sets how many are drawn across each
# axis. The arrows fall inside the bubble, where there is no dye, so
# the band that bounds it stays readable.

_ = cv.record(
    "tracers_collision.zarr", var="dye", x="x", y="y", dims={"z": 0},
    plot_type="heatmap", ani_dim="time",
    over=["u,v"], over_plot=["quiver"],
    kwargs={"animlabel": "t = {rawvalue}", "animlabelnumfmt": "%.0f",
            "colormap": "dense", "colorrange": (0, 1), "figsize": (800, 756),
            "over.color": "black", "arrows": (22, 22), "minspeed": 0.15,
            "cbarlabel": "auto", "title": "A dipole crossing a tracer band"},
    filename="tracers_collision.mp4", framerate=24)
