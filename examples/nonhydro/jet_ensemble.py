r"""
Jet Ensemble
============

The same unstable jet from forty-eight random seeds, and the mean over
all of them.
"""

# %%
# Experiment Settings
# -------------------
# We work in nondimensional *advective* units, so the jet has unit
# width and unit velocity and one time unit is one eddy turnover. The
# flow is two-dimensional and has neither rotation nor stratification.
# Every member of the ensemble starts from the same jet plus white
# noise in the meridional velocity, each member drawn from its own
# seed.
import cdfviewer as cv
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xarray as xr

import fridom as fr
import fridom.nonhydro2 as nh

scaling = fr.scaling.Advective()   # time unit prop to eddy turnover times
members = 48              # ensemble size
seed_amplitude = 0.05     # white noise in v, relative to the jet speed

lx, ly = 24, 8            # domain in jet widths, along and across the jet
nx, ny = 96, 32
runlen = 48.0             # eddy turnover times
frames = 48

# %%
# Grid and Model
# --------------
# Without a Coriolis or a buoyancy module the model solves the
# two-dimensional Euler equations, so there are no waves and the
# advection alone limits the time step. One cell in the vertical
# carries the flow, and the WENO reconstruction is the only
# dissipation, as in the :ref:`barotropic jet
# <sphx_glr_auto_examples_nonhydro_barotropic_jet.py>`.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, 1),
    extent=(lx, ly, 1.0),
    periodic=(True, True, True))

dx = grid.factor("x").dx
# advective Courant number 0.3, fitted so that the run and each of its
# frames are whole numbers of steps
dt = fr.model.fit_dt(runlen, 0.3 * dx, parts=frames)

model = nh.Model(
    grid=grid,
    scaling=scaling,
    advection=nh.WENOAdvection(order=5),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Initial Condition
# -----------------
# A Gaussian zonal jet is a steady solution of the two-dimensional
# Euler equations, so left alone it would sit in the box forever. Each
# member adds white noise to the meridional velocity. The noise comes
# from the grid's seeded random field generator, so a member is fixed
# by its seed alone.

# a Gaussian zonal jet of unit width, centered in the domain
jet = model.blank_state(
    u=lambda x, y, z: jnp.exp(-((y - 0.5 * ly) ** 2)))


def perturbed_jet(seed):
    """Return the jet plus white noise in v drawn from ``seed``."""
    noise = grid.random.normal(jet.v.function_space, seed)
    return jet.replace(v=seed_amplitude * noise)


# %%
# Running the Ensemble
# --------------------
# The members run one after another on the same model. Each starts
# from its own seed with the clock reset, and writes the zonal
# velocity and the vertical vorticity, both at the cell centers, to a
# store of its own once per time unit. The first member compiles the
# step and the others reuse it.
center = model.state["p"].function_space
stores = []
for seed in range(members):
    store = f"jet_ensemble_member_{seed:02d}.zarr"
    writer = fr.io.Writer(
        store,
        fields="u",
        derived={"rel_vort_z": lambda ms: ms.state.rel_vort_z},
        space=center,
        trigger=fr.io.every(time_units=runlen / frames),
        mode="w")
    model.reset()
    model.set_state(perturbed_jet(seed))
    model.run(runlen=runlen, outputs=writer)
    stores.append(store)

# %%
# The Ensemble Mean
# -----------------
# xarray stacks the member stores along a new ``member`` dimension,
# so the ensemble mean is one reduction. The mean and the first two
# members go to stores of their own for the animations.
ensemble = xr.concat(
    [xr.open_zarr(store, consolidated=False) for store in stores],
    dim="member").isel(z=0, drop=True)   # the one vertical layer
# the frames sit at whole time units
ensemble = ensemble.assign_coords(time=ensemble.time.round())
mean = ensemble.mean("member", keep_attrs=True)
# CDFViewer reads the zarr v2 format
mean.to_zarr("jet_ensemble_mean.zarr", mode="w", zarr_format=2)
for seed in range(2):
    ensemble.isel(member=seed).to_zarr(
        f"jet_ensemble_seed_{seed}.zarr", mode="w", zarr_format=2)

# three members and the mean at the end of the run
final = ensemble.rel_vort_z.isel(time=-1)
style = {"x": "x", "vmin": -1.5, "vmax": 1.5, "cmap": "RdBu_r",
         "add_colorbar": False}
fig, axs = plt.subplots(2, 2, figsize=(11, 4.4), constrained_layout=True)
for seed, ax in enumerate(axs.flat[:3]):
    final.isel(member=seed).plot(ax=ax, **style)
    ax.set_title(f"seed {seed}")
mesh = mean.rel_vort_z.isel(time=-1).plot(ax=axs[1, 1], **style)
axs[1, 1].set_title("ensemble mean")
fig.colorbar(mesh, ax=axs, label="Vertical relative vorticity [1]")
for ax in axs.flat:
    ax.set_aspect("equal")

# %%
# Every member has rolled up into a street of vortices, at positions
# set by its seed. The mean over the members keeps none
# of them. What remains is a faint band of positive vorticity on the
# northern flank and negative vorticity on the southern flank, the
# signature of a jet that is broader and weaker than the one the run
# started from.
#
# The zonal velocity of the ensemble mean, averaged along the jet,
# shows the same broadening in profile.
profile = mean.u.mean("x")
_ = profile.sel(time=[0, 12, 24, 36, 48]).plot.line(y="y", hue="time")

# %%
# The jet loses half its peak speed and spreads over twice its width,
# the way a viscous jet diffuses. No member is diffusing, however.
# Each one keeps its vortices as sharp as the scheme allows, and the
# spreading appears only in the average over members that disagree
# about where the vortices are.
#
# Rendering the Animations
# ------------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# vorticity of the two members and of the ensemble mean from their
# stores, all on the colour scale of the stills.
for name, title in [("seed_0", "Member, seed 0"),
                    ("seed_1", "Member, seed 1"),
                    ("mean", "Ensemble mean")]:
    _ = cv.record(
        f"jet_ensemble_{name}.zarr", var="rel_vort_z", x="x", y="y",
        plot_type="heatmap", ani_dim="time",
        # the time axis is nondimensional, so the label drops its unit
        kwargs={"animlabel": "t = {rawvalue}", "animlabelnumfmt": "%.1f",
                "colormap": "balance", "colorrange": (-1.5, 1.5),
                "cbarlabel": "auto", "figsize": (1100, 430), "title": title},
        filename=f"jet_ensemble_{name}.mp4", framerate=12)
# sphinx_gallery_video_columns = 1

# %%
# In each member the jet meanders, breaks into a street of vortices,
# and the vortices keep drifting and pairing to the end of the run,
# at positions and at a pace that differ from one member to the next.
# The mean never rolls up. Its vorticity band widens and fades from
# the first frame to the last, and nothing of what the members do on
# their own survives in it.
