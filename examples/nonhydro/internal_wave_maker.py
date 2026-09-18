r"""
Internal Wave Maker
===================

A localized oscillating force radiates internal gravity waves along
four beams.
"""

# %%
# Experiment Settings
# -------------------
# A rotating stratified x-z slice, one cell thick in y, with rigid
# walls at the top and the bottom. A small oscillating force in the
# lower half of the box drives waves at a period of 45 minutes. The
# forcing frequency lies between :math:`f` and :math:`N`, the band
# in which internal gravity waves exist, so the response radiates
# away from the source.
import cdfviewer as cv
import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.nonhydro2 as nh

CORIOLIS_F0 = 1e-4                  # 1/s
STRATIFICATION_N2 = 2.5e-5          # 1/s^2
LX, LY, LZ = 800.0, 1.0, 200.0      # box extents, m

FORCING_PERIOD = 45.0 * 60.0        # s
# the tendency amplitude is weak enough to keep the waves linear
FORCING_AMPLITUDE = 1e-5            # m/s^2

nx, ny, nz = 512, 1, 128
frames = 180
runlen = 6.0 * 3600.0               # six hours, eight forcing periods

# %%
# Grid and Model
# --------------
# The forcing is weak, so we solve the linearized equations and
# assemble the model without the advection module. The wave maker
# is a source module and joins the preset assembly through
# ``modules_extra``. A source module adds a separable term
#
# .. math::
#     S(x, z, t) = A \cos(2\pi f t + \varphi) \, Q(x, z)
#
# to the tendency of the field it forces, here the zonal velocity.
# The pattern :math:`Q` is a stationary Gaussian mask, sampled on
# the zonal velocity's own nodes. The default phase
# :math:`\varphi = -\pi/2` makes the law a sine, so the forcing
# starts from zero and ramps up gently.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(LX, LY, LZ),
    periodic=(True, True, False))

# omega dt <= 0.1, fitted so that the run and each of its frames
# are whole numbers of steps
dt = fr.model.fit_dt(
    runlen, 0.1 / STRATIFICATION_N2 ** 0.5, parts=frames)

wave_maker = fr.model.modules.Source(
    "wave_maker",
    pattern={"u": nh.gaussian(pos={"x": 400.0, "z": 75.0}, width=4.0)},
    law=fr.Harmonic(
        amplitude=FORCING_AMPLITUDE,
        frequency=1.0 / FORCING_PERIOD))

model = nh.Model(
    grid=grid,
    coriolis=nh.FPlaneCoriolis(f0=CORIOLIS_F0),
    buoyancy=nh.ConstantStratification(n2=STRATIFICATION_N2),
    advection=None,
    modules_extra=wave_maker,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Wave Beams
# ----------
# The dispersion relation of internal gravity waves ties the wave
# frequency to the direction of the wavevector alone. Energy leaves
# an oscillating source along beams whose angle :math:`\alpha`
# above the horizontal satisfies
#
# .. math::
#     \omega^2 = N^2 \sin^2\alpha + f^2 \cos^2\alpha,
#
# independent of the wavelength. The 45 minute period gives
# :math:`\omega / N = 0.47`, so the four beams leave the source at
# about 28 degrees.
#
# Running and Recording
# ---------------------
# Six hours are eight forcing periods, enough for the beams to
# cross the box and reflect off the walls. We write the buoyancy
# and the wave energy once per frame. The energy is the model's
# bound ``etot`` diagnostic, the kinetic energy plus the available
# potential energy :math:`b^2 / (2 N^2)`.
writer = fr.io.Writer(
    "internal_wave_maker.zarr", fields="b",
    derived={"e": lambda ms: model.diagnostics.etot(ms.state)},
    trigger=fr.io.every(seconds=runlen / frames), mode="w")
model.run(runlen=runlen, outputs=writer)

# plot the final buoyancy in the slice plane
_ = model.state.b.xr.isel(y=0).plot(x="x", size=1.6, aspect=4)

# %%
_ = cv.record(
    "internal_wave_maker.zarr", var="b", x="x", y="z", dims={"y": 0},
    plot_type="heatmap", ani_dim="time",
    kwargs={"colormap": "balance", "colorrange": (-7e-6, 7e-6),
            "figsize": (1000, 400),
            "animunit": "hours", "animlabelnumfmt": "%.1f",
            "titlesize": 28, "xlabelsize": 24, "ylabelsize": 24,
            "title": "Internal wave beams"},
    filename="internal_wave_maker.mp4", framerate=24)
_ = cv.record(
    "internal_wave_maker.zarr", var="e", x="x", y="z", dims={"y": 0},
    plot_type="heatmap", ani_dim="time",
    kwargs={"colormap": "thermal", "colorrange": (0.0, 3.0e-7),
            "figsize": (1000, 400),
            "animunit": "hours", "animlabelnumfmt": "%.1f",
            "titlesize": 28, "xlabelsize": 24, "ylabelsize": 24,
            "title": "Wave energy"},
    filename="internal_wave_maker_energy.mp4", framerate=24)
# sphinx_gallery_video_columns = 1

# %%
# The four beams grow out of the source and keep their inclination
# as they cross the box. Where a beam meets the top or the bottom
# wall it reflects at the same angle to the horizontal, because the
# angle is set by the forcing frequency alone. The crossing beams
# interfere into the steady ray pattern of the closing frames. The
# energy travels along the same four beams and shows the ray pattern
# even more cleanly than the buoyancy.

# %%
# Chirped Forcing
# ---------------
# The beam angle follows the forcing frequency, so a forcing that
# sweeps through the wave band draws beams that steepen over time.
# ``Harmonic`` deliberately keeps its frequency constant, because
# the naive :math:`A \sin(2\pi f(t) t)` oscillates at the
# instantaneous frequency :math:`f + t f'` rather than
# :math:`f(t)`. A correct chirp advances the phase by the integral
# of the instantaneous frequency, and we spell that integral
# directly as a ``TimeFunction`` law. The instantaneous period
# falls from 60 to 25 minutes over the run, so the beam angle
# rises from about 20 to almost 60 degrees. We record only the
# wave energy this time.
CHIRP_PERIOD_START = 60.0 * 60.0    # s
CHIRP_PERIOD_END = 25.0 * 60.0      # s

def chirp(t, amp, freq_lo, freq_hi):
    """Advance the phase as the integral of a linear frequency sweep."""
    freq_slope = (freq_hi - freq_lo) / runlen
    phase = 2.0 * jnp.pi * (freq_lo + 0.5 * freq_slope * t) * t
    return amp * jnp.sin(phase)

chirp_maker = fr.model.modules.Source(
    "wave_maker",
    pattern={"u": nh.gaussian(pos={"x": 400.0, "z": 75.0}, width=4.0)},
    law=fr.TimeFunction(
        chirp, params=(FORCING_AMPLITUDE,
                       1.0 / CHIRP_PERIOD_START,
                       1.0 / CHIRP_PERIOD_END)))

chirp_model = nh.Model(
    grid=grid,
    coriolis=nh.FPlaneCoriolis(f0=CORIOLIS_F0),
    buoyancy=nh.ConstantStratification(n2=STRATIFICATION_N2),
    advection=None,
    modules_extra=chirp_maker,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))


chirp_writer = fr.io.Writer(
    "internal_wave_maker_chirp.zarr", fields=[],
    derived={"e": lambda ms: chirp_model.diagnostics.etot(ms.state)},
    trigger=fr.io.every(seconds=runlen / frames), mode="w")
chirp_model.run(runlen=runlen, outputs=chirp_writer)

_ = cv.record(
    "internal_wave_maker_chirp.zarr", var="e", x="x", y="z",
    dims={"y": 0}, plot_type="heatmap", ani_dim="time",
    kwargs={"colormap": "thermal", "colorrange": (0.0, 3.0e-7),
            "figsize": (1000, 400),
            "animunit": "hours", "animlabelnumfmt": "%.1f",
            "titlesize": 28, "xlabelsize": 24, "ylabelsize": 24,
            "title": "Wave energy, chirped forcing"},
    filename="internal_wave_maker_chirp.mp4", framerate=24)

# %%
# Waves keep the frequency they were born with, so beams radiated
# early in the run hold their shallow angle while fresh beams leave
# the source ever steeper. The fan of rays fills from shallow to
# steep as the sweep proceeds.
