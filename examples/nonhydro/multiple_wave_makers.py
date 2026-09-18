r"""
Multiple Wave Makers
====================

Two oscillating sources radiate internal-wave beams at two
different angles.
"""

# %%
# Experiment Settings
# -------------------
# A rotating stratified x-z slice, one cell thick in y, periodic on
# every side. Two small oscillating sources sit at different
# positions along x and force the flow, each from a carrier
# wavevector pointing in a different direction. The carrier
# direction sets the wave frequency, and both frequencies lie
# between :math:`f` and :math:`N`, the band in which internal
# gravity waves exist, so each source radiates.
import cdfviewer as cv
import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.nonhydro2 as nh

CORIOLIS_F0 = 1e-4                  # 1/s
STRATIFICATION_N2 = 2.5e-5          # 1/s^2
LX, LY, LZ = 250.0, 1.0, 200.0      # box extents, m

WAVE_LENGTH = 10.0                  # carrier wavelength, m
WAVE_ANGLES = (60.0, 30.0)          # carrier angle(kx, kz), degrees
# both beams travel down and to the right, so the sources sit high
# and to the left and the run ends before either one meets a wall
MAKER_X = (50.0, 140.0)             # source positions along x, m
MAKER_Z = 140.0                     # source height, m
ENVELOPE_WIDTH = 25.0               # packet envelope width, m
# the tendency amplitude keeps the isopycnal displacement near a
# tenth of a wavelength, so the linearization stays self-consistent
FORCING_AMPLITUDE = 3.1e-7          # m/s^2

nx, ny, nz = 512, 1, 320
frames = 288
runlen = 6.0 * 3600.0               # six hours, before either beam walls

# %%
# Grid and Model
# --------------
# The sources are weak, so we solve the linearized equations and
# assemble the model without the advection module. Each source
# carries a polarized wave packet drawn from the model eigenmodes,
# so we assemble a plain model first, build the packets from it, and
# then assemble the running model with the three sources attached
# through ``modules_extra``. A source multiplies its packet by a
# sine in time and adds the product to the tendency, so the forcing
# starts from zero and ramps up gently.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(LX, LY, LZ),
    periodic=(True, True, True))

# omega dt <= 0.1, fitted so that the run and each of its frames
# are whole numbers of steps
dt = fr.model.fit_dt(
    runlen, 0.1 / STRATIFICATION_N2 ** 0.5, parts=frames)

base = nh.Model(
    grid=grid,
    coriolis=nh.FPlaneCoriolis(f0=CORIOLIS_F0),
    buoyancy=nh.ConstantStratification(n2=STRATIFICATION_N2),
    advection=None,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

makers = []
for i, (angle, x) in enumerate(zip(WAVE_ANGLES, MAKER_X, strict=True)):
    # the carrier direction fixes the wavevector components
    kx = 2.0 * jnp.pi / WAVE_LENGTH * jnp.cos(jnp.deg2rad(angle))
    kz = 2.0 * jnp.pi / WAVE_LENGTH * jnp.sin(jnp.deg2rad(angle))
    # nearest integer mode numbers on the periodic box (k = 2 pi m / L)
    mx = round(kx * LX / (2.0 * jnp.pi))
    mz = round(kz * LZ / (2.0 * jnp.pi))
    omega, packet = nh.wave_package(
        base,
        mode_number={"x": mx, "y": 0, "z": mz},
        family="wave+",
        envelope=nh.gaussian(pos={"x": x, "z": MAKER_Z},
                             width=ENVELOPE_WIDTH))
    makers.append(fr.model.modules.Source(
        f"maker_{i}",
        pattern=packet,
        law=fr.Harmonic(
            amplitude=FORCING_AMPLITUDE,
            frequency=omega / (2.0 * jnp.pi))))

model = nh.Model(
    grid=grid,
    coriolis=nh.FPlaneCoriolis(f0=CORIOLIS_F0),
    buoyancy=nh.ConstantStratification(n2=STRATIFICATION_N2),
    advection=None,
    modules_extra=tuple(makers),
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Beam Angles
# -----------
# The dispersion relation of internal gravity waves sets the wave
# frequency from the direction of the wavevector alone. A carrier
# wavevector at angle :math:`\theta` to the horizontal oscillates at
#
# .. math::
#     \omega^2 = N^2 \cos^2\theta + f^2 \sin^2\theta,
#
# independent of the wavelength. Energy leaves the source along a
# beam perpendicular to the wavevector, so a steeper wavevector
# radiates a shallower beam. The two carriers point at 60 and 30
# degrees, so the sources run at different frequencies and send
# their beams off at different angles. Every axis is periodic here,
# so the sign of each carrier picks one running direction rather
# than a standing pair, and each source emits a single beam instead
# of a fan.
#
# Each source drives its own eigenmode at exactly that mode's
# frequency, and the box is periodic with no damping, so the forcing
# pumps the mode resonantly and the amplitude keeps climbing while
# the run lasts.
#
# Running and Recording
# ---------------------
# Six hours are long enough for both beams to cross most of the box
# and meet, and short enough that neither one reaches a wall, so no
# beam wraps around and re-enters from the far side. We write the
# buoyancy once per frame and render the slice.
writer = fr.io.Writer(
    "multiple_wave_makers.zarr", fields="b",
    trigger=fr.io.every(seconds=runlen / frames), mode="w")
model.run(runlen=runlen, outputs=writer)

# plot the final buoyancy in the slice plane. The colorbar takes its
# width out of the figure, so the figure aspect runs wider than the
# box and the data aspect is set on the axes, where the beam angles
# are read off.
plot = model.state.b.xr.isel(y=0).plot(x="x", size=3.6, aspect=1.6)
_ = plot.axes.set_aspect("equal")

# %%
_ = cv.record(
    "multiple_wave_makers.zarr", var="b", x="x", y="z", dims={"y": 0},
    plot_type="heatmap", ani_dim="time",
    kwargs={"colormap": "balance", "colorrange": (-2e-5, 2e-5),
            "figsize": (900, 600),
            "titlesize": 28, "xlabelsize": 24, "ylabelsize": 24,
            "animlabel": "{duration}",
            "title": "Wave beams at two angles"},
    filename="multiple_wave_makers.mp4", framerate=24)

# %%
# Each beam grows out of its source and keeps the angle its carrier
# sets. The steeper carrier gives the shallower beam, so the beam
# from the 60 degree source runs out across the box while the one
# from the 30 degree source drops almost twice as steeply for the
# ground it covers. Their paths converge, and the two beams overlap
# in the lower middle of the box in the closing frames.
