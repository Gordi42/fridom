r"""
Reflecting Wave Packet
======================

A polarized internal-wave packet sinks through a stratified box and
reflects off the bottom.
"""

# %%
# Experiment Settings
# -------------------
# An x-z slice, one cell thick in y, with mid-latitude rotation and
# a constant stratification. Rigid walls close the box at the top
# and the bottom. The horizontal directions stay periodic.
import subprocess

import jax.numpy as jnp

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.nonhydro2 as nh

CORIOLIS_F0 = 1e-4                  # 1/s
STRATIFICATION_N2 = 2.5e-5          # 1/s^2
LX, LY, LZ = 2000.0, 1.0, 1000.0    # box extents, m

nx, ny, nz = 256, 1, 128
frames = 240
runlen = 8.0 * 3600.0                # eight hours of model time

# %%
# Grid and Model
# --------------
# The packet solves the linearized equations, so we assemble the
# model without the advection module.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(LX, LY, LZ),
    periodic=(True, True, False))

# omega dt <= 0.1, fitted so that the run and each of its frames
# are whole numbers of steps
dt = fr.model.fit_dt(
    runlen, 0.1 / STRATIFICATION_N2 ** 0.5, parts=frames)

model = nh.Model(
    grid=grid,
    coriolis=nh.FPlaneCoriolis(f0=CORIOLIS_F0),
    buoyancy=nh.ConstantStratification(n2=STRATIFICATION_N2),
    advection=False,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# A Traveling Packet from One Carrier Mode
# ----------------------------------------
# ``nh.wave_package`` multiplies one eigenmode with an envelope
# function and projects the product back onto its wave family, so
# the packet stays polarized. Between rigid walls a single mode is
# standing in z, and an envelope alone would split into an upward
# and a downward beam. ``traveling={"z": -1}`` selects the sinking
# one. The sign is the direction the envelope drifts. On the walled
# vertical the ``z`` mode number counts half wavelengths over the
# depth. This carrier has a 133 m wavelength in x and in z.
def envelope(x, z):
    """Gaussian bump at the middle of the slice, 1/e radius 150 m."""
    r2 = (x - 500.0) ** 2 + (z - 500.0) ** 2
    return jnp.exp(-r2 / 150.0 ** 2)

omega, packet = nh.wave_package(
    model,
    mode_number={"x": 15, "y": 0, "z": 15},
    family="wave+",
    envelope=envelope,
    traveling={"z": -1})
print(f"carrier period: {2 * jnp.pi / omega / 60:.1f} min")

# normalize the packet so the largest buoyancy value is one
packet /= packet.b.max()
model.set_state(packet)

# plot the initial buoyancy in the slice plane
_ = model.state.b.xr.isel(y=0).plot(x="x", size=3.2, aspect=2)

# %%
# .. note::
#     The built-in ``nh.gaussian`` builds the same callable, with a
#     single width applied to every named axis.
#
#     .. code-block:: python
#
#         envelope = nh.gaussian(
#             pos={"x": 500.0, "z": 500.0}, width=150.0)

# %%
# Running and Recording
# ---------------------
# The envelope sinks at about 4 cm/s and meets the bottom after
# roughly four hours. Eight hours shows the descent, the
# reflection, and the climb back toward mid depth. We write the
# buoyancy once per frame and render the slice.
writer = fr.io.Writer(
    "wave_package.zarr", fields="b",
    trigger=fr.io.every(seconds=runlen / frames), mode="w")
model.run(runlen=runlen, outputs=writer)

_ = subprocess.run(
    "cdfviewer wave_package.zarr -v b -x x -y z --dims=y=0"
    " -p heatmap -a time"
    " --kwargs='colormap=:balance, colorrange=(-1, 1),"
    " figsize=(1000, 500),"
    " titlesize=28, xlabelsize=24, ylabelsize=24,"
    ' xlabel="x [m]", ylabel="z [m]",'
    ' animunit="hours", animlabelnumfmt="%.1f",'
    " title=\"Reflecting internal-wave packet\"'"
    " --record -s 'filename=\"wave_package.mp4\", framerate=24'",
    shell=True, check=True)

# %%
# The packet glides down along the tilted phase lines of the
# carrier, reflects off the bottom into its mirror image, and
# climbs back. Within the envelope the crests keep sweeping through
# at the carrier period of about half an hour. The packet also
# smears out as it travels. The envelope holds a band of modes
# around the carrier, and each mode moves at a slightly different
# group velocity. A larger carrier wavenumber or a wider envelope
# (both narrower relative to the carrier in spectral space) would
# keep the packet compact for longer.
