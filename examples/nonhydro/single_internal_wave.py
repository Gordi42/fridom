r"""
Single Internal Wave
====================

One polarized internal-wave mode crosses a rotating stratified box.
"""

# %%
# Experiment Settings
# -------------------
# A triply periodic box with mid-latitude rotation and a constant
# stratification. The two frequencies bracket the internal-wave
# band, :math:`f < \omega < N`.
import subprocess

import numpy as np

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.nonhydro2 as nh

CORIOLIS_F0 = 1e-4                 # 1/s
STRATIFICATION_N2 = 2.5e-5         # 1/s^2
LX, LY, LZ = 300.0, 100.0, 100.0   # box extents, m

nx, ny, nz = 96, 32, 32            # a single mode needs no more
frames = 60

# %%
# Grid and Model
# --------------
# The wave solves the linearized equations, so we assemble the model
# without the advection module. Internal-wave frequencies are capped
# by :math:`N`, so a time step that resolves :math:`N` resolves every
# mode in the box.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(LX, LY, LZ),
    periodic=(True, True, True))

dt = 0.1 / np.sqrt(STRATIFICATION_N2)      # omega dt <= 0.1

model = nh.Model(
    grid=grid,
    core=nh.Core(),
    coriolis=nh.FPlaneCoriolis(f0=CORIOLIS_F0),
    buoyancy=nh.ConstantStratification(n2=STRATIFICATION_N2),
    advection=False,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# One Mode from the Eigenbasis
# ----------------------------
# On the fully periodic grid ``nh.eigenbasis`` diagonalizes the
# linearized model analytically. We take the positive inertia-gravity
# branch with two zonal wavelengths, no meridional structure, and one
# vertical wavelength, and set it as the initial condition.
eigenmodes = nh.eigenbasis(model)
omega, wave = eigenmodes.mode("wave+", indices={"x": 2, "y": 0, "z": 1})
period = 2.0 * np.pi / abs(omega)
model.set_state(wave)

# plot the initial buoyancy in the front plane
_ = model.state.b.xr.isel(y=0).plot(x="x", size=2.4, aspect=3)

# %%
# Running and Recording
# ---------------------
# We write the buoyancy once per frame over one wave period and
# record a top view and a front view of the store.
writer = fr.io.Writer(
    "single_internal_wave.zarr", fields=["b"],
    trigger=fr.io.every(seconds=period / frames), mode="w")
model.run(runlen=period, outputs=(writer,), progress=False)

views = {
    "top": ("-x x -y y --dims=z=0", "y"),
    "front": ("-x x -y z --dims=y=0", "z"),
}
for view, (axes, yax) in views.items():
    _ = subprocess.run(
        f"cdfviewer single_internal_wave.zarr -v b {axes} -p heatmap"
        f" -a time --kwargs='colormap=:balance, figsize=(1000, 450),"
        f" titlesize=28, xlabelsize=24, ylabelsize=24,"
        f' xlabel="x [m]", ylabel="{yax} [m]",'
        f" title=\"Single internal wave, {view} view\"'"
        f" --record -s 'filename=\"single_internal_wave_{view}.mp4\","
        f" framerate=24'",
        shell=True, check=True)

# %%
# The crests translate through the box and return to their initial
# position after one period. The front view shows the phase lines
# tilted by the wave vector.
