r"""
Equatorial Waves
================

The equator traps Kelvin, gravity, and Rossby waves.
"""

# %%
# The Wave Families
# -----------------
# For a zonal wavenumber :math:`k` and a meridional mode number
# :math:`m`, the wave frequencies of the unbounded equatorial beta
# plane are the roots of the dispersion relation
#
# .. math::
#     \omega_m \left(\omega_m^2 - c^2
#         \left(k^2 + (2m + 1)\frac{\beta}{c}\right)\right)
#     = k \beta c^2,
#
# with :math:`c` the gravity-wave phase speed. It has two fast gravity
# branches and the slow westward Rossby wave, all trapped within the
# equatorial Rossby radius :math:`R_e = \sqrt{c / \beta}` (Matsuno,
# 1966). We take the model's own discrete eigenmodes instead, which is
# simpler than adapting these solutions to the bounded domain.
#
# Experiment Settings
# -------------------
# An equatorial beta plane on a 6000 km by 3000 km basin. The phase
# speed is the one of the second baroclinic mode of a 4000 m ocean.
import subprocess

import numpy as np

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.shallowwater2 as sw

GRAVITY = 9.81                               # m/s^2
LX = 6000e3                                  # zonal extent, m
LY = 3000e3                                  # meridional extent, m

DEPTH = 4000.0                               # m
STRATIFICATION_N2 = 2.5e-5                   # 1/s^2
CSQR = STRATIFICATION_N2 * (DEPTH / (2 * np.pi)) ** 2
# the layer depth that carries this phase speed under gravity
EQUIVALENT_DEPTH = CSQR / GRAVITY            # about one metre

nx = ny = 128                                # grid cells per side
frames = 60                                  # animation frames per period

# %%
# Grid and Model
# --------------
# The zonal axis is periodic and the meridional axis is walled and
# centered on the equator, so the beta-plane Coriolis field
# :math:`f = \beta y` changes sign in the middle of the domain. The
# waves solve the linearized equations, so we assemble the model
# without the advection module.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny),
    extent=(LX, (-LY / 2, LY / 2)),
    periodic=(True, False))

dx = grid.factor("x").dx
# AB3 is stable up to a gravity-wave Courant number of 0.16 here
dt = 0.15 * dx / np.sqrt(CSQR)

model = sw.Model(
    grid=grid,
    core=sw.Core(gravity=GRAVITY, depth=EQUIVALENT_DEPTH),
    coriolis=sw.modules.BetaPlaneCoriolis.from_latitude(0.0),
    advection=False,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# The Numeric Eigenbasis
# ----------------------
# ``sw.eigenbasis`` solves the dense eigenvalue problem of the
# linearized model, one meridional column per zonal wavenumber, and
# labels every mode with its family. Positive frequencies propagate
# eastward. The labels are the slow ``vortical`` branch (the equatorial
# Rossby waves here), the ``kelvin+``/``kelvin-`` pair, and the fast
# ``wave+``/``wave-`` gravity branches.
eigenmodes = sw.eigenbasis(model)
dict(eigenmodes.families)

# %%
# One Function per Wave
# ---------------------
# ``eigenmodes.mode`` returns a labeled mode as its frequency and
# physical state. The zonal index counts wavelengths around the domain
# and the meridional index is the ordinal within the family, ordered by
# ascending frequency. Each animation selects the mode of zonal
# wavenumber two, integrates it for one wave period, and records it.
# The reset restarts the model clock, so every animation begins at
# time zero.
def simulate(family, m, title):
    """Integrate one labeled mode for a period and record it."""
    # a fresh model with the mode as its initial condition
    omega, state = eigenmodes.mode(family, mode_number={"x": 2, "y": m})
    model.reset()
    model.set_state(state)
    period = 2.0 * np.pi / abs(omega)
    # every mode has its own period, so the step is retuned to divide
    # it, and each of the frames with it
    model.update_parameters(
        {fr.model.params.TIME_STEP: fr.model.fit_dt(
            period, dt, parts=frames)})

    # write the pressure once per frame over that period
    tag = family.replace("+", "_east").replace("-", "_west")
    name = f"equatorial_{tag}_{m}"
    writer = fr.io.Writer(
        f"{name}.zarr", fields="p",
        trigger=fr.io.every(seconds=period / frames), mode="w")
    model.run(runlen=period, outputs=writer)

    # record the animation
    _ = subprocess.run(
        f"cdfviewer {name}.zarr -v p -x x -y y -p heatmap -a time"
        f" --kwargs='colormap=:balance, figsize=(1000, 500),"
        f" titlesize=28, xlabelsize=24, ylabelsize=24,"
        f' xunit="km", yunit="km",'
        f' animlabel="t = {{rawvalue}} days", animunit="days",'
        f' animlabelnumfmt="%.1f", animlabelsize=24,'
        f" title=\"{title}\"'"
        f" --record -s 'filename=\"{name}.mp4\", framerate=24'",
        shell=True, check=True)

# %%
# The first Rossby mode beyond the mixed Rossby-gravity wave shows the
# equatorial trapping, a double row of pressure cells straddling the
# equator that has decayed well before the walls.
_, rossby_mode = eigenmodes.mode("vortical", mode_number={"x": 2, "y": 1})
_ = rossby_mode.p.xr.plot(x="x", size=3.2, aspect=2)

# %%
# Kelvin Waves
# ------------
# The equatorial Kelvin wave propagates eastward, nondispersive and
# in exact geostrophic balance in the meridional direction. Its
# westward counterpart, ``kelvin-``, hugs the domain walls instead
# of the equator, and the first ``wave-`` mode is wall-trapped in
# the same way at a nearly equal frequency.
simulate("kelvin+", 0, title="Equatorial Kelvin wave")
simulate("kelvin-", 0, title="Boundary Kelvin wave, symmetric")
simulate("wave-", 0, title="Boundary Kelvin wave, antisymmetric")

# %%
# Eastward Gravity Waves
# ----------------------
# The three slowest modes of the eastward ``wave+`` branch. The first
# is the gravity branch of the mixed Rossby-gravity (Yanai) wave and
# the higher ordinals are the regular trapped gravity modes.
simulate("wave+", 0, title="Gravity-Yanai wave")
for m in (1, 2):
    simulate("wave+", m, title=f"Eastward gravity wave, mode {m}")

# %%
# Equatorial Rossby Waves
# -----------------------
# The slow ``vortical`` family drifts westward, an order of magnitude
# slower than the gravity waves. Compare the time spans of the
# animations. We skip ordinal zero, the wall mode that accompanies the
# boundary Kelvin wave at nearly the same frequency, which is not
# equatorially trapped.
for m in (1, 2, 3):
    simulate("vortical", m, title=f"Equatorial Rossby wave, mode {m}")

# %%
# Each animation covers one period of its mode. The numeric modes are
# exact eigenmodes of the discrete model, so the patterns propagate
# without changing shape and return to their initial state at the end
# of the loop.
