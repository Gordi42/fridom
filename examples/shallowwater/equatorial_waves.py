r"""
Equatorial Waves
================

The equator acts as a waveguide: because the Coriolis parameter
changes sign there, the shallow-water equations on the equatorial
beta plane :math:`f = \beta y` carry wave modes that are trapped
around the equator. We compute the discrete wave modes of the model
numerically, select the two propagating families, eastward gravity
waves and westward Rossby waves, and render one animation per
meridional mode.
"""

# %%
# The Wave Families
# -----------------
# For a zonal wavenumber :math:`k` and a meridional mode number
# :math:`m`, the wave frequencies :math:`\omega_m` of the unbounded
# equatorial beta plane are the roots of the dispersion relation
#
# .. math::
#     \omega_m \left(\omega_m^2 - c^2
#         \left(k^2 + (2m + 1)\frac{\beta}{c}\right)\right)
#     = k \beta c^2,
#
# where :math:`c` is the gravity-wave phase speed: two fast gravity
# branches and the slow westward equatorial Rossby wave, all trapped
# within the equatorial Rossby radius :math:`R_e = \sqrt{c / \beta}`
# (Matsuno, 1966). On a meridionally bounded domain, however, exact
# analytic modes are hard to write down. Therefore we take the model's
# own discrete eigenmodes: ``sw.eigenbasis`` computes them numerically
# for every zonal wavenumber and labels the families through the
# Kelvin separatrix (the Kelvin frequency separates the slow vortical
# family from the fast wave branches for any Coriolis profile).
#
# Experiment Settings
# -------------------
# We work on an Earth-like beta plane: an Atlantic-sized basin of
# sixty degrees longitude and thirty degrees around the equator, and
# the phase speed of the second baroclinic vertical mode of a
# 4000 m deep ocean, which keeps the equatorial Rossby radius well
# clear of the walls:
import os
import subprocess

import numpy as np

# sphinx_gallery_thumbnail_number = 1
import fridom as fr
import fridom.shallowwater2 as sw
from fridom.spatial.meshes import IntervalMesh

EARTH_RADIUS = 6371e3                        # m
OMEGA = 2 * np.pi / 86400                    # rotation rate, 1/s
BETA = 2 * OMEGA / EARTH_RADIUS              # df/dy at the equator
LX = EARTH_RADIUS * np.deg2rad(60)           # zonal extent, m
LY = EARTH_RADIUS * np.deg2rad(30)           # meridional extent, m

DEPTH = 4000.0                               # m
STRATIFICATION_N2 = 2.5e-5                   # 1/s^2
CSQR = STRATIFICATION_N2 * (DEPTH / (2 * np.pi)) ** 2

fast = "FRIDOM_EXAMPLES_FAST" in os.environ
nx = ny = 64 if fast else 128
frames = 30 if fast else 60

# %%
# Grid and Model
# --------------
# The zonal axis is periodic. The meridional axis is walled and
# centered on the equator, so the beta-plane Coriolis field
# :math:`f = \beta y` changes sign in the middle of the domain. The
# waves solve the linearized equations. Therefore we assemble the
# model without the advection module:
mesh_x = IntervalMesh(nx, (0.0, LX), periodic=True, name="x")
mesh_y = IntervalMesh(ny, (-LY / 2, LY / 2), periodic=False, name="y")
grid = fr.spatial.Grid((mesh_x, mesh_y))
dt = 0.1 * (LX / nx) / np.sqrt(CSQR)

model = sw.Model(
    grid=grid,
    csqr=CSQR,
    coriolis=fr.model.modules.BetaPlaneCoriolis(f0=0.0, beta=BETA),
    advection=False,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# The Numeric Eigenbasis
# ----------------------
# ``sw.eigenbasis`` solves the dense eigenvalue problem of the
# linearized model, one meridional column per zonal wavenumber, and
# labels every mode with its family. Positive frequencies propagate
# eastward. The labeled families are the slow ``vortical`` branch
# (the equatorial Rossby waves here), the ``kelvin+``/``kelvin-``
# pair (here the eastward equatorial Kelvin wave and its westward
# boundary-trapped counterpart), and the fast ``wave+``/``wave-``
# gravity branches:
em = sw.eigenbasis(model)
dict(em.families)

# %%
# One Function per Wave
# ---------------------
# ``em.mode`` returns one labeled mode as its frequency together with
# the physical state: the zonal index counts wavelengths around the
# domain, and the meridional index is the mode ordinal within the
# family, ordered by ascending frequency magnitude. Every animation
# follows the same recipe: select the mode of zonal wavenumber two,
# integrate it for one wave period, write the pressure to a zarr
# store, and record the animation with CDFViewer:
def simulate(family, m, title):
    """Integrate one labeled mode for a period and record it."""
    omega, z = em.mode(family, {"x": 2, "y": m})
    model.set_state(z)
    period = 2.0 * np.pi / abs(omega)
    name = f"equatorial_{family.strip('+-')}_{m}"
    writer = fr.model.io.Writer(
        f"{name}.zarr", fields=["p"],
        trigger=fr.model.io.every(seconds=period / frames), mode="w")
    model.run(runlen=period, outputs=(writer,), progress=False)
    _ = subprocess.run(
        f"cdfviewer {name}.zarr -v p -x x -y y -p heatmap -a time"
        f" --kwargs='colormap=:balance, title=\"{title}\"'"
        f" --record -s 'filename=\"{name}.mp4\", framerate=24'",
        shell=True, check=True)

# %%
# Before running the families, we look at the initial pressure of the
# first Rossby mode beyond the mixed Rossby-gravity wave. It shows the
# equatorial trapping: a row of pressure cells on the equator, decayed
# well before the walls:
_, z0 = em.mode("vortical", {"x": 2, "y": 1})
_ = z0.p.xr.plot(x="x")

# %%
# The Equatorial Kelvin Wave
# --------------------------
# The most famous equatorial mode: nondispersive, eastward, and in
# exact geostrophic balance in the meridional direction. Its westward
# twin at the same frequency magnitude, ``kelvin-``, hugs the domain
# walls instead of the equator:
simulate("kelvin+", 0, title="Equatorial Kelvin wave")

# %%
# Eastward Gravity Waves
# ----------------------
# The three slowest modes of the eastward ``wave+`` branch. The first
# one is the gravity branch of the mixed Rossby-gravity (Yanai) wave,
# and the higher ordinals are the regular trapped gravity modes:
simulate("wave+", 0, title="Gravity-Yanai wave")
for m in (1, 2):
    simulate("wave+", m, title=f"Eastward gravity wave, mode {m}")

# %%
# Equatorial Rossby Waves
# -----------------------
# The slow ``vortical`` family drifts westward, an order of magnitude
# slower than the gravity waves. Compare the time spans of the
# animations. We skip ordinal zero, which is not equatorially trapped:
# it is the wall mode that accompanies the boundary Kelvin wave at
# nearly the same frequency:
for m in (1, 2, 3):
    simulate("vortical", m, title=f"Equatorial Rossby wave, mode {m}")

# %%
# Each animation covers one period of its mode. In contrast to
# sampled analytic expressions, the numeric modes are exact
# eigenmodes of the discrete model, so the patterns propagate without
# changing shape and return to their initial state at the end of the
# loop.
