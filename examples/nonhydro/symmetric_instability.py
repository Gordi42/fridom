r"""
Symmetric Instability
=====================

A front in thermal wind balance overturns along its own isopycnals.

Parameters follow Stamper and Taylor (2016), *The transition from
symmetric to baroclinic instability in the Eady model*; the linear
theory of the problem goes back to Stone (1966), *On non-geostrophic
baroclinic stability*.
"""

# %%
# Experiment Settings
# -------------------
# A front is a horizontal buoyancy gradient, and rotation holds it up
# with a vertical shear through thermal wind. That balance is exact but
# not always stable. When the Richardson number
# :math:`\mathrm{Ri} = N^2 f_0^2 / (M^2)^2` drops below one, the flow
# can release energy by overturning in slanted cells that follow the
# tilted isopycnals rather than crossing them, which is why the
# instability is called symmetric.
#
# The background state is
#
# .. math::
#     V(z) = \frac{M^2}{f_0} z
#     \quad , \quad
#     B(x, z) = N^2 z + M^2 x
#
# with :math:`f_0 \partial_z V = \partial_x B = M^2`, which is thermal
# wind. Every field is taken independent of the along-front direction,
# so the problem is two-dimensional in the cross-front plane. The front
# sits at :math:`\mathrm{Ri} = 0.25`, well inside the unstable range.
import subprocess

import jax.numpy as jnp
import numpy as np

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

coriolis = 1e-4           # f0 [1/s]
front_gradient = -1e-7    # M^2 = dB/dx [1/s^2]
richardson_number = 0.25  # Ri = N^2 f0^2 / M^4
box_length = 500.0        # metres across the front
box_depth = 200.0         # metres

seed_wavenumber = 4       # cells across the box
seed_amplitude = 5e-6     # m/s^2 of buoyancy anomaly to start from

runlen = 8.0 * 3600.0     # seconds

nz = 128
nx = 2 * nz
ny = 1                    # the flow is two-dimensional, in x and z
frames = 288

# %%
# Linear Theory
# -------------
# :class:`~fridom.nonhydro2.modules.thermal_wind.ThermalWindBackground`
# supplies the two terms the background adds to the perturbation
# equations, the tilting of the mean shear by :math:`w` and the
# advection of the mean buoyancy by :math:`u`. It reads the Coriolis
# parameter from the model, so thermal wind holds by construction
# rather than by the caller getting the arithmetic right. There is no
# background advection term, because the mean flow points along the
# front and nothing varies in that direction.
#
# Those two terms are all it takes to get the growth rate in closed
# form. The derivation is not needed to run the example, and only its
# result is used later, to tilt the seed. Linearized about the
# background, with nothing varying along the front, the equations of
# motion are
#
# .. math::
#     \partial_t u - f_0 v = -\partial_x p
#     \quad , \quad
#     \partial_t w = -\partial_z p + b
#     \quad , \quad
#     \partial_x u + \partial_z w = 0 ,
#
# .. math::
#     \partial_t v = -f_0 u - \frac{M^2}{f_0} w
#     \quad , \quad
#     \partial_t b = -M^2 u - N^2 w ,
#
# where the terms in :math:`M^2` are the two the background supplies.
# Continuity lets a streamfunction carry the cross-front flow,
# :math:`u = \partial_z \psi` and :math:`w = -\partial_x \psi`, and the
# pressure drops out of the difference between the :math:`z` derivative
# of the :math:`u` equation and the :math:`x` derivative of the
# :math:`w` equation, since both contain
# :math:`\partial_x \partial_z p`:
#
# .. math::
#     \partial_t \nabla^2 \psi = f_0 \partial_z v - \partial_x b .
#
# Now take a disturbance growing as :math:`e^{\sigma t}` with
# :math:`\psi \propto \sin(k x + m z)`, so that :math:`u`, :math:`w`,
# :math:`v` and :math:`b` all vary as :math:`\cos(k x + m z)`. The
# three equations turn into three relations between the amplitudes,
#
# .. math::
#     \sigma v = \left(\frac{M^2}{f_0} k - f_0 m\right) \psi
#     \quad , \quad
#     \sigma b = \left(N^2 k - M^2 m\right) \psi
#     \quad , \quad
#     \sigma (k^2 + m^2) \psi = f_0 m v - k b ,
#
# and substituting the first two into the third gives the dispersion
# relation
#
# .. math::
#     \sigma^2 (k^2 + m^2) = -N^2 k^2 + 2 M^2 k m - f_0^2 m^2 .
#
# After some algebra, the fastest growth rate is
#
# .. math::
#     \sigma_\text{max}^2 = \frac{\sqrt{(N^2 - f_0^2)^2 + 4 (M^2)^2}
#                           - (N^2 + f_0^2)}{2} .
#
# It is positive exactly when :math:`(M^2)^2 > N^2 f_0^2`, which is
# :math:`\mathrm{Ri} < 1`. It is maximum for
#
# .. math::
#     \frac{m}{k} = \frac{M^2}{\sigma_\text{max}^2 + f_0^2} .
#
# Here :math:`M^2` is negative, so :math:`m` and :math:`k` have
# opposite signs and the wave leans the same way as the isopycnals.
#
front = nh.ThermalWindBackground(m2=front_gradient)
stratification = front.stratification_n2(
    coriolis, richardson_number=richardson_number)

discriminant = np.sqrt(
    (stratification - coriolis**2) ** 2 + 4 * front_gradient**2)
rate = np.sqrt(0.5 * (discriminant - (stratification + coriolis**2)))

# %%
# Grid and Model
# --------------
# The box is periodic across the front and walled at the top and the
# bottom. Fifth-order WENO advection carries the overturning, and a
# ten-second step resolves it.
grid = fr.spatial.cartesian.Grid(
    shape=(nx, ny, nz),
    extent=(box_length, box_depth, box_depth),
    periodic=(True, True, False))

# fitted so that the run and each of its frames are whole numbers of
# steps
dt = fr.model.fit_dt(runlen, 10.0, parts=frames)

model = nh.Model(
    grid=grid,
    core=nh.Core(aspect_ratio=1.0),
    coriolis=nh.FPlaneCoriolis(f0=coriolis),
    buoyancy=nh.ConstantStratification(n2=stratification),
    advection=nh.WENOAdvection(order=5),
    modules_extra=front,
    time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))

# %%
# Seeding a Single Mode
# ---------------------
# The fastest wave cannot be the seed as it stands, because it does
# not vanish at the top and the bottom of the box, where :math:`w`
# must. We fade it out with the gravest standing envelope,
# :math:`\sin(\pi z / L_z)`, and place four wavelengths across the
# box. The result differs slightly from the fastest mode the box
# itself admits, which grows about one percent slower than
# :math:`\sigma_\text{max}` and has a tilt to match, but a seed only
# needs a good projection onto the growing mode.
#
# The seed is a buoyancy anomaly, because buoyancy is what the
# animation shows. It carries the buoyancy of the mode but not its
# velocities, and the flow spends the first hours building those
# before the growth sets in.
wavenumber = 2 * np.pi * seed_wavenumber / box_length
tilt = wavenumber * front_gradient / (rate**2 + coriolis**2)


def seed(x, y, z):  # y is named but the flow is two-dimensional
    """Return the tilted mode, faded out at the two walls."""
    return (seed_amplitude * jnp.cos(wavenumber * x + tilt * z)
            * jnp.sin(jnp.pi * z / box_depth))


model.set_fields(b=seed)


def plot_slice(field):
    """Draw the x-z slice with the background isopycnals over it."""
    slice_2d = field.xr.isel(y=0, drop=True)
    plot = slice_2d.plot(x="x", size=2.4, aspect=3.1, cmap="RdBu_r")
    plot.axes.set_aspect("equal")
    background = (stratification * slice_2d.z
                  + front_gradient * slice_2d.x)
    background.plot.contour(
        x="x", ax=plot.axes, colors="black", linewidths=0.7,
        linestyles="solid", add_colorbar=False)


plot_slice(model.state.b)

# %%
# The seed leans along the black lines, the background isopycnals, and
# the envelope fades it out at the two walls.
#
# Running and Writing Output
# --------------------------
# The seed is large enough to show on the colour scale of the
# animation from the first frame, and eight hours carry it through the
# roll-up to the breakdown. Every frame writes the buoyancy to a zarr
# store.
writer = fr.io.Writer(
    "symmetric_instability.zarr",
    fields="b",
    trigger=fr.io.every(time_units=runlen / frames),
    mode="w")

model.run(runlen=runlen, outputs=writer)

# %%
# The bands lean along the isopycnals and steepen for about four
# hours, then roll up into a regular row of cells that repeats across
# the box. Within an hour the rolls tangle and break into finer
# filaments. The run ends with the buoyancy anomaly positive along the
# top of the box and negative along the bottom, with filaments in
# between.
plot_slice(model.state.b)

# %%
# Rendering the Animation
# -----------------------
# `CDFViewer <https://gordi42.github.io/CDFViewer.jl/>`_ records the
# buoyancy animation from the store.
command = (
    "cdfviewer symmetric_instability.zarr"
    " -v b -x x -y z --dims=y=0 -p heatmap -a time"
    " --kwargs='animlabel=\"{duration}\", colormap=:balance,"
    ' animlabelnumfmt="%.0f",'
    " colorrange=(-8e-5, 8e-5)'"
    " --record -s 'filename=\"symmetric_instability.mp4\", framerate=24'"
)
_ = subprocess.run(command, shell=True, check=True)
