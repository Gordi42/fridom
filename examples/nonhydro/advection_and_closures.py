r"""
Advection and Closures
======================

Eight ways to handle the scales a model cannot resolve, measured
against each other on one rising bubble.
"""

# %%
# Experiment Settings
# -------------------
# The flow is the one from the
# :ref:`sphx_glr_auto_examples_nonhydro_bubble_test.py`: a warm bubble
# in a stratified box, buoyant enough to overturn the fluid above it,
# rolls into a mushroom, overshoots, collapses, and leaves behind
# intrusions, filaments and waves. Every stage of that hands structure
# to the grid scale, and a model has to decide what happens to it
# there.
#
# There are two ways to decide. One is to add a closure and let it
# remove the energy explicitly. The other is to use an advection
# scheme whose truncation error is dissipative and let it remove the
# energy implicitly. We run the same hour eight times and change
# nothing but that choice. The grid is half as fine as in the bubble
# test, 128 cells a side, so that all eight runs together cost about
# what the one did, and the step doubles with the cell.
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 2
import fridom as fr
import fridom.nonhydro2 as nh

box_size = 100.0          # metres, square
stratification = 2.5e-5   # N^2 [1/s^2]
bubble_buoyancy = 1e-3    # b_0 [m/s^2] at the centre of the bubble
bubble_radius = 10.0      # R [m]

n = 128                   # cells along x and z
runlen = 3600.0           # seconds, one hour
samples = 60              # diagnostic samples, one a model minute

dx = box_size / n
bubble_speed = (bubble_buoyancy * bubble_radius) ** 0.5   # free-rise speed
# a quarter of a cell per step at the rise speed, fitted so that the
# run and each of its sample intervals are whole numbers of steps
dt = fr.model.fit_dt(runlen, 0.25 * dx / bubble_speed, parts=samples)


def make_grid():
    """Return a fresh grid."""
    return fr.spatial.cartesian.Grid(
        shape=(n, 1, n),
        extent=(box_size, dx, box_size),
        periodic=(True, True, False))


def bubble(x, y, z):
    """Return the initial buoyancy, a Gaussian bump at the centre."""
    r2 = (x - 0.5 * box_size) ** 2 + (z - 0.5 * box_size) ** 2
    return bubble_buoyancy * jnp.exp(-r2 / bubble_radius**2)


# %%
# The Eight Configurations
# ------------------------
# The first four use centered advection, which is not dissipative at
# all, and differ only in the closure bolted onto it. The last four
# use an upwind-biased advection scheme and no closure whatsoever.
#
# The two constant coefficients are scaled to the flow rather than
# picked by hand. Writing :math:`k_{max} = \pi/\Delta x` for the
# Nyquist wavenumber and :math:`U` for the speed the bubble reaches, a
# harmonic coefficient :math:`\kappa = \gamma U / k_{max}` and a
# biharmonic one :math:`\kappa_4 = \gamma U / k_{max}^3` both damp the
# grid scale at a rate :math:`\gamma` times the advection there. Each
# then gets the gentlest :math:`\gamma` that still keeps its field
# free of grid-scale noise. The Smagorinsky closure sets its own
# viscosity from the resolved strain and runs at its default constant.
#
# ``UpwindAdvection`` and ``WENOAdvection`` accept orders 3 and 5, so
# those four runs span the full 2x2 of order against smoothness
# weighting. ``CenteredAdvection`` takes no order at all, being fixed
# at second.
nyquist = np.pi / dx

kappa = 0.01 * bubble_speed / nyquist        # harmonic
kappa4 = 0.03 * bubble_speed / nyquist**3    # biharmonic, needs more

# each entry builds its modules fresh, because a module instance binds
# to exactly one model
configurations = {
    "centered": lambda: (nh.CenteredAdvection(), ()),
    "harmonic": lambda: (nh.CenteredAdvection(), (
        fr.model.closures.HarmonicDiffusion(kappa=kappa),
        fr.model.closures.HarmonicFriction(nu=kappa))),
    "biharmonic": lambda: (nh.CenteredAdvection(), (
        fr.model.closures.BiharmonicDiffusion(kappa=kappa4),
        fr.model.closures.BiharmonicFriction(nu=kappa4))),
    "smagorinsky": lambda: (nh.CenteredAdvection(), (
        nh.SmagorinskyLilly(),)),
    "upwind 3": lambda: (nh.UpwindAdvection(order=3), ()),
    "upwind 5": lambda: (nh.UpwindAdvection(order=5), ()),
    "weno 3": lambda: (nh.WENOAdvection(order=3), ()),
    "weno 5": lambda: (nh.WENOAdvection(order=5), ()),
}

# %%
# What to Measure
# ---------------
# The conserved quantity of this system is
# :math:`E = \tfrac12 \int (u^2 + v^2 + w^2) + \tfrac12 \int b^2/N^2`,
# kinetic plus available potential energy, and it has to be evaluated
# on the staggered grid where the fields actually live. Interpolating
# the velocities to the cell centres first would annihilate exactly
# the grid-scale mode this page is about. ``EnergyMetric`` evaluates
# it in place.
#
# Mixing is measured against the sorted state, the same buoyancy field
# rearranged into a stable column. Its potential energy can only rise
# by mixing, never by stirring, and the available potential energy the
# bubble started with is the scale it is reported in. The potential
# energy is :math:`-\int b_{tot}\, z`, with the background
# stratification added back onto the anomaly by the model's ``b_total``
# diagnostic. It is written in field arithmetic: ``evaluation_nodes``
# hands out the coordinate of a field's own cells and ``integrate`` the
# volume integral, so the same lines run unchanged on a sharded grid.
# Only the sort is a global operation.
energy_metric = fr.model.EnergyMetric(
    {"u": 1.0, "v": 1.0, "w": 1.0, "b": 1.0 / stratification})


def potential_energy(b):
    """Return the potential energy of a total buoyancy field."""
    return -(b * b.evaluation_nodes("z")).integrate()


def stacked(b):
    """Return the same field rearranged into a stable column.

    The sorted values go back into the field layer by layer, densest
    at the bottom. The sort runs on the field's array, which jax
    gathers when the field is sharded.
    """
    nx, ny, nz = b.data.shape
    layers = jnp.sort(b.data.ravel()).reshape(nz, nx * ny)
    return b.with_data(layers.T.reshape(nx, ny, nz))


# %%
# Running the Eight
# -----------------
# Each configuration runs the same hour and is sampled once a model
# minute by an ``fr.io.Series``, which evaluates its columns on the
# host. Nothing goes to disk, since these are scalars.


def run(build):
    """Run one configuration and return its series and final field."""
    advection, extra = build()
    model = nh.Model(
        grid=make_grid(),
        buoyancy=nh.ConstantStratification(n2=stratification),
        advection=advection,
        modules_extra=extra,
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3))
    model.set_fields(b=bubble)
    # the series samples the model's own total buoyancy
    total = model.diagnostics.b_total
    diagnostics = {
        "energy": lambda ms: 0.5 * energy_metric.inner(
            ms.state, ms.state).real,
        "potential": lambda ms: potential_energy(total(ms.state)),
        "background": lambda ms: potential_energy(stacked(total(ms.state))),
    }
    series = fr.io.Series(diagnostics,
                          trigger=fr.io.every(seconds=runlen / samples))
    model.run(runlen=runlen, outputs=series)
    return series, model.state["b"]


results = {name: run(build) for name, build in configurations.items()}

# %%
# Conservation and Dissipation
# ----------------------------
# Two panels. Total energy shows how much each option removes. The
# background potential energy is the energy of the sorted state, so
# its rise is the part of the stirring that has become irreversible
# mixing, expressed as a fraction of the available potential energy
# the bubble started with, which is the gap between the potential
# energy and its sorted floor at the first sample.
figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for name, (series, _field) in results.items():
    dashed = name.startswith(("upwind", "weno"))
    style = {"linestyle": "--" if dashed else "-", "linewidth": 1.4}
    minutes = series.time / 60.0
    energy = series["energy"]
    axes[0].plot(minutes, energy / energy[0], label=name, **style)
    available = series["potential"][0] - series["background"][0]
    mixed = series["background"] - series["background"][0]
    axes[1].plot(minutes, mixed / available, label=name, **style)

axes[0].set_title("total energy $E/E(0)$")
axes[1].set_title("irreversible mixing $\\Delta E_b / E_a(0)$")
for axis in axes:
    axis.set_xlabel("time [min]")
    axis.grid(alpha=0.3)
_ = axes[0].legend(fontsize=8, ncol=2)

# %%
# Two things are worth reading off these curves.
#
# **Centered advection with no closure conserves energy almost
# exactly, and that is its problem.** It loses four tenths of one
# percent over the hour, where everything else loses a fifth or more.
# But it has no way to remove the structure the cascade keeps
# delivering, so the grid fills up instead, and its irreversible
# mixing stays under one percent. It does not blow up. Energy
# conservation is precisely what stops it. It converts the cascade
# into reversible noise rather than into mixing, and the fields below
# show what that noise looks like.
#
# **Everything that does dissipate dissipates about the same amount.**
# With the two coefficients turned down to the gentlest setting that
# clears the noise, the seven dissipative runs land between 0.75 and
# 0.80 of their initial energy and between 0.17 and 0.22 of their
# available potential energy mixed away. Seven quite different
# mechanisms, agreeing to within a few percent. The rate is set by the
# large scales, which all eight resolve identically, and the scheme
# only decides at what scale the energy leaves and how much structure
# survives the trip. It is worth knowing before reaching for a closure
# that the total is not the knob it looks like.
#
# The Fields
# ----------
figure, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
for axis, (name, (_series, field)) in zip(axes.ravel(), results.items(),
                                          strict=True):
    slab = np.asarray(field.data)[:, 0, :]
    axis.imshow(slab.T, origin="lower", cmap="RdBu_r",
                vmin=-3e-4, vmax=3e-4, extent=(0, box_size, 0, box_size))
    axis.set_title(name, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])

# %%
# The fields say what the integrals cannot. Centered advection is
# visibly speckled and nothing else is, which is the whole of the
# difference between conserving energy and doing something useful with
# it. Past that the seven differ in degree rather than in kind. The
# three closures keep a grain in the cap that the biased schemes do
# not, each keeps the mushroom, and what separates them is how much
# of the filamentary detail around it survives the hour.
#
# Between the four biased schemes, the extra order is what shows. WENO
# at fifth order retains two and a half times the small-scale buoyancy
# variance of WENO at third, and the thin filaments of the roll-up
# survive where the third-order run has rounded them into lobes. The
# two still end the hour within one percent of the same total energy,
# which is the point worth carrying away. **The extra order buys
# effective resolution, not conservation.**
#
# One caveat on reading the orders as accuracy. Both schemes reconstruct
# at their stated order, but the tendency they contribute is formally
# second order once the advecting velocity varies. The higher order
# buys the non-oscillatory behaviour at fronts and the effective
# resolution above, not a faster convergence rate.
