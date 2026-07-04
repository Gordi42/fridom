r"""
Equatorial Waves.
=================

Simulation of linear equatorial waves on the equatorial beta-plane.

Description
-----------
We consider the linearized shallow water equations in the equatorial
:math:`\beta`-plane approximation. This system is given by:

.. math::

    \begin{align}
        \partial_t \mathbf z = - i \mathcal L \cdot \mathbf z
        \quad \text{with} \quad
        \mathbf z = \begin{pmatrix} u \\ v \\ p \end{pmatrix}
        \quad \text{and} \quad
        \mathcal L = \begin{pmatrix}
            0 & i \beta y & - i \partial_x \\
            - i \beta y & 0 & - i \partial_y \\
            -i c^2 \partial_x & -i c^2 \partial_y & 0 \end{pmatrix}
    \end{align}

we search for eigenvalues and corresponding eigenfunctions of the operator
:math:`\mathcal L`. For a given eigenfunction :math:`\mathbf z_m`, the eigenvalue
:math:`\omega_m` corresponds to the frequency of that eigenfunction:

.. math::

    \begin{align}
        \partial_t \mathbf z_m = - i \omega_m \mathbf z_m
        \quad \Rightarrow \quad
        \mathbf z_m(t) = \mathbf z_m(0) e^{-i \omega_m t}
    \end{align}

After some algebra, we find that the eigenvalues are implicitly given by the roots
of the dispersion relation:

.. math::
    \begin{align}
        \omega_m \left( \omega_m^2 - c^2 \left( k^2 + (2m + 1) \frac{\beta}{c} \right)
        \right) = k \beta c^2
    \end{align}

where $k$ is the zonal wavenumber and $m$ the meridional mode number.
This equation has three solutions. In general, one solution is close to zero and
corresponds to the equatorial Rossby waves. From the other two solutions, one is
positive and the other negative. The positive solution corresponds to eastward
propagating gravity waves, and the negative solution to westward propagating
gravity waves.

The corresponding eigenfunctions are given by:

.. math::

    \mathbf z_m = \begin{pmatrix} u_m \\ v_m \\ p_m \end{pmatrix}
                e^{- \frac{1}{2}\tilde{y}^2} e^{i(kx + \phi)}
    \quad \text{with} \quad
    \tilde{y} = \frac{y}{R_e}
    \quad \text{and} \quad
    R_e^2 = \frac{c}{\beta}

with

.. math::

    \begin{align}
        u_m &= \frac{i c}{2 R_e} \left( \frac{H_{m+1}(\tilde{y})}{\omega_m - c k}
            + \frac{2 m H_{m-1}(\tilde{y})}{\omega_m + c k} \right) \\
        v_m &= H_m(\tilde{y})  \\
        p_m &= \frac{i c^2}{2 R_e} \left( \frac{H_{m+1}(\tilde{y})}{\omega_m - c k}
            - \frac{2 m H_{m-1}(\tilde{y})}{\omega_m + c k} \right)
    \end{align}

where :math:`H_m` is the m-th Hermite polynomial, given by the recursive relation

.. math::

    H_m(x) = 2x H_{m-1}(x) - 2(m-1) H_{m-2}(x)

with :math:`H_{-1}(x) = 0` and :math:`H_0(x) = 1`.

Animation
---------

Eastward Propagating Gravity Waves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 3

    .. grid-item::

        .. math::
            m = 0

        .. video:: videos/equatorial_lon2_equ0_wave2.mp4
            :loop:

    .. grid-item::

        .. math::
            m = 1

        .. video:: videos/equatorial_lon2_equ1_wave2.mp4
            :loop:

    .. grid-item::

        .. math::
            m = 2

        .. video:: videos/equatorial_lon2_equ2_wave2.mp4
            :loop:

Equatorial Rossby Waves
^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 3

    .. grid-item::

        .. math::
            m = 0

        .. video:: videos/equatorial_lon2_equ0_wave1.mp4
            :loop:

    .. grid-item::

        .. math::
            m = 1

        .. video:: videos/equatorial_lon2_equ1_wave1.mp4
            :loop:

    .. grid-item::

        .. math::
            m = 2

        .. video:: videos/equatorial_lon2_equ2_wave1.mp4
            :loop:

"""

import matplotlib.pyplot as plt
import numpy as np

import fridom.shallowwater as sw

# ----------------------------------------------------------------
#  Experiment settings
# ----------------------------------------------------------------
# General settings
make_video  = True
fps         = 30
make_netcdf = False
exp_name    = "equatorial_waves"
thumbnail   = f"figures/{exp_name}.png"

# ----------------------------------------------------------------
#  Physical constants
# ----------------------------------------------------------------

# Constants related to the Earth
EARTH_RADIUS = 6371e3
OMEGA = 2 * np.pi / 86400

# Define domain extent
LATITUDE_EXTENT = [-30, 30]
LONGITUDE_EXTENT = [-50, 10]  # atlantic ocean, pacific would be [-140, -80]
DEPTH = 4000
RESOLUTION_POWER = 8

# pseudo cartesian coordinates (x, y) in meters
LENGTH_X = EARTH_RADIUS * np.diff(np.deg2rad(LONGITUDE_EXTENT))[0]
LENGTH_Y = EARTH_RADIUS * np.diff(np.deg2rad(LATITUDE_EXTENT))[0]
y0 = LENGTH_Y / 2

# Compute the coriolis parameter at the equator
LATITUDE = 0
F0 = 2 * OMEGA * np.sin(np.deg2rad(LATITUDE))
BETA = 2 * OMEGA / EARTH_RADIUS * np.cos(np.deg2rad(LATITUDE))

# Compute the phase velocity of a given mode
VERTICAL_MODE = 1
STRATIFICATION_N2 = 2.5e-5
CSQR = STRATIFICATION_N2 * ( DEPTH / (np.pi * VERTICAL_MODE) ) ** 2

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class Plotter(sw.modules.animation.ModelPlotter):

    """Custom plotter for the barotropic instability experiment."""

    @staticmethod
    def create_figure() -> plt.Figure:
        """Create a figure with a specific size and resolution."""
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=True)

    @staticmethod
    def prepare_arguments(mz: sw.ModelState) -> dict:
        """Prepare the arguments for the plot function."""
        # skip every 4th point for the quiver plot
        skip = 2**(RESOLUTION_POWER-5)
        state = mz.z.xrs[::skip,::skip]
        pot_vort = mz.z.p.xr
        return {"state": state, "pot_vort": pot_vort, "t": mz.clock.time}

    @staticmethod
    def update_figure(fig: plt.Figure, *_args: any, **kwargs: any) -> None:
        """Plot the fields on the figure."""
        # get the keyword arguments
        state = kwargs["state"]
        pot_vort = kwargs["pot_vort"]

        # convert the coordinates to km
        state["y"] = (state.y - y0) / 1e3
        state["x"] = state.x / 1e3
        pot_vort["y"] = (pot_vort.y - y0) / 1e3
        pot_vort["x"] = pot_vort.x / 1e3

        t = kwargs["t"] / 86400
        # plot the potential vorticity and the velocity field
        ax = fig.add_subplot(111)
        pot_vort.plot(ax=ax, cmap="RdBu_r", extend="both", vmax=8, vmin=-8)
        key = state.plot.quiver("x", "y", "u", "v", scale=35, add_guide=False)
        label_velo = 1
        ax.quiverkey(key, X=0.9, Y=1.05, U=label_velo,
                    label=f"{label_velo} [m/s]", labelpos="E")
        ax.set_aspect("equal")
        ax.set_title(f"t={t:.02f} Days", fontsize=18)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")


def create_animation(longitudinal_mode: int,
                     equatorial_mode: int,
                     wave_mode: int,
                     create_thumbnail: bool = False) -> None:  # noqa: FBT001 FBT002
    """Run the model and create an animation from the output."""
    grid = sw.grid.cartesian.Grid(
        shape=(2**RESOLUTION_POWER, 2**RESOLUTION_POWER),
        domain_size=(LENGTH_X, LENGTH_Y),
        periodic_bounds=(True, False),
    )
    mset = sw.ModelSettings(grid, csqr=CSQR, f0=0, beta=BETA).setup()

    # set the correct coriolis parameter
    y_coord = mset.f_coriolis.get_mesh()[1]
    mset.f_coriolis.arr = BETA * (y_coord - LENGTH_Y / 2)

    # Construct the initial conditions
    z = sw.initial_conditions.EquatorialWave(mset,
                                            longitudinal_mode=longitudinal_mode,
                                            equatorial_mode=equatorial_mode,
                                            wave_mode=wave_mode)


    mset.time_stepper.dt = 0.1 * np.min(grid.dx) / np.sqrt(CSQR)
    mset.tendencies.advection.disable()

    f_name = f"equatorial_lon{longitudinal_mode}_equ{equatorial_mode}_wave{wave_mode}"

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(sw.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=z.period/5,
            filename=f"{f_name}.mp4",
            fps=fps,
            parallel=False,
        ))

    # create a thumbnail saver
    if create_thumbnail:
        mset.diagnostics.add_module(sw.modules.FigureSaver(
            filename=thumbnail, model_time=0, plotter=Plotter))

    # add a netcdf output
    if make_netcdf:
        mset.diagnostics.add_module(sw.modules.NetCDFWriter(
            write_trigger=sw.ClockTrigger(time_interval=z.period/20),
            filename=f"{f_name}.cdf",
        ))

    model = sw.Model(mset)
    model.z = z
    model.run(runlen=z.period)

@sw.utils.skip_on_doc_build
def main() -> None:
    """Run the different setups."""
    # First the 0th, 1st and 2nd equatorial wave mode
    create_animation(2, 0, 2)
    create_animation(2, 1, 2, create_thumbnail=True)
    create_animation(2, 2, 2)

    # Now the 0th, 1st and 2nd equatorial rossby modes
    create_animation(2, 0, 1)
    create_animation(2, 1, 1)
    create_animation(2, 2, 1)

if __name__ == "__main__":
    main()
