r"""
Internal Gravity Wave Maker
===========================

An internal gravity wave maker using a gaussian wave maker.

In this example we use the :py:class:`GaussianWaveMaker <fridom.nonhydro.modules.forcings.GaussianWaveMaker>` 
module to add an internal gravity wave to the model.

.. note::
    This example uses the cmocean package for colormaps. You can install it with:

    .. code-block:: bash

        pip install cmocean

.. video:: videos/internal_wave_maker.mp4
"""
import fridom.nonhydro as nh
import numpy as np

# ----------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------
make_video  = True
fps         = 30
make_netcdf = False
wave_width  = 4                          # width of the wave maker
wave_period = 45 * 60                    # period of the wave maker (in seconds)
run_length  = np.timedelta64(6, 'h')     # simulation run length
exp_name    = "internal_wave_maker"
thumbnail   = f"figures/{exp_name}.png"

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(8, 4.5), dpi=256, tight_layout=True)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        return {"b": mz.z.b.xr, "etot": mz.z.etot.xr, "t": mz.time}

    def update_figure(fig, b, etot, t) -> None:
        import cmocean
        time = nh.utils.humanize_number(t, unit="seconds")

        ax = fig.add_subplot(211)
        b.plot(ax=ax, cmap=cmocean.cm.balance, 
               vmax=7e-6, vmin=-7e-6, extend='both')
        ax.set_aspect('equal')

        ax = fig.add_subplot(212)
        etot.plot(ax=ax, cmap=cmocean.cm.matter, 
                  vmax=1e-6, vmin=0, extend='max')
        ax.set_aspect('equal')
        fig.suptitle(f'Buoancy and Energy:  t={time}', fontsize=18)

# ----------------------------------------------------------------
#  The main model
# ----------------------------------------------------------------
@nh.utils.skip_on_doc_build
def main():
    # Create the grid and model settings
    grid = nh.grid.cartesian.Grid(
        N=(512, 1, 512), 
        L=(800, 1, 200), 
        periodic_bounds=(True, True, False))
    mset = nh.ModelSettings(
        grid=grid, f0=1e-4, N2=2.5e-5)
    mset.time_stepper.dt = np.timedelta64(1, 'm')

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=np.timedelta64(1, "h"),
            filename=exp_name, fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables = lambda mz: [mz.z.etot, mz.z.b],
            write_interval = np.timedelta64(20, 'm'),
            filename=exp_name))

    # add a Gaussian wave maker
    mset.tendencies.add_module(nh.modules.forcings.GaussianWaveMaker(
        position = (400, None, 75),
        width = (wave_width, None, wave_width),
        frequency = 1/(wave_period), 
        amplitude = 1e-5))

    mset.setup()
    model = nh.Model(mset)
    model.run(runlen=run_length)

    # plot the final state (thumbnail)
    import os
    os.makedirs("figures", exist_ok=True)
    fig = Plotter(model.model_state)
    fig.savefig(thumbnail)


if __name__ == "__main__":
    main()