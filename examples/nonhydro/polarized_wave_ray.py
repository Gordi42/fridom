r"""
Three polarized wave rays with different angles
===============================================

This example demonstrates how to create a model with three polarized wave rays


.. youtube:: Fotni4P2ZQs
   :width: 100%
"""

import fridom.nonhydro as nh
import numpy as np
import matplotlib.pyplot as plt

# Create a custom plotter for creating a video animation
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        return plt.figure(figsize=(12.8, 7.2))

    def prepare_arguments(mz: nh.ModelState) -> dict:
        return {"b": mz.z.b.xr, "t": mz.time}

    def update_figure(fig, b, t) -> None:
        # convert the time to a human readable format
        time = nh.utils.humanize_number(t, unit="seconds")
        # create the plot
        ax = fig.add_subplot(111)
        plot = b.plot(ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, extend='both')
        # make the plot look nice
        ax.set_aspect('equal')
        ax.set_title(f"Time: {time}", fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        cbar = plot.colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=18)
        return

def main():
    # resolution factor
    fac = 10

    grid = nh.grid.cartesian.Grid(
        N=(2**fac, 1, 2**fac), L=(300, 1, 200), 
        periodic_bounds=(True, True, True))
    mset = nh.ModelSettings(grid=grid, f0=1e-4, N2=2.5e-5)
    mset.time_stepper.dt = np.timedelta64(30, 's')
    mset.tendencies.advection.disable()

    # add a video writer
    mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
        Plotter, 
        write_interval=np.timedelta64(30, 's'), 
        filename="polarized_wave_ray.mp4",
        fps=60))

    # create a NetCDF writer to save the output
    mset.diagnostics.add_module(nh.modules.NetCDFWriter(
        get_variables = lambda mz: mz.z.field_list + [mz.z.etot, mz.z.ekin],
        write_interval = np.timedelta64(20, 'm'),
        filename="polarized_wave_ray.nc"))

    # Add 3 wave makers with different angles
    for kx, x in zip([10, 20, 40], [25, 125, 225]):
        mset.tendencies.add_module(nh.modules.forcings.PolarizedWaveMaker(
            position = (x, None, 50),
            width = (10, None, 10),
            k = (kx, 0, -20),
            amplitude=20.0))

    mset.setup()
    model = nh.Model(mset)
    model.run(runlen=np.timedelta64(8, 'h'))
    return

if __name__ == "__main__":
    main()
