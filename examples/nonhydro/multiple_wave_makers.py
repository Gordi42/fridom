r"""
Multiple Wave Makers
====================

Multiple internal waves with different angles

In this example we add three polarized wave makers to the model. Each wave maker
has a different angle and is located at a different position in the x-direction.
The phase and group velocity of the waves depend on the angle of the wave vector,
and hence differ between the different wave makers.

.. video:: videos/multiple_wave_makers.mp4
"""
import fridom.nonhydro as nh
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------
make_video  = True
fps         = 30
make_netcdf = False
resolution  = 1024                       # number of grid points in x,z
wave_length = 5                          # wave length in meters
wave_angles = [65, 45, 25]               # angle(kx, kz) in degrees
run_length  = np.timedelta64(12, 'h')    # simulation run length

exp_name    = "multiple_wave_makers"
thumbnail   = f"figures/{exp_name}.png"

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        return plt.figure(figsize=(12.8, 7.2), dpi=200)

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

# ----------------------------------------------------------------
#  The main model
# ----------------------------------------------------------------
@nh.utils.skip_on_doc_build
def main():

    grid = nh.grid.cartesian.Grid(
        N=(resolution, 1, resolution), L=(300, 1, 200), 
        periodic_bounds=(True, True, True))
    mset = nh.ModelSettings(grid=grid, f0=1e-4, N2=2.5e-5)
    mset.time_stepper.dt = np.timedelta64(30, 's')
    mset.tendencies.advection.disable()

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter, 
            model_time_per_second=np.timedelta64(1, "h"),
            filename=exp_name, fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables = lambda mz: mz.z.field_list + [mz.z.etot, mz.z.ekin],
            write_interval = np.timedelta64(20, 'm'),
            filename=exp_name))

    # Add 3 wave makers with different angles
    for angle, x in zip(wave_angles, [25, 125, 225]):
        # convert the angle to actual wave vector components
        kx = 2 * np.pi / wave_length * np.cos(np.deg2rad(angle))
        kz = 2 * np.pi / wave_length * np.sin(np.deg2rad(angle))

        # convert the wave vector to wavenumber of the grid (k = 2pi/L * kp)
        wavenum_x = int(kx * grid.L[0] / (2 * np.pi))
        wavenum_z = int(kz * grid.L[2] / (2 * np.pi))

        mset.tendencies.add_module(nh.modules.forcings.PolarizedWaveMaker(
            position = (x, None, 150),
            width = (10, None, 10),
            k = (wavenum_x, 0, wavenum_z),
            amplitude=20.0))

    mset.setup()
    model = nh.Model(mset)
    model.run(runlen=run_length)

    # plot the final state (thumbnail)
    import os
    os.makedirs("figures", exist_ok=True)
    fig = Plotter(model.model_state)
    fig.savefig(thumbnail, dpi=200)
    return


if __name__ == "__main__":
    main()
