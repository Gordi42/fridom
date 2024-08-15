r"""
Reflecting Wave Package
=======================

A polarized internal wave package reflecting on the bottom.

This example shows the :py:class:`WavePackage <fridom.nonhydro.initial_conditions.WavePackage>`
initial condition.

.. video:: videos/wave_package.mp4
"""
import fridom.nonhydro as nh
import numpy as np

# ----------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------
make_video  = True
fps         = 60
make_netcdf = False
exp_name    = "wave_package"
thumbnail   = f"figures/{exp_name}.png"

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------

class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(8, 4.5), dpi=256, tight_layout=True)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        return {"b": mz.z.b.xr, "t": mz.time}

    def update_figure(fig, b, t) -> None:
        ax = fig.add_subplot(111)
        b.plot(ax=ax, cmap="RdBu_r", extend='both', vmax=7e-5, vmin=-7e-5)
        ax.set_aspect('equal')
        ax.set_title(f"t = {nh.utils.humanize_number(t, 'seconds')}", fontsize=16)

# ----------------------------------------------------------------
#  The main model
# ----------------------------------------------------------------
@nh.utils.skip_on_doc_build
def main():
    grid = nh.grid.cartesian.Grid(
        N=(1024, 1, 512), L=(8000, 1, 4000), periodic_bounds=(True, True, False))
    mset = nh.ModelSettings(grid=grid, f0=1e-4, N2=2.5e-5)
    mset.time_stepper.dt = np.timedelta64(20, 's')

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=np.timedelta64(4, "h"),
            filename=f"{exp_name}", fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            write_interval = np.timedelta64(1, 'm'),
            filename=exp_name))

    mset.setup()


    # *********************************************************************
    #  Create the initial condition
    # *********************************************************************
    # For the correct fourier transform, we need triple periodic boundaries
    grid_periodic = nh.grid.cartesian.Grid(
        N=(1024, 1, 512), L=(8000, 1, 4000), periodic_bounds=(True, True, True))
    mset_periodic = nh.ModelSettings(grid=grid_periodic, f0=1e-4, N2=2.5e-5)
    mset_periodic.setup()
    # Create the initial conditions from the periodic settings
    z = nh.initial_conditions.WavePackage(
        mset_periodic, 
        mask_pos=(1000, None, 2000), 
        mask_width=(400, None, 400), 
        k=(60, 0, 30))


    # create model and set the initial conditions
    model = nh.Model(mset)
    model.z = z * 1e2

    # plot the initial state (thumbnail)
    import os
    os.makedirs("figures", exist_ok=True)
    fig = Plotter(model.model_state)
    fig.savefig(thumbnail)

    # run the model
    model.run(runlen=np.timedelta64(1, 'D'))


if __name__ == "__main__":
    main()