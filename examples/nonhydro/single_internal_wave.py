r"""
Single Internal Wave
====================

A polarized single internal wave in a triple periodic domain.

This example shows the :py:class:`SingleWave <fridom.nonhydro.initial_conditions.SingleWave>`
initial condition.

Top view
--------
.. video:: videos/single_internal_wave_top.mp4
    :loop:

Front view
----------
.. video:: videos/single_internal_wave_front.mp4
    :loop:
"""
import fridom.nonhydro as nh
import numpy as np

# ----------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------
make_video  = True
fps         = 30
make_netcdf = False
exp_name    = "single_internal_wave"
thumbnail   = f"figures/{exp_name}.png"

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class TopPlotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=False)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        skip = 4
        return {"b": mz.z.b.xrs[:,:,-1],
                "p": mz.z_diag.p.xrs[:,:,-1],
                "z": mz.z.xrs[::skip,::skip,-1],
                "t": mz.time}

    def update_figure(fig, b, p, z, t) -> None:
        time = nh.utils.humanize_number(t, unit="seconds")
        ax = fig.add_subplot(211)
        p.plot(ax=ax, cmap="RdBu_r", vmax=0.9, vmin=-0.9, extend='both')
        ax.set_aspect('equal')
        z.plot.quiver("x", "y", "u", "v", ax=ax, scale=700, add_guide=False)

        ax = fig.add_subplot(212)
        b.plot(ax=ax, cmap="RdBu_r", vmax=0.09, vmin=-0.09, extend='both')
        ax.set_aspect('equal')
        z.plot.quiver("x", "y", "u", "v", ax=ax, scale=700, add_guide=False)
        fig.suptitle(f"t = {time}")

class FrontPlotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=False)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        skip = 4
        return {"b": mz.z.b.xrs[:,0,:],
                "p": mz.z_diag.p.xrs[:,0,:],
                "z": mz.z.xrs[::skip,-1,::skip],
                "t": mz.time}

    def update_figure(fig, b, p, z, t) -> None:
        time = nh.utils.humanize_number(t, unit="seconds")
        ax = fig.add_subplot(211)
        p.plot(ax=ax, cmap="RdBu_r", vmax=0.9, vmin=-0.9, extend='both')
        ax.set_aspect('equal')
        z.plot.quiver("x", "z", "u", "w", ax=ax, scale=700, add_guide=False)

        ax = fig.add_subplot(212)
        b.plot(ax=ax, cmap="RdBu_r", vmax=0.09, vmin=-0.09, extend='both')
        ax.set_aspect('equal')
        z.plot.quiver("x", "z", "u", "w", ax=ax, scale=700, add_guide=False)
        fig.suptitle(f"t = {time}")

# ----------------------------------------------------------------
#  The main model
# ----------------------------------------------------------------
@nh.utils.skip_on_doc_build
def main():
    # Create the grid and model settings
    grid = nh.grid.cartesian.Grid(
        N=[64*3, 64, 64], L=[300, 100, 100], periodic_bounds=(True, True, True))
    mset = nh.ModelSettings(grid=grid, f0=1e-4, N2=2.5e-5)
    mset.time_stepper.dt = np.timedelta64(20, 's')
    mset.tendencies.advection.disable()

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            TopPlotter,
            model_time_per_second=np.timedelta64(10, "m"),
            max_jobs=0.05,
            parallel=False,
            filename=f"{exp_name}_top", fps=fps))
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            FrontPlotter,
            model_time_per_second=np.timedelta64(10, "m"),
            max_jobs=0.05,
            parallel=False,
            filename=f"{exp_name}_front", fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            write_interval = np.timedelta64(1, 'm'),
            filename=exp_name))

    mset.setup()

    model = nh.Model(mset)
    # Create the initial conditions
    z = nh.initial_conditions.SingleWave(mset, k=(2, 0, 1)) 
    period = float(z.period)

    # set the initial conditions and run the model
    model.z = z * 2e4
    model.run(runlen=float(period))

    # plot the final state (thumbnail)
    import os
    os.makedirs("figures", exist_ok=True)
    fig = FrontPlotter(model.model_state)
    fig.savefig(thumbnail)


if __name__ == "__main__":
    main()