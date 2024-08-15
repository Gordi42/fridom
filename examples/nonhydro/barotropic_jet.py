r"""
Barotropic Jet
==============

Two opposing barotropic zonal jets with a perturbation on top of it.

This example shows the :py:class:`BarotropicJet <fridom.nonhydro.initial_conditions.BarotropicJet>`
initial condition in a scaled setup (Rossby number = 0.5). 

.. note::
    This example uses the cmocean package for colormaps. You can install it with:

    .. code-block:: bash

        pip install cmocean

.. video:: videos/barotropic_jet.mp4
"""
import fridom.nonhydro as nh

# ----------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------
make_video  = True
fps         = 30
make_netcdf = False
rossby_number = 0.5
wavenumber  = 3                          # wavenumber of the perturbation
run_length  = 5                          # simulation run length
exp_name    = "barotropic_jet"
thumbnail   = f"figures/{exp_name}.png"

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=True)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        # skip every 4th point for the quiver plot
        skip = 4
        return {"z": mz.z.xrs[::skip,::skip,0], 
                "etot": mz.z.etot.xrs[:,:,0], 
                "t": mz.time}

    def update_figure(fig, z, etot, t) -> None:
        import cmocean

        ax = fig.add_subplot(111)
        etot.plot(ax=ax, cmap=cmocean.cm.matter, vmax=2, vmin=0, extend='max')
        key = z.plot.quiver("x", "y", "u", "v", scale=100, add_guide=False)
        label_velo = 2
        ax.quiverkey(key, X=0.9, Y=1.05, U=label_velo,
                    label=f'{label_velo} [m/s]', labelpos='E')
        ax.set_aspect('equal')
        ax.set_title(f't={t:.3f}s', fontsize=18)

# ----------------------------------------------------------------
#  The main model
# ----------------------------------------------------------------
@nh.utils.skip_on_doc_build
def main():
    # Create the grid and model settings
    grid = nh.grid.cartesian.Grid(
        N=(256, 256, 16), L=(1, 1, 1), periodic_bounds=(True, True, True))
    mset = nh.ModelSettings(
        grid=grid, f0=1, N2=1, Ro=0.5)
    mset.time_stepper.dt = 0.002

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=0.5,
            filename=exp_name, fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables = lambda mz: [mz.z.u, mz.z.v, mz.z.ekin],
            write_interval = 0.1,
            filename=exp_name))

    mset.setup()
    model = nh.Model(mset)
    # Create the initial conditions
    z = nh.initial_conditions.BarotropicJet(
        mset, wavenum=wavenumber, waveamp=0.1, geo_proj=True, jet_width=0.01)
    model.z = z 

    # Run the model
    model.run(runlen=run_length)

    # plot the final state (thumbnail)
    import os
    os.makedirs("figures", exist_ok=True)
    fig = Plotter(model.model_state)
    fig.savefig(thumbnail)


if __name__ == "__main__":
    main()