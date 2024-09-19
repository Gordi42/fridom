r"""
Barotropic Instability
======================

An instable barotropic jet with a perturbation on top of it.

.. video:: videos/barotropic_instability.mp4
"""
import fridom.shallowwater as sw

# ----------------------------------------------------------------
#  Experiment settings
# ----------------------------------------------------------------
# General settings
make_video  = True
fps         = 30
make_netcdf = False
exp_name    = "barotropic_instability"
thumbnail   = f"figures/{exp_name}.png"

# Physical parameters
rossby_number = 1.0
burger_number = 1.0 / 100
f0 = 1.0          # Coriolis parameter
L = 1.0           # 1 m in x and y (scaled domain)

# specific settings for the barotropic jet
jet_width = L / 20
U_jet = f0 * jet_width  # so that the local and global Rossby number are the same

# Numerical parameters
resolution_factor = 9            # 2^9 = 512 grid points
Nx = 2**(resolution_factor)  # Number of grid points in x and y

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class Plotter(sw.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=True)

    def prepare_arguments(mz: sw.ModelState) -> dict:
        # skip every 4th point for the quiver plot
        skip = 2**(9-5)
        return {"z": mz.z.xrs[::skip,::skip],
                "pot_vort": mz.z.pot_vort.xr,
                "t": mz.time}

    def update_figure(fig, z, pot_vort, t) -> None:
        ax = fig.add_subplot(111)
        pot_vort.plot(ax=ax, cmap="RdBu_r", vmax=170, vmin=30, extend='both')
        key = z.plot.quiver("x", "y", "u", "v", scale=1, add_guide=False)
        label_velo = 0.05
        ax.quiverkey(key, X=0.9, Y=1.05, U=label_velo,
                    label=f'{label_velo} [m/s]', labelpos='E')
        ax.set_aspect('equal')
        ax.set_title(f't={t:.0f}s', fontsize=18)

# ----------------------------------------------------------------
#  The main model
# ----------------------------------------------------------------
@sw.utils.skip_on_doc_build
def main():
    # ----------------------------------------------------------------
    #  Create the grid and model settings
    # ----------------------------------------------------------------
    grid = sw.grid.cartesian.Grid(N=(Nx,Nx), L=(1,1))
    mset = sw.ModelSettings(grid=grid, 
                            f0=1.0,
                            Ro=rossby_number, 
                            csqr=burger_number)
    mset.time_stepper.dt = 2 / Nx

    # ----------------------------------------------------------------
    #  Add custom modules to the model settings
    # ----------------------------------------------------------------
    # add a video writer
    if make_video:
        mset.diagnostics.add_module(sw.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=20.0,
            filename=exp_name, fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(sw.modules.NetCDFWriter(
            get_variables = lambda mz: mz.z.field_list + [mz.z.pot_vort],
            write_interval = 1.0,
            filename=exp_name))

    # create a thumbnail saver
    mset.diagnostics.add_module(sw.modules.FigureSaver(
        filename=thumbnail, model_time=70, plotter=Plotter))
    
    # biharmonic friction as a simple way to dissipate energy at the smallest scales
    dx = L/Nx
    viscosity = 0.01 * U_jet * rossby_number * dx**3
    friction = sw.modules.closures.BiharmonicFriction(ah=viscosity)
    mset.tendencies.add_module(friction)

    mset.setup()

    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    z = U_jet * sw.initial_conditions.Jet(
        mset, 
        jet_width=jet_width, 
        wavenum=2,           # wavenumber of the perturbation
        jet_pos=(0.5, -10),  # one jet is in the middle, the other not in the domain
        waveamp=1e-2)

    # ----------------------------------------------------------------
    #  Run the model
    # ----------------------------------------------------------------
    model = sw.Model(mset)
    model.z = z
    model.run(runlen=200.0)
    return model

if __name__ == "__main__":
    main()