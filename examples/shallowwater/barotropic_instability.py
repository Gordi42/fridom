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
MAKE_VIDEO  = True
FPS         = 30
MAKE_NETCDF = False
EXP_NAME    = "barotropic_instability"
THUMBNAIL   = f"figures/{EXP_NAME}.png"

# Physical parameters
ROSSBY_NUMBER = 1.0
BURGER_NUMBER = 1.0 / 100
F0 = 1.0          # Coriolis parameter
L = 1.0           # 1 m in x and y (scaled domain)

# specific settings for the barotropic jet
JET_WIDTH = L / 20
U_JET = F0 * JET_WIDTH  # so that the local and global Rossby number are the same

# Numerical parameters
RESOLUTION_FACTOR = 9            # 2^9 = 512 grid points
NX = 2**(RESOLUTION_FACTOR)  # Number of grid points in x and y

# ----------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------
class Plotter(sw.modules.animation.ModelPlotter):
    """Custom plotter for the barotropic instability experiment."""
    @staticmethod
    def create_figure():
        """Create a figure with a specific size and resolution."""
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=True)

    @staticmethod
    def prepare_arguments(mz: sw.ModelState) -> dict:
        """Prepare the arguments for the plot function."""
        # skip every 4th point for the quiver plot
        skip = 2**(9-5)
        state = mz.z.xrs[::skip,::skip]
        pot_vort = mz.z.pot_vort.xr
        return {"state": state, "pot_vort": pot_vort, "t": mz.clock.time}

    @staticmethod
    def update_figure(fig, *args, **kwargs):
        """Plot the fields on the figure."""
        # get the keyword arguments
        state = kwargs["state"]
        pot_vort = kwargs["pot_vort"]
        t = kwargs["t"]
        # plot the potential vorticity and the velocity field
        ax = fig.add_subplot(111)
        pot_vort.plot(ax=ax, cmap="RdBu_r", vmax=250, vmin=-50, extend='both')
        key = state.plot.quiver("x", "y", "u", "v", scale=1.5, add_guide=False)
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
    """Run the barotropic instability experiment."""
    # ----------------------------------------------------------------
    #  Create the grid and model settings
    # ----------------------------------------------------------------
    grid = sw.grid.cartesian.Grid(N=(NX,NX), L=(L,L))
    mset = sw.ModelSettings(grid=grid,
                            f0=F0,
                            Ro=ROSSBY_NUMBER,
                            csqr=BURGER_NUMBER)
    mset.time_stepper.dt = 2 / NX

    # ----------------------------------------------------------------
    #  Add custom modules to the model settings
    # ----------------------------------------------------------------
    # add a video writer
    if MAKE_VIDEO:
        mset.diagnostics.add_module(sw.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=20.0,
            filename=EXP_NAME, fps=FPS))

    # create a NetCDF writer to save the output
    if MAKE_NETCDF:
        mset.diagnostics.add_module(sw.modules.NetCDFWriter(
            get_variables = lambda mz: mz.z.field_list + [mz.z.pot_vort],
            write_interval = 1.0,
            filename=EXP_NAME))

    # create a thumbnail saver
    mset.diagnostics.add_module(sw.modules.FigureSaver(
        filename=THUMBNAIL, model_time=40, plotter=Plotter))

    # biharmonic friction as a simple way to dissipate energy at the smallest scales
    dx = L/NX
    viscosity = 0.01 * U_JET * ROSSBY_NUMBER * dx**3
    friction = sw.modules.closures.BiharmonicFriction(ah=viscosity)
    mset.tendencies.add_module(friction)

    mset.setup()

    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    z = U_JET * sw.initial_conditions.Jet(
        mset,
        width=JET_WIDTH,
        wavenum=2,           # wavenumber of the perturbation
        pos=0.5,             # jet is in the middle
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
