r"""
Rayleigh-Bénard Convection
==========================

A 2D setup with heating from below and cooling from above.

Note that the model parameter are not tuned to be realistic, but to show
the basic features of 2D-Rayleigh-Bénard convection.

video:: videos/rayleigh_bénard_convection.mp4
"""
import fridom.nonhydro as nh
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
#  Experiment settings
# ----------------------------------------------------------------
# General settings
make_video  = True
fps         = 30
make_netcdf = False
run_length  = 10.0
exp_name    = "rayleigh_bénard_convection"
thumbnail   = f"figures/{exp_name}.png"

# Physical parameters
f0 = 0            # No rotation
N2 = 0            # No stratification
Lx = 2            # 2 m in x
Lz = 1            # 1 m in z

# Numerical parameters
resolution_factor = 10           # 2^10 = 1024 grid points
Nx = 2**(resolution_factor + 1)  # Number of grid points in x
Nz = 2**resolution_factor        # Number of grid points in z

# ----------------------------------------------------------------
#  Create a plotting module for the animation and thumbnail
# ----------------------------------------------------------------
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        return plt.figure(figsize=(8, 3.5), dpi=256, tight_layout=True)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        skip = 30
        return {"z": mz.z.xrs[::skip,0,::skip],
                "b": mz.z.b.xrs[:,0,:],
                "t": mz.time}

    def update_figure(fig, z, b, t) -> None:
        ax = fig.add_subplot(111)
        b.plot(vmax=0.75)
        key = z.plot.quiver("x", "z", "u", "w", scale=50, add_guide=False)
        label_velo = 1
        ax.quiverkey(key, X=0.9, Y=1.05, U=label_velo,
                    label=f'{label_velo} [m/s]', labelpos='E')
        ax.set_aspect('equal')
        ax.set_title(f't={t:.1f}s', fontsize=18)

# ----------------------------------------------------------------
#  Main routine
# ----------------------------------------------------------------
@nh.utils.skip_on_doc_build
def main():
    # ----------------------------------------------------------------
    #  Create the grid and model settings
    # ----------------------------------------------------------------
    grid = nh.grid.cartesian.Grid(N=(Nx, 1, Nz), L=(Lx, 1, Lz), 
                                periodic_bounds=(True, True, False))
    mset = nh.ModelSettings(grid=grid, f0=f0, N2=N2)
    mset.time_stepper.dt = 0.2 / Nz

    # ----------------------------------------------------------------
    #  Add custom modules to the model settings
    # ----------------------------------------------------------------
    relax_top = nh.modules.forcings.Relaxation(
        tau=0.5, field_name="b", target=-1, 
        domain_function=lambda mesh: mesh[2] > Lz - 0.02)
    relax_bot = nh.modules.forcings.Relaxation(
        tau=0.5, field_name="b", target=1,
        domain_function=lambda mesh: mesh[2] < 0.02)

    mset.tendencies.add_module(relax_top)
    mset.tendencies.add_module(relax_bot)
    mset.tendencies.add_module(nh.modules.closures.SmagorinskyLilly())

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter, model_time_per_second=0.5, filename=exp_name, fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            write_interval = 0.2, filename=exp_name))
    
    # create a thumbnail saver
    mset.diagnostics.add_module(nh.modules.FigureSaver(
        filename=thumbnail, model_time=4, plotter=Plotter))

    mset.setup()
    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    z = nh.State(mset)
    # add some white noise to the initial condition so that the
    # instabilities are triggered
    z.u.arr += nh.utils.random_array(z.u.arr.shape) * 1e-4

    # ----------------------------------------------------------------
    #  Run the model
    # ----------------------------------------------------------------
    model = nh.Model(mset)
    model.z = z
    model.run(runlen=run_length)

if __name__ == "__main__":
    main()
