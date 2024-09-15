r"""
Tracers and Eddies
==================

Adding passive tracers to the model.

Experiment 1
------------
A passive tracer forms a spiral pattern when advected by an eddy.
.. video:: videos/tracer_and_eddies_spriale.mp4

Experiment 2
------------
An eddy-dipole collides with a tracer band.
.. video:: videos/tracer_and_eddies_dipole.mp4

"""
import fridom.nonhydro as nh
import os
import matplotlib.pyplot as plt

ncp = nh.config.ncp

# ----------------------------------------------------------------
#  Experiment settings
# ----------------------------------------------------------------
# General settings
make_video  = True
make_netcdf = False
fps         = 30
exp_name    = "tracer_and_eddies"
thumbnail   = f"figures/{exp_name}.png"

# Physical parameters
f0 = 1.0          # Coriolis parameter
N0 = 1.0          # Brunt-Väisälä frequency
Lx = 1.0          # non-dimensional domain size in x and y

# Numerical parameters
periodic = (True, True, True)    # periodic in y and z, non-periodic in x
resolution_factor = 10           # 2^10 = 1024 grid points
Nx = 2**(resolution_factor)      # Number of grid points in x and y


# ----------------------------------------------------------------
#  Create a plotting module for the animation and thumbnail
# ----------------------------------------------------------------
def create_plotter(skip):
    class Plotter(nh.modules.animation.ModelPlotter):
        def create_figure():
            return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=True)

        def prepare_arguments(mz: nh.ModelState) -> dict:
            return {"z": mz.z.xrs[::skip,::skip,0],
                    "tracer": mz.z['dye'].xrs[:,:,0],
                    "t": mz.time}

        def update_figure(fig, z, tracer, t) -> None:
            ax = fig.add_subplot(111)
            tracer.plot(ax=ax, cmap="Blues", vmax=1.0, vmin=0, extend='max')
            key = z.plot.quiver("x", "y", "u", "v", scale=30, add_guide=False, 
                                color="black", width=0.003, headwidth=3)
            label_velo = 2
            ax.quiverkey(key, X=0.9, Y=1.05, U=label_velo,
                        label=f'{label_velo} [m/s]', labelpos='E')
            ax.set_aspect('equal')
            ax.set_title(f't={t:.3f}s', fontsize=18)
    return Plotter

# ----------------------------------------------------------------
#  Create the grid and model settings
# ----------------------------------------------------------------
def create_modelsettings(exp_name, quiver_skip):
    grid = nh.grid.cartesian.Grid(
        L=(Lx, Lx, Lx), N=(Nx, Nx, 1), periodic_bounds=periodic)

    mset = nh.ModelSettings(grid=grid, f0=f0, N2=N0**2)
    mset.time_stepper.dt = 0.4 * 1/Nx

    # add the passive tracer to the state vector
    mset.add_field_to_state({'name':"dye", 
                             'long_name':"Dye concentration",
                             "flags": {'ENABLE_MIXING': True}})

    # ----------------------------------------------------------------
    #  Add custom modules to the model settings
    # ----------------------------------------------------------------
    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
                    create_plotter(quiver_skip),
                    model_time_per_second=0.5,
                    filename=exp_name, fps=30))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables=lambda mz: [mz.z['dye'], mz.z.cfl],
            write_interval = 0.1,
            filename=exp_name))

    # add mixing for the tracer
    # mset.tendencies.add_module(nh.modules.closures.HarmonicMixing(
    #     kh=0.1*(Lx/Nx)**3, kv=0))

    mset.setup()
    return mset

# ================================================================
#  Experiment 1: Spiral pattern
# ================================================================
@nh.utils.skip_on_doc_build
def experiment_1():
    mset = create_modelsettings(exp_name + "_spiral", quiver_skip=40)

    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    z_ini = nh.initial_conditions.CoherentEddy(
        mset=mset, pos_x=0.5, pos_y=0.5, 
        width=0.2, gauss_field="vorticity")
    z_ini *= 1 / ((z_ini.u**2 + z_ini.v**2)**0.5).max()

    # add the passive tracer
    X, Y, Z = z_ini["dye"].get_mesh()
    band_width = 0.05
    # z_ini['dye'] += ( (Y < 0.5 + band_width/2) & (Y > 0.5 - band_width/2) ) * 1.0
    z_ini['dye'] += ncp.exp(-((Y-0.5)/band_width)**2)

    # ----------------------------------------------------------------
    #  Create and run the model
    # ----------------------------------------------------------------
    model = nh.Model(mset)
    model.z = z_ini
    model.run(runlen=1.5)

    # plot the final state (thumbnail)
    os.makedirs("figures", exist_ok=True)
    fig = create_plotter(40)(model.model_state)
    fig.savefig(thumbnail, dpi=200)


# ================================================================
#  Experiment 2: Eddy-dipole
# ================================================================
@nh.utils.skip_on_doc_build
def experiment_2():
    mset = create_modelsettings(exp_name + "_dipole", quiver_skip=20)

    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    L_eddy = 0.05
    def create_dipole(pos_x, pos_y):
        z_eddy = nh.initial_conditions.CoherentEddy(
            mset=mset, pos_x=pos_x-L_eddy/Lx, pos_y=pos_y, 
            width=L_eddy, gauss_field="vorticity")
        z_eddy -= nh.initial_conditions.CoherentEddy(
            mset=mset, pos_x=pos_x+L_eddy/Lx, pos_y=pos_y, 
            width=L_eddy, gauss_field="vorticity")
        z_eddy *= 1 / ((z_eddy.u**2 + z_eddy.v**2)**0.5).max()
        return z_eddy

    z_ini = create_dipole(0.5, 0.3)

    # add the passive tracer
    X, Y, Z = z_ini["dye"].get_mesh()
    band_width = 0.05
    z_ini['dye'] += ncp.exp(-((Y-0.75)/band_width)**2)

    # ----------------------------------------------------------------
    #  Create and run the model
    # ----------------------------------------------------------------
    model = nh.Model(mset)
    model.z = z_ini
    model.run(runlen=5)


if __name__ == "__main__":
    experiment_1()
    experiment_2()
