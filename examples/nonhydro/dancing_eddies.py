r"""
Dancing Eddies
==============

A barotropic setup with four eddies that interact with each other and the walls.

In this setup we initialize two barotropic eddy-dipoles in a setup with
periodic boundaries in the y-direction and non-periodic boundaries in the x-direction.
The dipole on the western side move eastward while the dipole on the western side
move westward. When the two dipoles collide, they form two new eddy-dipoles. One,
that moves northward and one that moves southward. Due to the periodic boundaries
in the y direction, the two dipoles will collide again. After the second collision,
the new formed western dipole will move westward while the new formed eastern dipole
will move eastward. Since the boundaries in the x-direction are non-periodic, the
dipoles will eventually hit the walls. The norther eddy of the dipole will move
northward along the wall, while the southern eddy will move southward along the wall.
Finally, the northward and southward moving eddies will collide and form a new dipole.
This dipole moves into the domain and the whole process starts again.


Physical parameters
-------------------
We use the following scaled parameters:

+-------------+--------------------+------------------------------------------+
| Parameter   | Value              | Description                              |
+=============+====================+==========================================+
| :math:`Ro`  | :math:`0.5`        | Rossby Number                            |
+-------------+--------------------+------------------------------------------+
| :math:`f`   | :math:`1`          | Coriolis parameter                       |
+-------------+--------------------+------------------------------------------+
| :math:`N^2` | :math:`1`          | Vertical background buoyancy gradient    |
+-------------+--------------------+------------------------------------------+
| :math:`L_x` | 1                  | Domain size in x and y                   |
+-------------+--------------------+------------------------------------------+

Numerical parameters
--------------------
We use a cartesian grid with 512x512x1 grid points and a third order Adams-Bashforth
time stepping sheme with a time step size of :math:`\Delta t = 0.01`. 
Furthermore, we use a biharmonic friction closure with a viscosity
of :math:`A_h = 0.01 \cdot U_{\text{eddy}} \cdot Ro \cdot \Delta x^3`, where
:math:`U_{\text{eddy}}` is the maximum velocity of the eddies.

Animation
---------
.. video:: videos/dancing_eddies.mp4

"""
import fridom.nonhydro as nh
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use(['dark_background'])

ncp = nh.config.ncp
PI = ncp.pi

# ----------------------------------------------------------------
#  Experiment settings
# ----------------------------------------------------------------
# General settings
make_video  = True
fps         = 30
exp_name    = "dancing_eddies"
thumbnail   = f"figures/{exp_name}.png"

# Physical parameters
rossby_number = 0.5
f0 = 1.0          # Coriolis parameter
N0 = 1.0          # Brunt-Väisälä frequency
Lx = 1.0          # non-dimensional domain size in x and y
L_eddy = Lx / 2 / 6       # eddy radius
U_eddy = f0 * L_eddy / 2  # eddy velocity

# Numerical parameters
periodic = (False, True, True)   # periodic in y and z, non-periodic in x
resolution_factor = 9            # 2^9 = 512 grid points
Nx = 2**(resolution_factor + 1)  # Number of grid points in x and y


# ----------------------------------------------------------------
#  Create a plotting module for the animation and thumbnail
# ----------------------------------------------------------------
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        return plt.figure(figsize=(6, 4.5), dpi=256, tight_layout=True)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        return {"z": mz.z.xrs[::10,::10,0],
                "zeta": mz.z.rel_vort_z.xrs[:,:,0],
                "t": mz.time}

    def update_figure(fig, z, zeta, t) -> None:
        ax = fig.add_subplot(111)
        colors = ['#ff8b87', "#a10000", 'black', "#0050a1", '#80bfff']
        custom_cmap = LinearSegmentedColormap.from_list('RedBlackBlue', colors)

        zeta.plot(ax=ax, cmap=custom_cmap, vmax=0.8, vmin=-0.8)
        z.plot.quiver("x", "y", "u", "v", scale=2, add_guide=False, 
                      color="#333232", width=0.001, headwidth=5)
        ax.set_aspect('equal')
        ax.set_title(f't={int(t)}s', fontsize=18)


@nh.utils.skip_on_doc_build
def main():
    # ----------------------------------------------------------------
    #  Create the grid and model settings
    # ----------------------------------------------------------------
    grid = nh.grid.cartesian.Grid(
        L=(Lx, Lx, Lx), N=(Nx, Nx, 1), periodic_bounds=periodic)

    mset = nh.ModelSettings(
        grid=grid, f0=f0, N2=N0**2, Ro=rossby_number)
    mset.time_stepper.dt = 0.01
    mset.setup()  # This will calculate the grid spacings

    # ----------------------------------------------------------------
    #  Add custom modules to the model settings
    # ----------------------------------------------------------------
    # add a turbulent closure
    mset.tendencies.add_module(nh.modules.closures.BiharmonicFriction(
        ah = 0.01 * U_eddy * rossby_number * grid.dx[0]**3, av = 0))

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
                    Plotter,
                    model_time_per_second=15/rossby_number,
                    filename=exp_name, fps=30))

    mset.setup()

    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    def create_dipole(pos_x, pos_y):
        z_eddy = nh.initial_conditions.CoherentEddy(
            mset=mset, pos_x=pos_x, pos_y=pos_y-L_eddy/Lx, 
            width=L_eddy, gauss_field="vorticity")
        z_eddy -= nh.initial_conditions.CoherentEddy(
            mset=mset, pos_x=pos_x, pos_y=pos_y+L_eddy/Lx, 
            width=L_eddy, gauss_field="vorticity")
        return z_eddy

    z_ini = create_dipole(0.75, 0.3)
    z_ini -= create_dipole(0.25, 0.3)
    z_ini *= U_eddy / ((z_ini.u**2 + z_ini.v**2)**0.5).max()

    # ----------------------------------------------------------------
    #  Create and run the model
    # ----------------------------------------------------------------
    model = nh.Model(mset)
    model.z = z_ini
    model.run(runlen=286)

    # plot the final state (thumbnail)
    os.makedirs("figures", exist_ok=True)
    fig = Plotter(model.model_state)
    fig.savefig(thumbnail, dpi=200)


if __name__ == "__main__":
    main()
