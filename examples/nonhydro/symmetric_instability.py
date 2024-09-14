r"""
Symmetric Instability
=====================

An eady-like vertical shear flow that is unstable to symmetric instabilities.
Model parameters are taken from Stamper and Taylor (2016) [1].

Model equations
---------------
We consider the nonhydrostatic boussinesq equations and assume that all fields
are constant in the y-direction (:math:`\partial_y \phi = 0`). The model equations 
are:

.. math::
    D_t u = f v - \partial_x p
    \quad , \quad
    D_t v = - f u
    \quad , \quad
    D_t w = b - \partial_z p
    \quad , \quad
    D_t b = 0

where :math:`D_t = \partial_t + u \partial_x + w \partial_z` is the material derivative.
We define a background state :math:`(U,V,W,B,P)` that is in thermal wind balance:

.. math::
    U = 0
    \quad , \quad
    V(z) = - \frac{M^2}{f} z
    \quad , \quad
    W = 0
    
    B(x,z) = N^2 z - M^2 x
    \quad , \quad
    P(x,z) = \frac{N^2}{2} z^2 - M^2 x z

where :math:`M^2` is the vertical shear and :math:`N^2` is the horizontal shear.
Inserting :math:`u = U + u'`, etc. into the model equations and dropping the primes yield:

.. math::
    D_t u = f v - \partial_x p
    \quad , \quad
    D_t w = b - \partial_z p

    D_t v = - f u + \frac{M^2}{f} w
    \quad , \quad
    D_t b = M^2 u - N^2 w

The background vertical stratification is already implemented in the nonhydrostatic model,
but we need to add the tendency terms due to the horizontal background shear. In the
code below, this is done with the `BackgroundAdvection` class.

Physical parameters
-------------------
We use the following parameters:

+-------------+--------------------+------------------------------------------+
| Parameter   | Value              | Description                              |
+=============+====================+==========================================+
| :math:`f`   | :math:`10^{-4}`    | Coriolis parameter                       |
+-------------+--------------------+------------------------------------------+
| :math:`M^2` | :math:`10^{-7}`    | Horizontal background buoyancy gradient  |
+-------------+--------------------+------------------------------------------+
| :math:`N^2` | :math:`Ri~M^4/f^2` | Vertical background buoyancy gradient    |
+-------------+--------------------+------------------------------------------+
| :math:`L_x` | 500 m              | Domain size in x                         |
+-------------+--------------------+------------------------------------------+
| :math:`L_z` | 200 m              | Domain size in z                         |
+-------------+--------------------+------------------------------------------+

with the Richardson number :math:`Ri = [0.25, 0.5, 0.75]`.

Numerical parameters
--------------------
We use a triple periodic domain with 1024x1x512 grid points and a RKF45 time stepping
scheme with an adaptive time step size.

Animations
----------

Richardson number :math:`Ri = 0.25`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. video:: videos/symmetric_instability_ri_0.25.mp4

Richardson number :math:`Ri = 0.5`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. video:: videos/symmetric_instability_ri_0.50.mp4

Richardson number :math:`Ri = 0.75`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. video:: videos/symmetric_instability_ri_0.75.mp4


References
----------
.. [1] Stamper, M. A., & Taylor, J. R. (2016). The transition from symmetric to
    baroclinic instability in the Eady model.

"""
import fridom.nonhydro as nh
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.lines as mlines

# ----------------------------------------------------------------
#  Experiment settings
# ----------------------------------------------------------------
# General settings
make_video  = True
fps         = 30
make_netcdf = False
run_length  = np.timedelta64(6, 'D')     # simulation run length
exp_name    = "symmetric_instability"
thumbnail   = f"figures/{exp_name}.png"

# Physical parameters
f0 = 1e-4         # Coriolis parameter
M2 = 1e-7         # Horizontal background density gradient
Lx = 500          # 500 m in x
Lz = 200          # 200 m in z

# Numerical parameters
resolution_factor = 9            # 2^9 = 512 grid points
Nx = 2**(resolution_factor + 1)  # Number of grid points in x
Nz = 2**resolution_factor        # Number of grid points in z


# ----------------------------------------------------------------
#  Create a plotting module for the animation and thumbnail
# ----------------------------------------------------------------
class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        return plt.figure(figsize=(12, 7), dpi=128)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        return {"b": mz.z.b.xrs[:,0,:],
                "z": mz.z.xrs[::10,0,::10],
                "N2": mz.mset.N2,
                "t": mz.time}

    def update_figure(fig, b, z, N2, t) -> None:
        ax = fig.add_subplot(111)
        # plot the buoyancy field in log scale
        b.plot(ax=ax, norm=SymLogNorm(
            linthresh=1e-7, linscale=1, vmax=1e-5, vmin=-1e-5))
        # add a quiver plot to show the velocity field
        Q = z.plot.quiver('x', 'z', 'u', 'w', 
                          scale=0.4, ax=fig.gca(), width=0.001, add_guide=False)
        arrow = ax.quiverkey(Q, 0.83, 1.03, 0.005, label='Velocity: 0.5 cm/s', 
                             labelpos='E', coordinates='axes')
        # add contours for the background density
        X, Z = np.meshgrid(z.x, z.z)
        contours = ax.contour(X, Z, N2*Z-M2*X, colors='black', linestyles="solid")
        line = mlines.Line2D([], [], color='black', label='Background Density')
        ax.legend(handles=[line], loc='upper right')
        time = nh.utils.humanize_number(int(t), "seconds")
        plt.title(f"Time: {time}", fontsize=20)


@nh.utils.skip_on_doc_build
def perform_experiment(richardson_number, make_thumbnail=False):
    # ----------------------------------------------------------------
    #  Create the grid and model settings
    # ----------------------------------------------------------------
    N2 = richardson_number * M2**2 / f0**2
    grid = nh.grid.cartesian.Grid(N=(Nx, 1, Nz), L=(Lx, 1, Lz), 
                                periodic_bounds=(True, True, True))
    time_stepper = nh.time_steppers.RungeKutta(
        method=nh.time_steppers.RKMethods.RKF45)
    mset = nh.ModelSettings(grid=grid, f0=f0, N2=N2, dsqr=1, time_stepper=time_stepper)

    # ----------------------------------------------------------------
    #  Create a tendency module that includes the background state
    # ----------------------------------------------------------------
    @nh.utils.jaxify
    class BackgroundAdvection(nh.modules.Module):
        name = "Background Advection"
        @nh.modules.module_method
        def update(self, mz: nh.ModelState) -> nh.ModelState:
            mz.dz = self.advect(mz.z, mz.dz)
            return mz

        @nh.utils.jaxjit
        def advect(self, z: nh.State, dz: nh.State) -> nh.State:
            interp = self.interp_module.interpolate
            dz.v += interp(z.w, z.v.position) * M2 / f0
            dz.b += interp(z.u, z.b.position) * M2
            return dz

    # ----------------------------------------------------------------
    #  Add custom modules to the model settings
    # ----------------------------------------------------------------
    # add the background state advection and turbulence closure
    mset.tendencies.add_module(BackgroundAdvection())
    mset.tendencies.add_module(nh.modules.closures.SmagorinskyLilly())

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter, 
            model_time_per_second=np.timedelta64(4, 'h'),
            filename=f"{exp_name}_ri_{richardson_number:.2f}.mp4", fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            get_variables = lambda mz: mz.z.field_list + [mz.z.ekin],
            write_interval = np.timedelta64(20, 'm'),
            filename=f"{exp_name}_ri_{richardson_number:.2f}.nc"))

    mset.setup()

    # ----------------------------------------------------------------
    #  Create the initial condition
    # ----------------------------------------------------------------
    z = nh.State(mset)
    z.v.arr += nh.utils.random_array(z.v.arr.shape) * 1e-6

    # ----------------------------------------------------------------
    #  Run the model
    # ----------------------------------------------------------------
    model = nh.Model(mset)
    model.z = z
    model.run(runlen=run_length)

    # plot the final state (thumbnail)
    if make_thumbnail:
        os.makedirs("figures", exist_ok=True)
        fig = Plotter(model.model_state)
        fig.savefig(thumbnail, dpi=200)

if __name__ == "__main__":
    perform_experiment(0.25)
    perform_experiment(0.50, make_thumbnail=True)
    perform_experiment(0.75)