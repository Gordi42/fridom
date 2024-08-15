r"""
Convection and Closures
=======================

Testing different momentum closures in a 2D convection problem.

In this example, I test different closures for the momentum equations. The
model setup is a 2D stratified fluid with a buoyancy perturbation in the middle
of the domain. Thies buoyancy perturbation leads to static instability and
convection. When no closure is used, a lot of energy is accumulated at the grid
scale, leading to a very noisy solution:

without any closure
-------------------
.. video:: videos/convection_and_closures_no_closure.mp4

The most straightforward closure is by including harmonic diffusion with 
turbulent viscosities and diffusivities. The following video shows the solution
when using :py:class:`HarmonicDiffusion <fridom.framework.modules.closures.HarmonicDiffusion>`
with a coefficient of :math:`10^{-4}`:

harmonic diffusion
------------------
.. video:: videos/convection_and_closures_harmonic_diffusion.mp4

The solution is much smoother, but there is also an unrealistic amount of energy
dissipation. To minimize the effect of the closure on the larger scales, one can
use a biharmonic diffusion closure instead of a harmonic one. The following video
shows the solution when using 
:py:class:`BiharmonicDiffusion <fridom.framework.modules.closures.BiharmonicDiffusion>`
with a coefficient of :math:`10^{-6}`:

biharmonic diffusion
--------------------
.. video:: videos/convection_and_closures_biharmonic_diffusion.mp4

Finally, there is also the Smagorinsky-Lilly closure implemented in the model,
this is a dynamic closure that adjusts the turbulent viscosities and diffusivities
based on the resolved scales. The following video shows the solution when using
:py:class:`SmagorinskyLilly <fridom.nonhydro.closures.SmagorinskyLilly>`:

smagorinsky-lilly
-----------------
.. video:: videos/convection_and_closures_Smagorinsky-Lilly.mp4

.. note::
    The diffusion coefficients are not fine-tuned. The purpose of this example
    is to show how different closures can be used in the model and how they
    affect the solution.
"""

import fridom.nonhydro as nh
import fridom.framework as fr
import numpy as np

# ----------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------
make_video  = True
fps         = 30
make_netcdf = False
exp_name    = "convection_and_closures"
run_length  = np.timedelta64(1, 'h')
thumbnail   = f"figures/{exp_name}.png"

class Plotter(nh.modules.animation.ModelPlotter):
    def create_figure():
        import matplotlib.pyplot as plt
        return plt.figure(figsize=(6, 4.5), dpi=32*12, tight_layout=True)

    def prepare_arguments(mz: nh.ModelState) -> dict:
        return {"b": mz.z.b.xr, "t": mz.time}

    def update_figure(fig, b, t) -> None:
        import cmocean
        ax = fig.add_subplot(111)
        b.plot(ax=ax, cmap=cmocean.cm.dense_r, extend='both', vmax=3e-4, vmin=-3e-4)
        ax.set_aspect('equal')
        ax.set_title(f"t = {nh.utils.humanize_number(t, 'seconds')}", fontsize=16)

def test_closure(closure: fr.modules.Module):

    grid = nh.grid.cartesian.Grid(
        N=(512, 1, 512), L=(100, 1, 100), periodic_bounds=(True, True, False))
    mset = nh.ModelSettings(grid=grid, f0=0, N2=2.5e-5)
    mset.time_stepper.dt = np.timedelta64(1, 's')

    fname = exp_name + "_" + closure.name

    # add a video writer
    if make_video:
        mset.diagnostics.add_module(nh.modules.animation.VideoWriter(
            Plotter,
            model_time_per_second=np.timedelta64(4, "m"),
            filename=f"{fname}", fps=fps))

    # create a NetCDF writer to save the output
    if make_netcdf:
        mset.diagnostics.add_module(nh.modules.NetCDFWriter(
            write_interval = np.timedelta64(10, 'm'),
            filename=fname))

    # add the closure
    mset.tendencies.add_module(closure)

    # setup all the modules
    mset.setup()

    # create an initial condition
    ncp = nh.config.ncp  # the array backend (numpy, cupy, ...)
    X, Y, Z = grid.X
    z = nh.State(mset)
    z.b.arr = 1e-3 * ncp.exp(-((X-50)**2 + (Z-50)**2)/(10)**2)

    # create the model, set initial conditions, and run
    model = nh.Model(mset)
    model.z = z
    model.run(runlen=run_length)

    # save thumbnail from biharmonic diffusion
    if closure.name == "biharmonic_diffusion":
        import os
        os.makedirs("figures", exist_ok=True)
        fig = Plotter(model.model_state)
        fig.savefig(thumbnail)

@fr.utils.skip_on_doc_build
def main():
    # test without any closure
    dummy_closure = fr.modules.Module()
    dummy_closure.name = "no_closure"
    test_closure(dummy_closure)

    # test with smagorinsky-lilly closure
    test_closure(nh.modules.closures.SmagorinskyLilly())

    # test with harmonic diffusion
    coeff = 1e-4
    friction = nh.modules.closures.HarmonicFriction(ah=coeff, av=coeff)
    mixing = nh.modules.closures.HarmonicMixing(kh=coeff, kv=coeff)
    test_closure(fr.modules.ModuleContainer("harmonic_diffusion", [friction, mixing]))

    # test with biharmonic diffusion
    coeff = 1e-6
    friction = nh.modules.closures.BiharmonicFriction(ah=coeff, av=coeff)
    mixing = nh.modules.closures.BiharmonicMixing(kh=coeff, kv=coeff)
    test_closure(fr.modules.ModuleContainer("biharmonic_diffusion", [friction, mixing]))


if __name__ == "__main__":
    main()
