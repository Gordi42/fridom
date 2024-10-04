Using FRIDOM Models
===================

This chapter provides a comprehensive guide on how to work with FRIDOM and utilize its built-in models. We will introduce the foundational concepts step by step. By the end of this chapter, you will be equipped to:

- Visualize and plot field variables,
- Create custom initial conditions and forcings,
- Save model outputs as netCDF files,
- Generate animations,
- Submit simulations to a computing cluster with automated restarts.

Throughout this chapter, we primarily use the 2D shallow water model as our example, as its 2D data is easier to visualize compared to 3D data. However, most of the concepts covered can be seamlessly applied to other models. To illustrate this, we will set up a basic simulation using both the 2D shallow water model and the 3D nonhydrostatic model. While the specific code details will be further explored in subsequent tutorials, our focus here is to showcase the similarities in setup between the two models.

.. tab-set::

   .. tab-item:: 2D Shallow Water Model

      .. code-block:: python

         import fridom.shallowwater as sw

         # Create the grid and model settings
         grid = sw.grid.cartesian.Grid(N=(256,256), L=(1,1), periodic_bounds=(True, True))
         mset = sw.ModelSettings(grid=grid, f0=1, csqr=1)
         mset.time_stepper.dt = 0.7e-3
         mset.setup()

         # Create the initial condition
         z = sw.initial_conditions.Jet(mset, width=0.1, wavenum=2, waveamp=0.05)

         # Create the model and run it
         model = sw.Model(mset)
         model.z = z  # set the initial condition
         model.run(runlen=2.5)

         # Plot the final total energy (kinetic + potential)
         model.z.etot.xr.plot(cmap="RdBu_r")

   .. tab-item:: 3D Nonhydrostatic Model

      .. code-block:: python

         # TODO: insert code snippet for 3D nonhydrostatic model


TODO: discuss the differences between the two models

TODO: List of tutorials in this chapter