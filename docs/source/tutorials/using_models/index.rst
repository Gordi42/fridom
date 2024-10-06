Using FRIDOM Models
===================

.. toctree::
   :hidden:

   grid_and_modelsettings
   field_variable_and_plotting
   state_and_initial_conditions
   running_the_model
   model_output
   understanding_modules
   animations
   time_stepping_schemes
   submit_to_cluster
   parallelization

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

         import fridom.nonhydro as nh

         # Create the grid and model settings
         grid = nh.grid.cartesian.Grid(N=(256,256,16), L=(1,1,1), periodic_bounds=(True, True, False))
         mset = nh.ModelSettings(grid=grid, f0=1, N2=1, Ro=0.5)
         mset.time_stepper.dt = 1e-3
         mset.setup()

         # Create the initial condition
         z = nh.initial_conditions.BarotropicJet(mset, wavenum=2, jet_width=0.01)

         # Create the model and run it
         model = nh.Model(mset)
         model.z = z  # set the initial condition
         model.run(runlen=2.5)

         # Plot the final total energy (kinetic + potential)
         model.z.etot.xrs[:,:,0].plot(cmap="RdBu_r")


Tutorials
---------
.. grid:: 1 2 2 1
   :margin: 4 4 0 0
   :gutter: 2

   .. grid-item-card::  1. The grid and the model settings
      :link: grid_and_modelsettings
      :link-type: doc

      Learn how to create a grid and set up the model settings.

   .. grid-item-card::  2. Field variables and plotting
      :link: field_variable_and_plotting
      :link-type: doc

      Learn how to work with field variables and how to plot them.

   .. grid-item-card::  3. The state vector and initial conditions
      :link: state_and_initial_conditions
      :link-type: doc

      Learn how to work with the state vector and how to create custom initial conditions.

   .. grid-item-card::  4. Running the model
      :link: running_the_model
      :link-type: doc

      Learn how to run the model.

   .. grid-item-card::  5. Saving model output
      :link: model_output
      :link-type: doc

      Learn how to save model output as netCDF files.

   .. grid-item-card::  6. Understanding FRIDOM modules
      :link: understanding_modules
      :link-type: doc

      Learn what FRIDOM modules are and how to use them to create custom components as for example friction or forcing.

   .. grid-item-card::  7. Generating animations
      :link: animations
      :link-type: doc

      Learn how to generate animations of the model outputs.

   .. grid-item-card::  8. Time-stepping schemes
      :link: time_stepping_schemes
      :link-type: doc

      Learn how to choose and modify time-stepping schemes.

   .. grid-item-card::  9. Submitting simulations to a computing cluster
      :link: submit_to_cluster
      :link-type: doc

      Learn how to submit simulations to a computing cluster with automated restarts.

   .. grid-item-card::  10. Parallelizing simulations
      :link: parallelization
      :link-type: doc

      Learn how to parallelize simulations to run them faster (future feature).
   
