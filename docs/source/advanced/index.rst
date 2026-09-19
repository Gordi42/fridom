===============
Advanced Topics
===============

Self-contained chapters on the machinery behind FRIDOM's models and on
running them. They need not be read in order. Each opens with its
prerequisites.

.. toctree::
   :hidden:

   adiabatic_ramping
   platform_and_precision
   benchmarking

.. grid:: 1 2 2 2
   :margin: 4 4 0 0
   :gutter: 2

   .. grid-item-card:: Adiabatic Ramping
      :link: adiabatic_ramping
      :link-type: doc

      Deforming a model between two operator configurations: the ramp
      legs, staggered protocols, the adiabatic projector, and the
      relative imbalance.

   .. grid-item-card:: Platform and Precision
      :link: platform_and_precision
      :link-type: doc

      Selecting the compute device and the floating point precision
      through JAX.

   .. grid-item-card:: Benchmarking
      :link: benchmarking
      :link-type: doc

      Measuring wall times, compile times, and memory, and comparing
      two runs.
