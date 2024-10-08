.. fridom documentation master file, created by
   sphinx-quickstart on Tue Jun 18 15:11:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FRIDOM: Framework for Idealized Ocean Models
============================================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   getting_started
   tutorials/index
   auto_examples/index
   fridom_api


.. video:: _static/fridom_bright.mp4
   :autoplay:
   :loop:

About FRIDOM
------------

- **Purpose & Flexibility:** FRIDOM is a powerful and modular framework originally developed for running simulations of idealized ocean models. Thanks to its modular design, it can be used to simulate any model represented by a set of partial differential equations, such as :math:`\partial_t \boldsymbol{z} = \boldsymbol{f}(\boldsymbol{z}, t)`.

- **Minimizing Boilerplate Code:** To streamline the development process, FRIDOM provides base classes for common components like grids, differential and interpolation operators, time-stepping schemes, netCDF output, animations, etc.

- **Easy Model Modifications:** Every component of a model in FRIDOM is fully exchangeable without changing the model's source code. This feature makes FRIDOM an excellent sandbox for testing new ideas and a useful tool for educational purposes.

- **Balancing Flexibility & Usability:** While modular frameworks often compromise user-friendliness for flexibility, FRIDOM strives to be both flexible and easy to use. It offers a high-level :doc:`API <fridom_api>`, comprehensive :doc:`tutorials <tutorials/index>`, and numerous :doc:`examples <auto_examples/index>`.

- **Performance through Python & JAX:** Written in Python for ease of use, FRIDOM overcomes Python's performance limitations by leveraging the Just-In-Time (JIT) compiler from JAX. This approach allows FRIDOM to achieve speeds comparable to compiled languages like Fortran or C, and it can further accelerate simulations by running on GPUs.

.. note::

   FRIDOM is in an early development stage, and as such, it may undergo significant changes.

Navigation
----------

.. grid:: 1 2 2 3
   :margin: 4 4 0 0
   :gutter: 2

   .. grid-item-card::  Installation
      :link: installation
      :link-type: doc

      How to install.

   .. grid-item-card::  Getting Started
      :link: getting_started
      :link-type: doc

      A quick introduction to the framework.

   .. grid-item-card::  Tutorials
      :link: tutorials/index
      :link-type: doc

      A collection of tutorials to get you started.

   .. grid-item-card::  Gallery
      :link: auto_examples/index
      :link-type: doc

      A collection of examples demonstrating FRIDOM's capabilities.

   .. grid-item-card::  API
      :link: fridom_api
      :link-type: doc

      The full API documentation.


Alternatives
------------
FRIDOM draws inspiration from several existing modeling frameworks and tools, which have influenced its design and capabilities. Some notable inspirations include:

.. grid:: 1 2 2 3
   :margin: 4 4 0 0
   :gutter: 2

   .. grid-item-card::  Oceananigans.jl
      :link: https://github.com/CliMA/Oceananigans.jl

      A very powerful ocean model written in Julia with CPU and GPU support. Oceananigans is suitable for both idealized and realistic ocean setups.

   .. grid-item-card::  pyOM2
      :link: https://github.com/ceden/pyOM2

      An ocean model written in Fortran with many available parameterizations and closures.

   .. grid-item-card::  Veros
      :link: https://github.com/team-ocean/veros

      A Python implementation of `pyOM2` that runs on CPUs and GPUs using JAX.

   .. grid-item-card::  ps3D
      :link: https://github.com/ceden/ps3D

      A pseudo-spectral non-hydrostatic incompressible flow solver written in Fortran.

   .. grid-item-card::  Shenfun
      :link: https://github.com/spectralDNS/shenfun

      A Python framework for solving systems of partial differential equations using the spectral Galerkin method.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
