.. fridom documentation master file, created by
   sphinx-quickstart on Tue Jun 18 15:11:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
========


.. video:: _static/fridom_bright.mp4
   :autoplay:
   :loop:

- **Purpose & Flexibility:** FRIDOM is a powerful and modular framework originally developed for running simulations of idealized ocean models. Thanks to its modular design, it can be used to simulate any model represented by a set of partial differential equations, such as :math:`\partial_t \boldsymbol{z} = \boldsymbol{f}(\boldsymbol{z}, t)`.

- **Minimizing Boilerplate Code:** To streamline the development process, FRIDOM provides base classes for common components like grids, differential and interpolation operators, time-stepping schemes, netCDF output, animations, etc.

- **Easy Model Modifications:** Every component of a model in FRIDOM is fully exchangeable without changing the model's source code. This feature makes FRIDOM an excellent sandbox for testing new ideas and a useful tool for educational purposes.

- **Balancing Flexibility & Usability:** While modular frameworks often compromise user-friendliness for flexibility, FRIDOM strives to be both flexible and easy to use. It offers a high-level :doc:`API <fridom_api>`, comprehensive :doc:`tutorials <tutorials/index>`, and numerous :doc:`examples <auto_examples/index>`.

- **Performance through Python & JAX:** Written in Python for ease of use, FRIDOM overcomes Python's performance limitations by leveraging the Just-In-Time (JIT) compiler from JAX. This approach allows FRIDOM to achieve speeds comparable to compiled languages like Fortran or C, and it can further accelerate simulations by running on GPUs.

.. note::

   FRIDOM is in an early development stage, and as such, it may undergo significant changes.

Navigation
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   index
   installation
   getting_started
   tutorials/index
   auto_examples/index
   fridom_api

- :doc:`Installation <installation>` - How to install FRIDOM.
- :doc:`Getting Started <getting_started>` - A quick introduction to FRIDOM.
- :doc:`Tutorials <tutorials/index>` - A collection of tutorials to get you started with FRIDOM.
- :doc:`Gallery <auto_examples/index>` - A collection of examples demonstrating FRIDOM's capabilities.
- :doc:`API <fridom_api>` - The full API documentation.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
