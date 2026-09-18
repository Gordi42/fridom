FRIDOM: Framework for Idealized Ocean Models
============================================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   getting_started
   advanced/index
   auto_examples/index


.. video:: _static/fridom_bright.mp4
   :autoplay:
   :loop:

FRIDOM is a Python framework for building idealized ocean and
geophysical fluid models. You assemble a model from a grid and a set of
modules, and the result runs as a single compiled JAX function on a
CPU, a GPU, or a TPU. FRIDOM is written for research and process
studies, where the equations change from one experiment to the next.

.. note::

   The framework has been rewritten and the documentation has not
   caught up with it yet. The tutorials for the earlier API have been
   removed rather than left in place to mislead, and the pages that
   remain are being rebuilt. Until that is done, the :doc:`Gallery
   <auto_examples/index>` is the best entry point. Every example in it
   is executed when this documentation is built, so it always matches
   the code.

Where to Go Next
----------------

.. grid:: 1 2 2 2
   :margin: 4 4 0 0
   :gutter: 2

   .. grid-item-card::  Installation
      :link: installation
      :link-type: doc

      Installing FRIDOM, with or without GPU support.

   .. grid-item-card::  Getting Started
      :link: getting_started
      :link-type: doc

      One complete shallow-water run, walked through line by line.

   .. grid-item-card::  Advanced Topics
      :link: advanced/index
      :link-type: doc

      Chapters on the machinery behind the models and on running them.

   .. grid-item-card::  Gallery
      :link: auto_examples/index
      :link-type: doc

      Complete experiments in both models, executed when the
      documentation is built.
