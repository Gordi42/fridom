Selecting the platform and precision
====================================

FRIDOM is built on `JAX <https://docs.jax.dev>`_, and all arrays are
``jax.numpy`` arrays. Which device the model runs on (CPU, GPU, or TPU)
and which floating point precision is used are both controlled through
JAX directly.

Selecting the platform
----------------------

By default, JAX selects the best available platform (a GPU or TPU if one
is available, otherwise the CPU). To override this choice, set the
``JAX_PLATFORMS`` environment variable before starting python:

.. code-block:: bash

   JAX_PLATFORMS=cpu python my_model.py

or update the JAX configuration at the very top of your script, before
importing fridom:

.. code-block:: python

   import jax
   jax.config.update("jax_platform_name", "cpu")

   import fridom.nonhydro as nh

For more details, see the
`JAX documentation on platforms <https://docs.jax.dev/en/latest/faq.html#controlling-data-and-computation-placement-on-devices>`_.

Selecting the precision
-----------------------

FRIDOM enables double precision (``float64`` / ``complex128``) by default
when it is imported. To run in single precision instead, disable the
``jax_enable_x64`` flag after importing fridom:

.. code-block:: python

   import fridom.nonhydro as nh

   import jax
   jax.config.update("jax_enable_x64", False)

All arrays created afterwards will use ``float32`` / ``complex64``.
