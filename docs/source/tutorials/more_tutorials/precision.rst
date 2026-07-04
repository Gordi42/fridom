Floating point precision
========================

FRIDOM enables double precision (``float64`` / ``complex128``) by default
when it is imported. The precision is controlled through the JAX
``jax_enable_x64`` flag. To run in single precision, disable the flag
after importing fridom:

.. code-block:: python

   import fridom.nonhydro as nh

   import jax
   jax.config.update("jax_enable_x64", False)

All arrays created afterwards will use ``float32`` / ``complex64``.
See also :doc:`backend`.
