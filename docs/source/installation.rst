Installation
============

Using pip
---------

.. warning::

   FRIDOM is in an early development stage and the latest version on PyPI might
   not be the most recent one. It is recommended to install FRIDOM from the
   source code repository (see below).

FRIDOM can be installed using pip:

.. code-block:: bash

   pip install fridom["<backend>"]

The backend can be one of the following:

- ``jax-cuda``: JAX backend for GPU acceleration with CUDA support. (recommended)
- ``jax``: JAX backend without GPU acceleration.
- ``cupy``: CuPy backend with GPU acceleration (fridom is not optimized for CuPy).

For example, to install FRIDOM with the JAX backend and CUDA support, run:

.. code-block:: bash

   pip install fridom["jax-cuda"]

If no backend is specified, only numpy will be available as a backend.


Building from source
--------------------
To install FRIDOM from the source code repository, clone the repository in 
your desired directory and install the package using pip:

.. code-block:: bash

   git clone https://github.com/Gordi42/FRIDOM
   cd FRIDOM
   pip install -e '.[jax-cuda]'

This will install FRIDOM with the JAX backend and CUDA support. To install FRIDOM
with a different backend, replace ``jax-cuda`` with the desired backend (see above).

.. note::

   The ``-e`` flag installs FRIDOM in editable mode, which means that changes 
   to the source code will be reflected in the installed package.

.. note::

   It is recommended to install FRIDOM in a virtual environment to avoid 
   conflicts with other packages. This can for example be done with conda 
   by running the following code before installing FRIDOM:

   .. code-block:: bash

      conda create -y --name fridom python=3.12
      conda activate fridom


Optional dependencies
---------------------

- ``xarray``: FRIDOM has the option to convert data to xarray datasets for easier plotting. An installation guide can be found `here <http://xarray.pydata.org/en/stable/installing.html>`_.
- ``mpi4py``: FRIDOM has the option to run simulations in parallel using MPI (currently only with the numpy backend). An installation guide can be found `here <https://mpi4py.readthedocs.io/en/stable/install.html>`_.


Installation on special systems
-------------------------------

Levante (DKRZ)
~~~~~~~~~~~~~~
If you plan to run FRIDOM on levante at DKRZ with GPU acceleration, make sure to
do the installation on a gpu node. A gpu node can be requested with the following
command:

.. code-block:: bash

   salloc -p gpu --gpus=1 --account=projectname

where ``projectname`` is the name of your project. After you have been assigned a
gpu node, you can install FRIDOM using the above instructions.