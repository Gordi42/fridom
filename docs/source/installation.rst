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

   pip install fridom

FRIDOM is built on `JAX <https://docs.jax.dev>`_. The default installation
runs on the CPU. To run FRIDOM on a GPU with CUDA support, install the
``cuda`` extra:

.. code-block:: bash

   pip install fridom["cuda"]

The compute device is selected through JAX directly, for example via the
``JAX_PLATFORMS`` environment variable (see :doc:`the platform tutorial
<tutorials/more_tutorials/backend>`).


Building from source
--------------------
To install FRIDOM from the source code repository, clone the repository in 
your desired directory and install the package using pip:

.. code-block:: bash

   git clone https://github.com/Gordi42/FRIDOM
   cd FRIDOM
   pip install -e '.[cuda]'

This will install FRIDOM with CUDA support. For a CPU-only installation,
drop the ``[cuda]`` extra.

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

- ``xarray``: To convert data to xarray datasets for easier plotting. An installation guide can be found `here <http://xarray.pydata.org/en/stable/installing.html>`_.
- ``imageio``: To create animations. It can be installed with ``pip install "imageio[ffmpeg]"``.
- ``mpi4py``: To run simulations in parallel using MPI. An installation guide can be found `at this link <https://mpi4py.readthedocs.io/en/stable/install.html>`_.


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
