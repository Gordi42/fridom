Installation
============

FRIDOM needs Python 3.11 or newer. It is built on
`JAX <https://docs.jax.dev>`_, and every compute array is a
``jax.numpy`` array.

Installing From Source
----------------------

The framework described here lives on the development branch, and the
``fridom`` release on PyPI predates the rewrite, so install from the
repository:

.. code-block:: bash

   git clone https://github.com/Gordi42/FRIDOM
   cd FRIDOM
   pip install -e .            # CPU
   pip install -e '.[cuda]'    # NVIDIA GPU (CUDA 12)

The ``-e`` flag installs FRIDOM in editable mode, so changes to the
source code are reflected in the installed package. The ``cuda`` extra
pulls in ``jax[cuda12]``. Without it, JAX runs on the CPU.

Plotting a field with ``field.xr`` needs ``xarray`` and ``matplotlib``,
which are not runtime dependencies of FRIDOM. Install them alongside it
with ``pip install xarray matplotlib``.

.. note::

   Install FRIDOM into a virtual environment to avoid conflicts with
   other packages. With conda, for example:

   .. code-block:: bash

      conda create -y --name fridom python=3.12
      conda activate fridom

Installing With uv
------------------

FRIDOM is developed with `uv <https://docs.astral.sh/uv/>`_ and the
repository carries a lock file, so ``uv sync`` reproduces the pinned
environment in ``.venv``:

.. code-block:: bash

   uv sync                             # runtime dependencies only
   uv sync --extra dev                 # adds pytest, ruff, xarray, matplotlib
   uv sync --extra dev --extra docs    # adds the sphinx toolchain

Commands then run through ``uv run``, for example
``uv run pytest tests/``. The ``cuda`` extra works here too
(``uv sync --extra cuda``).

Installing From PyPI
--------------------

.. warning::

   The ``fridom`` release on PyPI predates the rewrite and does not
   contain the API used in this documentation. Install from source
   instead.

.. code-block:: bash

   pip install fridom
   pip install 'fridom[cuda]'

Choosing the Compute Device
---------------------------

JAX decides whether a model runs on a CPU, a GPU, or a TPU, for
example through the ``JAX_PLATFORMS`` environment variable. FRIDOM has
no setting of its own for this. See
:doc:`advanced/platform_and_precision`.

Installing on Levante (DKRZ)
----------------------------

Do the installation on a GPU node if you plan to run with GPU
acceleration, so that the CUDA wheels match the node. A GPU node is
requested with

.. code-block:: bash

   salloc -p gpu --gpus=1 --account=projectname

where ``projectname`` is the name of your project. Once the node is
assigned, follow the source instructions above.

With FRIDOM installed, continue with :doc:`getting_started`.
