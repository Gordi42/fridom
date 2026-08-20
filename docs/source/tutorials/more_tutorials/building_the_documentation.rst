Building the Documentation
==========================

The documentation for FRIDOM is automatically built and deployed to ReadTheDocs whenever a new commit is pushed to the ``main`` branch on GitHub. However, if you wish to build the documentation locally to preview any changes before pushing, you can follow the steps below.

Setting Up a Build Environment
------------------------------

The documentation toolchain is declared in the ``docs`` extra of
``pyproject.toml``. To provision the uv-managed environment with it, run the
following command from the root directory of the FRIDOM repository:

.. code-block:: bash

    uv sync --extra dev --extra docs

Building the Documentation
--------------------------

Once you are in the root directory of the FRIDOM repository, you can build the documentation using ``make``. Ensure that the ``make`` command is available on your system:

.. code-block:: bash

    cd docs
    uv run make html

The built documentation will be located in the ``docs/build/html`` directory. To view it, open the ``index.html`` file in your web browser.

Quick Build Option
~~~~~~~~~~~~~~~~~~

Generating the API documentation and gallery can be time-consuming. To accelerate the build process, you can use the ``QUICKBUILD`` option:

.. code-block:: bash

    uv run make html QUICKBUILD=true

This will skip the generation of the API documentation and gallery, building only the remaining sections of the documentation.
