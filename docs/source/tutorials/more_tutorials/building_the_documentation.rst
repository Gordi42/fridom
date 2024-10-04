Building the Documentation
==========================

The documentation for FRIDOM is automatically built and deployed to ReadTheDocs whenever a new commit is pushed to the ``main`` branch on GitHub. However, if you wish to build the documentation locally to preview any changes before pushing, you can follow the steps below.

Setting Up a Build Environment
------------------------------

To set up a dedicated environment for building the documentation, execute the following commands:

.. code-block:: bash

    conda create -y --name fridom-docs python=3.11
    conda activate fridom-docs
    python3 -m pip install --upgrade --no-cache-dir setuptools sphinx readthedocs-sphinx-ext
    python3 -m pip install --exists-action=w --no-cache-dir -r requirements.txt

Building the Documentation
--------------------------

Once you are in the root directory of the FRIDOM repository, you can build the documentation using ``make``. Ensure that the ``make`` command is available on your system:

.. code-block:: bash

    cd docs
    make html

.. note::

    If you are using a conda environment, ensure you activate it before running the command above.

The built documentation will be located in the ``docs/build/html`` directory. To view it, open the ``index.html`` file in your web browser.

Quick Build Option
~~~~~~~~~~~~~~~~~~

Generating the API documentation and gallery can be time-consuming. To accelerate the build process, you can use the ``QUICKBUILD`` option:

.. code-block:: bash

    make html QUICKBUILD=true

This will skip the generation of the API documentation and gallery, building only the remaining sections of the documentation.
