Submitting Simulations to a Computing Cluster
============================================

.. include:: fridom_api_names.rst


This tutorial will teach you:

- How to submit a job to a computing cluster running SLURM.
- How to automate restarting simulations for long runs.

All examples here were tested on the `levante` computing cluster at the German Climate Computing Center (DKRZ).

Prerequisites
~~~~~~~~~~~~~

To follow this tutorial, create a new empty directory and navigate into it:

.. code-block:: bash

    mkdir cluster_tutorial
    cd cluster_tutorial

Setting Up the Simulation Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this directory, create a Python script named `main.py` with the following content:

.. code-block:: python
    :caption: main.py

    """An example script to run a shallow water model setup on a computing cluster"""
    import numpy as np

    import fridom.shallowwater as sw

    # Set the log level to INFO for more information about the simulation
    sw.log.setLevel("INFO")

    # Create the grid and model settings
    grid = sw.grid.cartesian.Grid(N=(2048,2048), L=(1,1), periodic_bounds=(True, True))
    mset = sw.ModelSettings(grid=grid, f0=1, csqr=1)
    mset.time_stepper.dt = 0.7e-4

    # Create the netCDF writer and add it to the diagnostics
    writer = sw.modules.NetCDFWriter(
        write_trigger=sw.ClockTrigger(time_interval=0.5),
        filename="output.cdf")
    mset.diagnostics.add_module(writer)

    # Add a restart module for automatic restarts
    restart = sw.modules.RestartModule(
        realtime_interval=np.timedelta64(1, "m"),
    )
    mset.restart_module = restart

    # Setup the model settings
    mset.setup()

    # Create the initial condition
    z = sw.initial_conditions.Jet(mset, width=0.1, wavenum=2, waveamp=0.05)

    # Create the model and run it
    model = sw.Model(mset)
    model.z = z  # set the initial condition
    model.run(runlen=2.5)


Understanding the Setup
~~~~~~~~~~~~~~~~~~~~~~~

Before submitting the job, let’s review the setup and compare it to the example in the
:doc:`quickstart tutorial </getting_started>`.

- The resolution is increased to ensure the model runs longer than just a few seconds.
- The log level is set to `INFO` for more detailed progress updates. (The default log level is `SILENT`; see the :doc:`logging tutorial </tutorials/more_tutorials/logging>` for details.)
- A |NetCDFWriter| is added to the diagnostics to save simulation output
- A |RestartModule| is included to handle automatic restarts, which will be discussed later in this tutorial.

Creating a Job Submission Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To submit the job to the cluster, create a shell script named `run_fridom.sh` with the following content:

.. code-block:: bash
    :caption: run_fridom.sh

    #!/bin/bash

    #SBATCH --job-name=sw_jet
    #SBATCH --output=log/%x_%j.out
    #SBATCH --error=log/%x_%j.err
    #SBATCH --partition=<partition>
    #SBATCH --account=<account>
    #SBATCH --nodes=1
    #SBATCH --gpus=1
    #SBATCH --ntasks=1
    #SBATCH --time=00:10:00

    source activate fridom  # assuming you have a conda environment named fridom

    srun -l python3 main.py

- Replace `<partition>` and `<account>` with the appropriate values for your cluster. For the `levante` cluster at DKRZ, use `gpu` for the partition.
- Adjust the `--time` parameter to reflect the expected runtime of your simulation.
- This example runs the model on a single node with a single GPU. For information on running the model on multiple devices, see the :doc:`parallelization tutorial </tutorials/using_models/parallelization>`.

Submitting the Job
~~~~~~~~~~~~~~~~~~

Make the script executable (only needed once):

.. code-block:: bash

    chmod +x run_fridom.sh

Submit the job to the cluster:

.. code-block:: bash

    sbatch run_fridom.sh

Reviewing the Output
~~~~~~~~~~~~~~~~~~~~

After the jobs finish, inspect the directory structure:

.. code-block:: bash

    ls -l
    drwxr-sr-x 2 u301533 uo0780 4096 Jan 15 18:57 log
    -rw-r--r-- 1 u301533 uo0780 1063 Jan 15 18:47 main.py
    drwxr-sr-x 2 u301533 uo0780 4096 Jan 15 18:57 restart
    -rwxr-xr-x 1 u301533 uo0780  276 Jan 15 18:54 run_fridom.sh
    drwxr-sr-x 2 u301533 uo0780 4096 Jan 15 18:58 snapshots

Three directories are created:

1. **`log`**: Contains the job output.
2. **`snapshots`**: Contains netCDF files written by the netCDF writer.
3. **`restart`**: Contains restart files written by the restart module

We will now discuss the contents of these directories in more detail.


Restart Files
-------------

In the main script, we added a restart module with a parameter `realtime_interval` set to 1 minute. This means that after one minute of runtime, the model will save its state to a restart file, stop the current execution, and submit a new job to the cluster by running `sbatch run_fridom.sh` again. This way, the same main script is executed in a new job.

You may wonder how the model knows to restart from the restart file. This is handled by the restart module. When the method `model.run(...)` is called, the restart module checks the `restart` directory for available restart files. If it finds any, it loads the latest restart file and continues the simulation from the saved state. In this example, the model restarted twice: once after iteration `12668` and once after iteration `24556`.

.. code-block:: bash
   ls -l restart/
   -rw-r--r-- 1 u301533 uo0780 491704033 Jan 15 18:56 model_12668_0.dill
   -rw-r--r-- 1 u301533 uo0780 491704092 Jan 15 18:57 model_24556_0.dill

.. info::
   The `restart` directory and the filenames can be customized by setting the `file_path` parameter in the |RestartModule|. The default path is `restart/model.dill`.

   Avoid including the character `_` in the filename, as it is used to separate the iteration number and the processor rank (e.g., `0` in this case).

If you submit the job again with `sbatch run_fridom.sh`, the model will continue from the last restart file. Make sure to delete the restart files if you wish to start a new simulation from scratch.

.. note::
   The |RestartModule| is optional and should be used for long simulations that might be interrupted by the cluster scheduler.

Log Files
---------

Let’s examine the `log` directory:

.. code-block:: bash

    ls -l log/
    -rw-r--r-- 1 u301533 uo0780       0 Jan 15 18:54 sw_jet_14792351.err
    -rw-r--r-- 1 u301533 uo0780 1460931 Jan 15 18:56 sw_jet_14792351.out
    -rw-r--r-- 1 u301533 uo0780       0 Jan 15 18:56 sw_jet_14792358.err
    -rw-r--r-- 1 u301533 uo0780 1442561 Jan 15 18:57 sw_jet_14792358.out
    -rw-r--r-- 1 u301533 uo0780       0 Jan 15 18:57 sw_jet_14792373.err
    -rw-r--r-- 1 u301533 uo0780 1387574 Jan 15 18:58 sw_jet_14792373.out

The `.out` files contain the job outputs, while the `.err` files capture error messages, which should be empty if everything went smoothly.

Snapshots
---------

In the `main.py` script, we added a |NetCDFWriter| to the diagnostics. This writer saves snapshots of the model state to a netCDF file every 0.5 time units. These files are stored in the `snapshots` directory:

.. code-block:: bash

    ls -l snapshots/
    -rw-r--r-- 1 u301533 uo0780 201385500 Jan 15 18:57 output_01s_0.20ms.cdf
    -rw-r--r-- 1 u301533 uo0780 201385500 Jan 15 18:58 output_02s_0.40ms.cdf
    -rw-r--r-- 1 u301533 uo0780 201385500 Jan 15 18:56 output_0s.cdf

For each restart, a new snapshot file is created. This behavior might be inconvenient in some cases. For example, if you want a single netCDF file per simulation month. If a restart occurs mid-month, you end up with two files for the same month. Ideally, the new data would append to the existing file, but this feature is not yet supported by the netCDF writer.

As a workaround, you can configure the restart module to trigger restarts based on simulation time rather than real-time runtime.


More Advanced Restart Strategies
--------------------------------

Trigger Restart by Simulation Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the previous example, the `realtime_interval` parameter of the restart module was used to trigger restarts based on the model's runtime. Alternatively, restarts can be triggered based on simulation time by using the `clock_trigger` parameter and assigning it a |ClockTrigger| object. For detailed information on the |ClockTrigger| class, refer to the :doc:`output writer tutorial </tutorials/using_models/model_output>`. Below is an example of setting a restart to occur every model year:

.. code-block:: python
    :caption: main.py

    restart = sw.modules.RestartModule(
        clock_trigger=sw.ClockTrigger(time_interval=np.timedelta64(1, "Y")),
    )

Ensure that the time required to simulate one year is less than the time limit of the job to avoid interruptions.

Custom Restart Command
~~~~~~~~~~~~~~~~~~~~~~

By default, the restart command is determined automatically using `scontrol` to retrieve the job submission command. However, this approach may fail in certain scenarios, such as when the submission script requires arguments. For example, consider the following script:

.. code-block:: bash
    :caption: run_fridom.sh

    #!/bin/bash

    # <insert appropriate sbatch options here>

    source activate fridom
    srun -l python3 "$1"

In this case, the restart command should be `sbatch run_fridom.sh main.py`. However, the restart module might submit only `sbatch run_fridom.sh`, causing the restart to fail because the script name is missing.

To address this issue, you can manually specify the `restart_command` parameter of the restart module as follows:

.. code-block:: python
    :caption: main.py

    restart = sw.modules.RestartModule(
        realtime_interval=np.timedelta64(1, "m"),
        restart_command="sbatch run_fridom.sh main.py",
    )

Alternatively, the `restart_command` parameter can be set to a custom function that handles the restart process. For example, using the `subprocess` module:

.. code-block:: python
    :caption: main.py

    import subprocess
    import sys

    def restart_job():
        subprocess.run(["sbatch", "run_fridom.sh", "main.py"])
        sys.exit(0)

    restart = sw.modules.RestartModule(
        realtime_interval=np.timedelta64(1, "m"),
        restart_command=restart_job,
    )

This approach allows for full customization of the restart command, enabling the use of other tools, such as `pyslurm`, to submit jobs efficiently.
