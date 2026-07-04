Saving model output
===================

.. include:: fridom_api_names.rst

In FRIDOM, the default behavior of a model run is to produce no output. 
This design choice was made to enhance the framework's flexibility, 
allowing users to tailor output based on the purpose of their model run. 
Depending on the scenario, output requirements can vary significantly:

- You might want to save all scalar fields at regular intervals for later analysis.
- Alternatively, you could only be interested in saving the surface kinetic energy.
- In other cases, you may wish to integrate a custom analysis tool that processes the model state directly during the run.

These requirements fall under the category of diagnostic modules. 
FRIDOM provides built-in diagnostic modules suitable for most use cases, 
but it also supports custom modules tailored to specific needs.

In this tutorial, we will explore the built-in |NetCDFWriter| module. 
In the next tutorial, we will delve into creating custom diagnostic modules.

Using the NetCDFWriter Module
-----------------------------

The |NetCDFWriter| module writes scalar fields to NetCDF files. 
Here is a simple example where all scalar fields of the state vector are saved 
every 0.5 model seconds into a NetCDF file:

.. code-block:: python
    :caption: Add a NetCDF writer to the diagnostics

    import xarray as xr
    import fridom.shallowwater as sw

    # Create the grid and model settings
    grid = sw.grid.cartesian.Grid(shape=(256,256), domain_size=(1,1), periodic_bounds=(True, True))
    mset = sw.ModelSettings(grid=grid, f0=1, csqr=1)
    mset.time_stepper.dt = 0.7e-3

    # Create the netCDF writer and add it to the diagnostics
    writer = sw.modules.NetCDFWriter(
        write_trigger=sw.ClockTrigger(time_interval=0.5),
        filename="output.cdf")
    mset.diagnostics.add_module(writer)

    # Setup the model settings
    mset.setup()

    # Create the initial condition
    z = sw.initial_conditions.Jet(mset, width=0.1, wavenum=2, waveamp=0.05)

    # Create the model and run it
    model = sw.Model(mset)
    model.z = z  # set the initial condition
    model.run(runlen=3)

    # load the output into an xarray dataset and print it
    ds = xr.open_dataset("snapshots/output_0s.cdf")
    print(ds)

.. dropdown:: Output
    :chevron: down-up

    ::

        <xarray.Dataset> Size: 11MB
        Dimensions:  (x: 256, y: 256, time: 7)
        Coordinates:
        * x        (x) float64 2kB 0.001953 0.005859 0.009766 ... 0.9902 0.9941 0.998
        * y        (y) float64 2kB 0.001953 0.005859 0.009766 ... 0.9902 0.9941 0.998
        * time     (time) timedelta64[ns] 56B 00:00:00 ... 00:00:03.000200
        Data variables:
            u        (time, y, x) float64 4MB ...
            v        (time, y, x) float64 4MB ...
            p        (time, y, x) float64 4MB ...
        Attributes:
            description:  fridom: ShallowWater
            created:      Thu Dec 12 09:27:29 2024

By default, files are saved in the `snapshots` directory, with a timestamp 
appended to the filename. You can change the target directory using the 
`directory` argument in the |NetCDFWriter| constructor. To disable the timestamp, 
use the `add_timestamp` property:

.. code-block:: python
    :caption: Disable the timestamp in the filename

    import fridom.framework as fr

    writer = fr.modules.NetCDFWriter(...)
    writer.add_timestamp = False

Using Clock Triggers
~~~~~~~~~~~~~~~~~~~~

In the above example, the |NetCDFWriter| module was initialized with a
|ClockTrigger|. The |ClockTrigger| controls when output is written to the NetCDF
file. Besides the `time_interval` argument, you can also specify a `start_date`
and `stop_date` to define the time range for output:

.. tab-set::

    .. tab-item:: float

        .. code-block:: python

            import numpy as np
            import fridom.framework as fr

            clock_trigger = fr.ClockTrigger(start_date=10.0,
                                            time_interval=1.0,
                                            stop_date=30.0)
            print(clock_trigger)

        .. dropdown:: Output
            :chevron: down-up

            ::

                ClockTrigger(start_date=10.0, time_interval=1.0, stop_date=30.0)

    .. tab-item:: datetime

        .. code-block:: python

            import numpy as np
            import fridom.framework as fr

            clock_trigger = fr.ClockTrigger(start_date=np.datetime64("2020-01-01"),
                                            time_interval=np.timedelta64(1, "D"),
                                            stop_date=np.datetime64("2020-01-30"))
            print(clock_trigger)

        .. dropdown:: Output
            :chevron: down-up

            ::

                ClockTrigger(start_date=2020-01-01, time_interval=86400.0, stop_date=2020-01-30)

    .. tab-item:: step number

        .. code-block:: python

            import numpy as np
            import fridom.framework as fr

            clock_trigger = fr.ClockTrigger(start_step=128,
                                            step_size=64,
                                            stop_step=512)
            print(clock_trigger)

        .. dropdown:: Output
            :chevron: down-up

            ::

                ClockTrigger(start_step=128, step_size=64, stop_step=512)


As shown, the |ClockTrigger| can be initialized using float values 
(in model seconds), datetime values, or step numbers. The arguments specify:

- `start_date` / `start_step`: The time or step when output starts.
- `time_interval` / `step_size`: The interval or step size for writing output.
- `stop_date` / `stop_step`: The time or step when output stops.

When initializing the |NetCDFWriter| module, the |ClockTrigger| is passed as 
the `write_trigger` argument. Additionally, a `restart_trigger` can be specified
to control when new files are created. For example, you can write to the file 
daily but create a new file every model year:

.. code-block:: python
    :caption: Create a new file every model year

    import numpy as np
    import fridom.framework as fr

    writer = fr.modules.NetCDFWriter(
        write_trigger=fr.ClockTrigger(time_interval=np.timedelta64(1, "D")),
        restart_trigger=fr.ClockTrigger(time_interval=np.timedelta64(1, "Y")))

Customizing Output Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, all scalar fields of the state vector are saved to the NetCDF file. 
If you want to change this behavior, you can define a custom callback function 
to specify which scalar fields to save. For example, in a nonhydrostatic model, 
you might save kinetic energy, potential vorticity, and pressure:

.. tab-set::

    .. tab-item:: Custom output variables

        .. code-block:: python

            import fridom.nonhydro as nh

            def get_output_fields(model_state: nh.ModelState) -> list[nh.ScalarField]:
                pressure = model_state.z_diag.p
                ekin = model_state.z.ekin
                pot_vort = model_state.z.pot_vort
                return [pressure, ekin, pot_vort]

            writer = nh.modules.NetCDFWriter(
                write_trigger=nh.ClockTrigger(time_interval=0.5),
                get_variables = get_output_fields)
    
    .. tab-item:: Full example

        .. code-block:: python

            import xarray as xr
            import fridom.nonhydro as nh

            # create the custom output writer
            def get_output_fields(model_state: nh.ModelState) -> list[nh.ScalarField]:
                pressure = model_state.z_diag.p
                ekin = model_state.z.ekin
                pot_vort = model_state.z.pot_vort
                return [pressure, ekin, pot_vort]

            writer = nh.modules.NetCDFWriter(
                write_trigger=nh.ClockTrigger(time_interval=0.5),
                get_variables = get_output_fields)

            # Create the grid and model settings
            grid = nh.grid.cartesian.Grid(shape=(256,256,16), domain_size=(1,1,1), periodic_bounds=(True, True, False))
            mset = nh.ModelSettings(grid=grid, f0=1, stratification_n2=1, rossby_number=0.5)
            mset.time_stepper.dt = 1e-3
            mset.diagnostics.add_module(writer)
            mset.setup()

            # Create the initial condition
            z = nh.initial_conditions.BarotropicJet(mset, wavenum=2, jet_width=0.01)

            # Create the model and run it
            model = nh.Model(mset)
            model.z = z  # set the initial condition
            model.run(runlen=2.5)

            # open the dataset and print it
            ds = xr.open_dataset("snapshots/snap_0s.cdf")
            print(ds)

    
        .. dropdown:: Output
            :chevron: down-up

            ::

                <xarray.Dataset> Size: 151MB
                Dimensions:   (x: 256, y: 256, z: 16, time: 6)
                Coordinates:
                  * x         (x) float64 2kB 0.001953 0.005859 0.009766 ... 0.9902 0.9941 0.998
                  * y         (y) float64 2kB 0.001953 0.005859 0.009766 ... 0.9902 0.9941 0.998
                  * z         (z) float64 128B 0.03125 0.09375 0.1562 ... 0.8438 0.9062 0.9688
                  * time      (time) timedelta64[ns] 48B 00:00:00 ... 00:00:02.501000
                Data variables:
                    p         (time, z, y, x) float64 50MB ...
                    ekin      (time, z, y, x) float64 50MB ...
                    pot_vort  (time, z, y, x) float64 50MB ...
                Attributes:
                    description:  fridom: 3D - Nonhydrostatic model
                    created:      Fri Dec 13 15:36:57 2024


Custom Output Region
~~~~~~~~~~~~~~~~~~~~

To save only a specific region of a scalar fields, use the `snap_slice` property. 
For instance, to save only the first 100 points in the x-direction and every 
second point in the y-direction:

.. tab-set::

    .. tab-item:: Custom output region

        .. code-block:: python

            import fridom.framework as fr

            # create the netCDF writer
            writer = fr.modules.NetCDFWriter(...)

            # save only the first 100 points in x and every second point in y
            writer.snap_slice = (slice(100), slice(None, None, 2))



    .. tab-item:: Full example

        .. code-block:: python

            import xarray as xr
            import fridom.shallowwater as sw

            # Create the grid and model settings
            grid = sw.grid.cartesian.Grid(shape=(256,256), domain_size=(1,1), periodic_bounds=(True, True))
            mset = sw.ModelSettings(grid=grid, f0=1, csqr=1)
            mset.time_stepper.dt = 0.7e-3

            # Create the netCDF writer and add it to the diagnostics
            writer = sw.modules.NetCDFWriter(
                write_trigger=sw.ClockTrigger(time_interval=0.5),
                filename="output.cdf")
            writer.snap_slice = (slice(100), slice(None, None, 2))
            mset.diagnostics.add_module(writer)

            # Setup the model settings
            mset.setup()

            # Create the initial condition
            z = sw.initial_conditions.Jet(mset, width=0.1, wavenum=2, waveamp=0.05)

            # Create the model and run it
            model = sw.Model(mset)
            model.z = z  # set the initial condition
            model.run(runlen=3)

            # load the output into an xarray dataset and print it
            ds = xr.open_dataset("snapshots/output_0s.cdf")
            print(ds)

        .. dropdown:: Output
            :chevron: down-up

            ::

                <xarray.Dataset> Size: 2MB
                Dimensions:  (x: 100, y: 128, time: 7)
                Coordinates:
                * x        (x) float64 800B 0.001953 0.005859 0.009766 ... 0.3848 0.3887
                * y        (y) float64 1kB 0.001953 0.009766 0.01758 ... 0.9785 0.9863 0.9941
                * time     (time) timedelta64[ns] 56B 00:00:00 ... 00:00:03.000200
                Data variables:
                    u        (time, y, x) float64 717kB ...
                    v        (time, y, x) float64 717kB ...
                    p        (time, y, x) float64 717kB ...
                Attributes:
                    description:  fridom: ShallowWater
                    created:      Fri Dec 13 15:52:04 2024

.. note::
    The current implementation of the `snap_slice` property is temporary and 
    will be changed in future versions of FRIDOM.

Multiple Output Writers
~~~~~~~~~~~~~~~~~~~~~~~

FRIDOM's modular design allows you to add multiple diagnostic modules. 
For example, you can use multiple |NetCDFWriter| instances to save different 
scalar fields at different intervals into separate files.


Summary
-------

In this tutorial, we explored the use of the |NetCDFWriter| module in FRIDOM to save model output in various configurations. Key takeaways include:

- The |NetCDFWriter| module provides a flexible way to store scalar fields in NetCDF files.
- You can control output timing using |ClockTrigger| with different configurations (e.g., time intervals, steps, or date ranges).
- Output can be customized by specifying variables or slicing regions of interest.
- Multiple |NetCDFWriter| modules can be used simultaneously for complex output needs.

This approach ensures that your diagnostic output aligns with your simulation objectives, offering both flexibility and scalability.
