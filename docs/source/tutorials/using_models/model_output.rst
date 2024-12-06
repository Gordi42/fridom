Saving model output
===================

.. include:: fridom_api_names.rst

- Standardmäßig gibt ein Modell keine Ausgabe zurück. Diese Entscheidung wurde
- getroffen, um die Flexibilität des Modells zu erhöhen.
- Je nachdem, was der Sinn und Zweck eines Modelllaufs ist, gibt es Unterschiedliche
- Ansprüche an die Ausgabe des Modells. 
- In einen Fall möchte man vielleicht alle Feldvariablen alle x Zeitschritte
- speichern, um sie später zu analysieren. In einem anderen
- Fall möchte man vielleicht nur die kinetische Energie an der Oberfläche
- speichern. In wieder einen anderen Fall möchte man vielleicht ein eigenes 
- Analyse tool schreiben, dass den Modelstate direkt verarbeitet. 
- All diese Anforderungen fallen in die Kateogrie der diagnostischen Module.
- Dabei kann man entweder built-in diagnostische Module verwenden, die für die
- meisten Anwendungsfälle ausreichen, oder man kann eigene diagnostische Module
- schreiben, die speziell auf die eigenen Bedürfnisse zugeschnitten sind.
- In diesem Tutorial werden wir uns mit den built-in Modul |NetCDFWriter| beschäftigen
und im nächsten Tutorial werden wir uns mit dem Schreiben eigener diagnostischer Module
beschäftigen.

- Das Modul |NetCDFWriter| ist ein diagnostisches Modul, das field variables in
NetCDF Dateien schreiben kann.
- Schauen wir uns ein simples Beispiel an, in dem alle Feldvariablen des State
vector alle 0.5 Modell sekunden in eine NetCDF Datei geschrieben werden.

.. code-block:: python
    :caption: Add a NetCDF writer to the diagnostics

    import xarray as xr
    import fridom.shallowwater as sw

    # Create the grid and model settings
    grid = sw.grid.cartesian.Grid(N=(256,256), L=(1,1), periodic_bounds=(True, True))
    mset = sw.ModelSettings(grid=grid, f0=1, csqr=1)
    mset.time_stepper.dt = 0.7e-3

    # Create the netCDF writer and add it to the diagnostics
    writer = sw.modules.NetCDFWriter(write_interval=0.5, filename="output.nc")
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
    ds = xr.open_dataset("snapshots/output_0s.nc")
    print(ds)

.. dropdown:: Output
    :chevron: down-up

    ::

        <xarray.Dataset> Size: 9MB
        Dimensions:  (x: 256, y: 256, time: 6)
        Coordinates:
          * x        (x) float64 2kB 0.001953 0.005859 0.009766 ... 0.9902 0.9941 0.998
          * y        (y) float64 2kB 0.001953 0.005859 0.009766 ... 0.9902 0.9941 0.998
          * time     (time) timedelta64[ns] 48B 00:00:00 ... 00:00:02.502499999
        Data variables:
            u        (time, y, x) float64 3MB ...
            v        (time, y, x) float64 3MB ...
            p        (time, y, x) float64 3MB ...
        Attributes:
            description:  fridom: ShallowWater
            created:      Fri Dec  6 14:59:56 2024

