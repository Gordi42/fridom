from fridom.framework.modules.netcdf_writer \
    import NetCDFWriter as NetCDFWriterBase
from fridom.framework.model_state import ModelState

class NetCDFWriter(NetCDFWriterBase):
    def __init__(self, 
                 name="NetCDFWriter", 
                 filename="snap.cdf", 
                 snap_interval=100, 
                 snap_slice=None) -> None:
        # define the variable names
        var_names = ["u", "v", "w", "b", "p"]
        var_long_names = ["Velocity u", "Velocity v", "Velocity w", 
                            "Buoyancy b", "Pressure p"]
        var_unit_names = ["m/s", "m/s", "m/s", "m/s^2", "m^2/s^2"]
        # call the base class constructor
        super().__init__(
            name, filename, snap_interval, snap_slice, 
            var_names, var_long_names, var_unit_names)

    def get_variables(self, mz: ModelState):
        return [mz.z.u, mz.z.v, mz.z.w, mz.z.b, mz.z_diag.p]

# remove symbols from the namespace
del NetCDFWriterBase, ModelState