from fridom.framework.modules.module import \
    Module, update_module
from fridom.nonhydro.state import State
from fridom.nonhydro.model_state import ModelState

class Diagnostics(Module):
    def __init__(self, 
                 interval = 50,
                 energy_info = True,
                 cfl_info = True,
                 ):
        super().__init__(
            name="Diagnostics",
            interval=interval,
            energy_info=energy_info,
            cfl_info=cfl_info,)

    @update_module
    def update(self, mz: ModelState, dz: State):
        """
        Print diagnostic information.
        """
        # check if it is time to print diagnostic information
        if mz.it % self.interval != 0:
            return
        
        # print diagnostic information
        out = "Diagnostic at t = {:.2f}\n".format(mz.time)
        if self.energy_info:
            out += "MKE = {:.2e},    ".format(mz.z.mean_ekin())
            out += "MPE = {:.2e},    ".format(mz.z.mean_epot())
            out += "MTE = {:.2e}\n".format(mz.z.mean_etot())
        if self.cfl_info:
            out += "hor. CFL = {:.2f},           ".format(mz.z.max_cfl_h())
            out += "vert. CFL = {:.2f}".format(mz.z.max_cfl_v())
        print(out)
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    interval: {}\n".format(self.interval)
        res += "    energy_info: {}\n".format(self.energy_info)
        res += "    cfl_info: {}\n".format(self.cfl_info)
        return res
