from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, module_method


class Diagnostics(Module):
    name = "Diagnostics"
    def __init__(self,
                 interval = 50,
                 energy_info = True,
                 cfl_info = True,
                 ):
        super().__init__()
        self.interval = interval
        self.energy_info = energy_info
        self.cfl_info = cfl_info

    @module_method
    def update(self, mz: ModelState) -> ModelState:
        """
        Print diagnostic information.
        """
        # check if it is time to print diagnostic information
        if mz.clock.it % self.interval != 0:
            return None

        # print diagnostic information
        out = f"Diagnostic at t = {mz.clock.time:.2f}\n"
        if self.energy_info:
            out += f"MKE = {mz.z.mean_ekin():.2e},    "
            out += f"MPE = {mz.z.mean_epot():.2e},    "
            out += f"MTE = {mz.z.mean_etot():.2e}\n"
        if self.cfl_info:
            out += f"hor. CFL = {mz.z.max_cfl_h():.2f},           "
            out += f"vert. CFL = {mz.z.max_cfl_v():.2f}"
        print(out)
        return mz

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    interval: {self.interval}\n"
        res += f"    energy_info: {self.energy_info}\n"
        res += f"    cfl_info: {self.cfl_info}\n"
        return res
