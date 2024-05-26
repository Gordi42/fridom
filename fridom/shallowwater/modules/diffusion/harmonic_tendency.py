from fridom.shallowwater.state import State
from fridom.shallowwater.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module

class HarmonicTendency(Module):
    """
    This class computes the harmonic friction tendency of the model.
    """
    def __init__(self, ah: float = 0.0, kh: float = 0.0):
        """
        ## Arguments:
        - `kh`: Biharmonic friction coefficient.
        - `kv`: Biharmonic mixing coefficient.
        """
        super().__init__(name="Harmonic Tendency", ah=ah, kh=kh)

    @start_module
    def start(self):
        # cast the parameters to the correct data type
        self.ah = self.mset.dtype(self.ah)
        self.kh = self.mset.dtype(self.kh)

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the harmonic tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        # shorthand notation
        ah = self.ah; kh = self.kh
        u = mz.z.u; v = mz.z.v; h = mz.z.h

        # [TODO] boundary conditions
        dz.u += (u.diff_2(0) + u.diff_2(1))*ah 
        dz.v += (v.diff_2(0) + v.diff_2(1))*ah 
        dz.h += (h.diff_2(0) + h.diff_2(1))*kh 
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    ah: {self.ah}\n    kh: {self.kh}"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module