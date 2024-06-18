from fridom.nonhydro.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class HarmonicMixing(Module):
    """
    This class computes the harmonic mixing tendency of the model.

    Computes:
    $ dz.b += kh \\nabla^2 b + kv \\partial_z^2 b $
    where:
    - `kh`: Horizontal harmonic mixing coefficient.
    - `kv`: Vertical harmonic mixing coefficient.
    """
    def __init__(self, kh: float = 0, kv: float = 0):
        """
        ## Arguments:
        - `kh`: Horizontal harmonic mixing coefficient.
        - `kv`: Vertical harmonic mixing coefficient.
        """
        super().__init__(name="Harmonic Mixing", kh=kh, kv=kv)

    @start_module
    def start(self):
        # cast the parameters to the correct data type
        self.kh = self.mset.dtype(self.kh)
        self.kv = self.mset.dtype(self.kv)
        return

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the harmonic mixing tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        # compute the harmonic friction tendency
        b = mz.z.b
        kh = self.kh; kv = self.kv; 

        # [TODO] boundary conditions
        dz.b += (b.diff_2(0) + b.diff_2(1))*kh + b.diff_2(2)*kv
        return 

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    kh: {self.kh}\n    kv: {self.kv}"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module