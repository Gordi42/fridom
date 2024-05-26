from fridom.nonhydro.state import State
from fridom.nonhydro.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class HarmonicFriction(Module):
    """
    This class computes the harmonic friction tendency of the model.

    Computes:
    $ dz.u += ah \\nabla^2 u + kh \\partial_z^2 u $
    $ dz.v += ah \\nabla^2 v + kh \\partial_z^2 v $
    $ dz.w += ah \\nabla^2 w + kh \\partial_z^2 w $
    where:
    - `ah`: Horizontal harmonic friction coefficient.
    - `av`: Vertical harmonic friction coefficient.
    """
    def __init__(self, ah: float = 0, av: float = 0):
        """
        ## Arguments:
        - `ah`: Horizontal harmonic friction coefficient.
        - `av`: Vertical harmonic friction coefficient.
        """
        super().__init__(name="Harmonic Friction", ah=ah, av=av)

    @start_module
    def start(self):
        # cast the parameters to the correct data type
        self.ah = self.mset.dtype(self.ah)
        self.av = self.mset.dtype(self.av)
        return

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the harmonic friction tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        # compute the harmonic friction tendency
        u = mz.z.u; v = mz.z.v; w = mz.z.w
        ah = self.ah; av = self.av; 

        # [TODO] boundary conditions
        dz.u += (u.diff_2(0) + u.diff_2(1))*ah + u.diff_2(2)*av
        dz.v += (v.diff_2(0) + v.diff_2(1))*ah + v.diff_2(2)*av
        dz.w += (w.diff_2(0) + w.diff_2(1))*ah + w.diff_2(2)*av
        return 

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    ah: {self.ah}\n    av: {self.av}"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module