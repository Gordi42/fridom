import fridom.framework as fr


@fr.utils.jaxify
class ResetTendency(fr.modules.Module):
    """
    A module that resets the tendency of a model state.

    Description
    -----------
    Time steppers may reuse tendency states to avoid unnecessary memory 
    deallocation and reallocation. For this reason, it is important to reset
    the tendency state before updating it. It should always be the first module
    of the tendencies list.
    """
    name = "Reset Tendency"

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.dz = self.set_state_to_zero(mz.dz)
        return mz

    @fr.utils.jaxjit
    def set_state_to_zero(self, dz):
        """
        Set the state to zero.
        """
        dz *= 0
        return dz
