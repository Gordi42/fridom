import fridom.framework as fr
from abc import abstractmethod


class AdvectionBase(fr.modules.Module):
    name = "Advection Base"
    _background = None
    _disable_nonlinear = False

    @abstractmethod
    def advection(self,
                  velocity: 'tuple[fr.FieldVariable]',
                  quantity: 'fr.FieldVariable') -> 'fr.FieldVariable':
        """
        Advect a quantity using the given velocity field.
        """

    @fr.utils.jaxjit
    def advect_state(self, z: fr.StateBase, dz: fr.StateBase) -> fr.StateBase:
        if self.background is None and self.disable_nonlinear:
            return dz
        if self.disable_nonlinear:
            zf = self.background
        else:
            # Compute the full state vector (including the background)
            if self.background is not None:
                zf = z + self.background
            else:
                zf = z
        # Compute the velocity field
        if self.grid.n_dims == 1:
            velocity = (zf.u,)
        elif self.grid.n_dims == 2:
            velocity = (zf.u, zf.v)
        elif self.grid.n_dims == 3:
            velocity = (zf.u, zf.v, zf.w)

        # get the scaling factor
        scaling = self.mset.nonlinear_scaling

        # calculate the advection term
        for name, quantity in z.fields.items():
            if quantity.flags["NO_ADV"]:
                continue
            dz.fields[name] += scaling * self.advection(velocity, quantity)
        return dz

    @fr.modules.module_method
    def update(self, mz: 'fr.ModelState') -> None:
        mz.dz = self.advect_state(mz.z, mz.dz)
        return mz

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def disable_nonlinear(self):
        """
        Whether to disable advection by the state vector itself.
        
        Advection by the background state is still enabled.
        """
        return self._disable_nonlinear

    @disable_nonlinear.setter
    def disable_nonlinear(self, value):
        self._disable_nonlinear = value

    @property
    def background(self) -> 'fr.StateBase':
        """The background state."""
        return self._background

    @background.setter
    def background(self, value):
        self._background = value