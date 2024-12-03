"""model_state.py - The base class for model states."""
from functools import partial
import fridom.framework as fr


# pylint: disable=too-many-instance-attributes
@partial(fr.utils.jaxify, dynamic=('_z', '_z_diag', '_dz', '_it', '_clock'))
class ModelState:
    """
    Stores the model state variables and the time information.
    
    Description
    -----------
    The base class for model states. It contains the state vector, the time step
    and the model time. Child classes may add more attributes as for example the
    diagnostic variables needed for the model.
    All model state variables should be stored in this class.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings object.
    `clock` : `Clock`, optional
        The clock object to keep track of the model time.
    """
    def __init__(self,
                 mset: 'fr.ModelSettingsBase',
                 clock: fr.Clock | None = None) -> None:
        self.mset = mset
        self.z = mset.state_constructor()
        self.z_diag = mset.diagnostic_state_constructor()
        self.dz = None
        self.it = 0
        self._clock = clock or fr.Clock()
        # flag to cancel the model run in case something goes wrong
        self.panicked = False

    def reset(self) -> None:
        """Reset the model state."""
        self._z *= 0.0
        self._z_diag *= 0.0
        self._dz = None
        self._it = 0
        self._clock.reset()

    # ================================================================
    #  xarray conversion
    # ================================================================
    @property
    def xr(self):
        """
        Model State as xarray dataset
        """
        return self.xrs[:]

    @property
    def xrs(self):
        """
        Model State of sliced domain as xarray dataset 
        """
        # xarray sometimes takes a long time to load, so we only import it here
        # if it is actually needed
        try:
            import xarray as xr  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "xarray is not installed. Please install it to use this feature."
            ) from e
        def slicer(key):
            ds_z = self.z.xrs[key]
            ds_zd = self.z_diag.xrs[key]
            ds = xr.merge([ds_z, ds_zd])
            return ds
        return fr.utils.SliceableAttribute(slicer)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def z(self) -> 'fr.StateBase':
        """
        The state vector.
        """
        return self._z

    @z.setter
    def z(self, value: 'fr.StateBase') -> None:
        # convert to correct space
        if value.is_spectral != value.grid.spectral_grid:
            value = value.fft()
        self._z = value

    @property
    def z_diag(self) -> 'fr.StateBase':
        """
        The diagnostic state vector.
        """
        return self._z_diag

    @z_diag.setter
    def z_diag(self, value: 'fr.StateBase') -> None:
        # convert to correct space
        if value.is_spectral != value.grid.spectral_grid:
            value = value.fft()
        self._z_diag = value

    @property
    def dz(self) -> 'fr.StateBase':
        """The tendency vector."""
        return self._dz

    @dz.setter
    def dz(self, value: 'fr.StateBase') -> None:
        # convert to correct space
        if value is not None and value.is_spectral != value.grid.spectral_grid:
            value = value.fft()
        self._dz = value

    @property
    def it(self) -> int:
        """The iteration number."""
        return self._it

    @it.setter
    def it(self, value: int) -> None:
        self._it = value

    @property
    def clock(self) -> 'fr.Clock':
        """
        The clock of the model.
        """
        return self._clock

    @property
    def panicked(self) -> bool:
        """Flag to cancel the model run in case something goes wrong."""
        return self._panicked

    @panicked.setter
    def panicked(self, value: bool) -> None:
        self._panicked = value
