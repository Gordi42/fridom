from abc import abstractmethod
from typing import TypeVar
try:
    import cupy as cp
except ImportError:
    pass
import numpy as np

from fridom.Framework.ModelSettingsBase import ModelSettingsBase

ndarray = TypeVar('ndarray')


class Periodic:
    """
    Periodic boundary conditions.
    No padding/unpadding required.
    """
    def __init__(self, mset:ModelSettingsBase, axis:int, btype=None) -> None:
        """
        Arguments:
            mset (ModelSettings): Model settings
            axis (int)          : Axis to apply boundary conditions to
            btype               : Not used (for interface compatibility)
        """
        self.mset = mset
        self.axis = axis
        self.cp = cp if mset.gpu else np
        return

    def pad_for_fft(self, p:ndarray) -> ndarray:
        """
        Extend the field to satisfy boundary conditions.
        For periodic boundary conditions, no padding is required.
        """
        return p

    def unpad_from_fft(self, p:ndarray) -> ndarray:
        """
        Get rid of the extended field.
        For periodic boundary conditions, no unpadding is required.
        """
        return p


class Dirichlet(Periodic):
    """
    Dirichlet boundary conditions. (Zero at the boundary)
    """
    def __init__(self, mset:ModelSettingsBase, axis:int, btype:int) -> None:
        """
        Arguments:
            mset (ModelSettings): Model settings
            axis (int)          : Axis to apply boundary conditions to
            btype (int)          : btype of the coordinate:
                                    1: most outside grid point on the boundary
                                    2: most outside grid point inside the domain
        """
        super().__init__(mset, axis)
        dims = range(mset.n_dims)

        N = mset.N
        self.xp = [(N[i],0) if i == axis else (0,0) for i in dims]
        self.sf = [slice(None, N[i]) if i == axis else slice(None) for i in dims]
        self.sb = [slice(N[i], None) if i == axis else slice(None) for i in dims]

        # convert to tuples
        self.xp = tuple(self.xp)
        self.sf = tuple(self.sf)
        self.sb = tuple(self.sb)

        if not mset.periodic_bounds[axis]:
            if btype == 1:
                self.pad_for_fft = self.pad_for_fft_btype_1
            elif btype == 2:
                self.pad_for_fft = self.pad_for_fft_btype_2
            else:
                raise ValueError("Invalid btype of boundary condition")
            self.unpad_from_fft = self._unpad_from_fft
        return

    def pad_for_fft_btype_1(self, p:ndarray) -> ndarray:
        """
        Pad for FFT btype 1 (Dirichlet boundary conditions).
        """
        cp = self.cp
        axis = self.axis

        p_pad = cp.pad(p, self.xp, mode="constant")
        p_pad[self.sf] = -cp.roll(cp.flip(p, axis=axis), 1, axis=axis)

        return p_pad

    def pad_for_fft_btype_2(self, p:ndarray) -> ndarray:
        """
        Pad for FFT btype 2 (Dirichlet boundary conditions).
        """
        cp = self.cp
        axis = self.axis

        p_pad = cp.pad(p, self.xp, mode="constant")
        p_pad[self.sf] = -cp.flip(p, axis=axis)

        return p_pad

    def _unpad_from_fft(self, p:ndarray) -> ndarray:
        """
        Unpadding is the same for FFT btype 1 and 2.
        """
        return p[tuple(self.sb)]


class Neumann(Periodic):
    """
    Neumann boundary conditions. (Zero derivative at the boundary)
    """
    def __init__(self, mset: ModelSettingsBase, axis:int, btype:int):
        """
        Arguments:
            mset (ModelSettings): Model settings
            axis (int)          : Axis to apply boundary conditions to
            btype (int)          : btype of the coordinate:
                                    1: most outside grid point on the boundary
                                    2: most outside grid point inside the domain
        """
        super().__init__(mset, axis)

        N = mset.N
        dims = range(mset.n_dims)

        self.xp = [(N[i],0) if i == axis else (0,0) for i in dims]
        self.sf = [slice(None, N[i]) if i == axis else slice(None) for i in dims]
        self.sb = [slice(N[i], None) if i == axis else slice(None) for i in dims]

        # convert to tuples
        self.xp = tuple(self.xp)
        self.sf = tuple(self.sf)
        self.sb = tuple(self.sb)
        
        if not mset.periodic_bounds[axis]:
            if btype == 1:
                self.pad_for_fft = self.pad_for_fft_btype_1
            elif btype == 2:
                self.pad_for_fft = self.pad_for_fft_btype_2
            else:
                raise ValueError("Invalid btype of boundary condition")
            self.unpad_from_fft = self._unpad_from_fft
        return

    def pad_for_fft_btype_1(self, p:ndarray) -> ndarray:
        """
        Pad for FFT btype 1 (Neumann boundary conditions).
        """
        cp = self.cp
        axis = self.axis

        p_pad = cp.pad(p, self.xp, mode="constant")
        p_pad[self.sf] = cp.roll(cp.flip(p, axis=axis), 1, axis=axis)
        return p_pad

    def pad_for_fft_btype_2(self, p:ndarray) -> ndarray:
        """
        Pad for FFT btype 2 (Dirichlet boundary conditions).
        """
        cp = self.cp
        axis = self.axis

        p_pad = cp.pad(p, self.xp, mode="constant")
        p_pad[self.sf] = cp.flip(p, axis=axis)
        return p_pad

    def _unpad_from_fft(self, p:ndarray) -> ndarray:
        """
        Unpadding is the same for FFT btype 1 and 2.
        """
        return p[self.sb]


class BoundaryConditions:
    """
    Boundary conditions in all directions.
    """
    def __init__(self, bounds:list) -> None:
        """
        Arguments:
            bounds (list): List of boundary conditions in each direction
        """
        self.bounds = bounds
        return
    
    def pad_for_fft(self, p:ndarray) -> ndarray:
        """
        Boundary conditions by extending the field.
        """
        p_pad = p.copy()
        for b in self.bounds:
            p_pad = b.pad_for_fft(p_pad)
        return p_pad

    def unpad_from_fft(self, p_pad:ndarray) -> ndarray:
        """
        Get rid of the extended field.
        """
        p = p_pad.copy()
        for b in self.bounds[::-1]:
            p = b.unpad_from_fft(p)
        return p