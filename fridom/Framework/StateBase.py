from typing import Type

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase
from fridom.Framework.FieldVariable import FieldVariable

class StateBase:
    """
    Base class for the state of the model, contains list of fields.
    
    Methods:
        fft              : Fourier transform of the state.
        project          : Project the state on a (spectral) vector.
        dot              : Dot product of the state with another state.
        norm_l2          : L2 norm of the state.
        norm_of_diff     : Norm of the difference between two states.
    """
    # ======================================================================
    #  STATE CONSTRUCTORS
    # ======================================================================

    def __init__(self, grid:GridBase, 
                 field_list:list, is_spectral=False) -> None:
        """
        Base Constructor.

        Arguments:
            grid (Grid)           : Grid object.
            field_list (list)     : List of FieldVariables.
            is_spectral (bool)    : State is in spectral space. (default: False)
        """
        self.mset = grid.mset
        self.grid = grid
        self.is_spectral = is_spectral
        self.cp = grid.cp
        self.constructor = StateBase
        self.field_list = field_list

    def copy(self) -> Type["StateBase"]:
        """
        Create a copy of the state.
        """
        fields = [field.copy() for field in self.field_list]
        z = self.constructor(self.grid, 
                             field_list=fields, is_spectral=self.is_spectral)
        return z

    def cpu(self) -> Type["StateBase"]:
        """
        Create a copy of the state on the CPU.
        """
        fields_cpu = [field.cpu() for field in self.field_list]
        grid = fields_cpu[0].grid
        z = self.constructor(grid, field_list=fields_cpu,
                                is_spectral=self.is_spectral)
        return z
    
    # ======================================================================
    #  BASIC OPERATIONS
    # ======================================================================
        
    def fft(self) -> Type["StateBase"]:
        """
        Calculate the Fourier transform of the state. (forward and backward)
        """
        fields_fft = [field.fft() for field in self.field_list]
        z = self.constructor(
            self.grid, field_list=fields_fft, 
            is_spectral=not self.is_spectral)
        return z

    def project(self, p_vec:Type["StateBase"], 
                      q_vec:Type["StateBase"]) -> Type["StateBase"]:
        """
        Project the state on a (spectral) vector.
        $ z = q_vec * (z \cdot p_vec) $

        Arguments:
            p_vec (State)  : P-Vector.
            q_vec (State)  : Q-Vector.
        """
        # transform to spectral space if necessary
        was_spectral = self.is_spectral
        if was_spectral:
            z = self
        else:
            z = self.fft()
        # project
        z = q_vec * (z.dot(p_vec))
        # transform back to physical space if necessary
        if not was_spectral:
            z = z.fft()
        return z

    def dot(self, other:Type["StateBase"]) -> FieldVariable:
        """
        Calculate the dot product of the state with another state.

        Arguments:
            other (State)  : Other state (gets complex conjugated).
        """
        my_fields = self.field_list
        other_fields = other.field_list
        result = my_fields[0] * other_fields[0].conj()
        if len(my_fields) > 1:
            for i in range(1, len(my_fields)):
                result += my_fields[i] * other_fields[i].conj()
        
        return result

    def norm_l2(self) -> float:
        """
        Calculate the L2 norm of the state.

        $$ ||z||_2 = \sqrt{ \sum_{i} \int z_i^2 dV } $$
        where $z_i$ are the fields of the state.

        Returns:
            norm (float)  : L2 norm of the state.
        """
        cell_volume = self.cp.prod(self.cp.array(self.mset.dg))
        return self.cp.sqrt(self.cp.sum(self.dot(self)) * cell_volume)

    def norm_of_diff(self, other:Type["StateBase"]) -> float:
        """
        Calculate the norm of the difference between two states.
        $$ 2 \frac{||z - z'||_2}{||z||_2 + ||z'||_2} $$

        Returns:
            norm (float)  : Norm of the difference between two states.
        """
        return 2 * (self - other).norm_l2() / (self.norm_l2() + other.norm_l2())

    # ======================================================================
    #  OPERATOR OVERLOADING
    # ======================================================================

    def __add__(self, other):
        """
        Add two states / fields together.
        """
        if isinstance(other, self.constructor):
            sums = [f1 + f2 for f1, f2 in zip(self.field_list, other.field_list)]
        else:
            sums = [field + other for field in self.field_list]

        z = self.constructor(self.grid, field_list=sums,
                                is_spectral=self.is_spectral)
        
        return z

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract two states / fields.
        """
        if isinstance(other, self.constructor):
            diffs = [f1 - f2 for f1, f2 in zip(self.field_list, other.field_list)]
        else:
            diffs = [field - other for field in self.field_list]

        z = self.constructor(self.grid, field_list=diffs,
                                is_spectral=self.is_spectral)
        return z
    
    def __rsub__(self, other):
        """
        Subtract something from the state.
        """
        diffs = [other - field for field in self.field_list]
        z = self.constructor(self.grid, field_list=diffs,
                                is_spectral=self.is_spectral)
        return z
        
    def __mul__(self, other):
        """
        Multiply two states / fields.
        """
        if isinstance(other, self.constructor):
            prods = [f1 * f2 for f1, f2 in zip(self.field_list, other.field_list)]
        else:
            prods = [field * other for field in self.field_list]

        z = self.constructor(self.grid, field_list=prods,
                                is_spectral=self.is_spectral)
        return z
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Divide two states / fields.
        """
        if isinstance(other, self.constructor):
            quot = [f1 / f2 for f1, f2 in zip(self.field_list, other.field_list)]
        else:
            quot = [field / other for field in self.field_list]

        z = self.constructor(self.grid, field_list=quot,
                                is_spectral=self.is_spectral)
        return z

    def __rtruediv__(self, other):
        """
        Divide something by the state.
        """
        quot = [other / field for field in self.field_list]
        z = self.constructor(self.grid, field_list=quot,
                                is_spectral=self.is_spectral)
        return z
    
    def __pow__(self, other):
        """
        Exponentiate two states / fields.
        """
        if isinstance(other, self.constructor):
            prods = [f1 ** f2 for f1, f2 in zip(self.field_list, other.field_list)]
        else:
            prods = [field ** other for field in self.field_list]

        z = self.constructor(self.grid, field_list=prods,
                                is_spectral=self.is_spectral)
        return z

# remove symbols from namespace
del Type, ModelSettingsBase, GridBase, FieldVariable