"""Custom exceptions for the framework."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fridom.framework as fr


class TooManyArgumentsError(Exception):

    """Raise when too many arguments are provided."""

    def __init__(self, max_args: int, **provided_args: any) -> None:
        message = (
            "Too many arguments provided. "
            f"Expected at most {max_args}, got {len(provided_args)}: "
            f"{provided_args}."
        )
        super().__init__(message        )

    @staticmethod
    def check(max_args: int, **provided_args: any) -> None:
        """Check if the number of provided arguments is correct."""
        if sum(arg is not None for arg in provided_args.values()) > max_args:
            raise TooManyArgumentsError(max_args, **provided_args)


class PartialDomainError(NotImplementedError):

    """
    Raise when an operation is attempted on a field that is not fully extended.

    Description
    -----------
    Fields in the computational framework are marked with a `topo` attribute,
    which is a tuple of booleans indicating whether the field is extended
    in each spatial direction. For example, a field with
    `topo=(True, True, False)` depends on `x` and `y`, but not on `z`.

    Certain operations require the field to be fully extended in all directions
    (i.e., `topo=(True, True, True)`). If an unsupported operation is attempted
    on a field that is not fully extended, this exception is raised.

    """

    def __init__(self, topo: tuple[bool]) -> None:
        super().__init__(
            f"Operation not available for fields with topo={topo}. "
            "Fields must be fully extended in all directions.")

    @staticmethod
    def check(field: fr.ScalarField) -> None:
        """
        Check if the given field is fully extended.

        Parameters
        ----------
        field : fr.ScalarField
            The field to check.

        Raises
        ------
            PartialDomainError: If `topo` is not fully extended.

        """
        if not all(field.topo):
            raise PartialDomainError(field.topo)


class FieldSpaceError(NotImplementedError):

    """
    Raise when an operation is attempted on a field in the wrong space.

    Description
    -----------
    Fields in the computational framework have an `is_spectral` attribute,
    which is a boolean indicating whether the field is in spectral space.

    Certain operations require the field to be either in spectral space
    (`is_spectral=True`) or physical space (`is_spectral=False`). If an
    unsupported operation is attempted on a field in the wrong state, this
    exception is raised.

    Parameters
    ----------
    is_spectral : bool
        The is_spectral attribute of the field.

    """

    def __init__(self, is_spectral: bool) -> None:
        self.is_spectral = is_spectral
        state = "spectral" if is_spectral else "physical"
        super().__init__(
            f"Operation not available for fields in {state} space. "
            "Ensure the field is in the correct space for the operation.")

    @staticmethod
    def check_if_spectral(field: fr.FieldBase) -> None:
        """
        Check if the given field is in spectral space.

        Parameters
        ----------
        field : fr.FieldBase
            The field to check.

        Raises
        ------
            FieldSpaceError: If `is_spectral` is False.

        """
        if not field.is_spectral:
            raise FieldSpaceError(field.is_spectral)

    @staticmethod
    def check_if_physical(field: fr.FieldBase) -> None:
        """
        Check if the given field is in physical space.

        Parameters
        ----------
        field : fr.FieldBase
            The field to check.

        Raises
        ------
            FieldSpaceError: If `is_spectral` is True.

        """
        if field.is_spectral:
            raise FieldSpaceError(field.is_spectral)


class NotSetUpError(AttributeError):

    """Raise when an operation is attempted before the module is set up."""

    def __init__(self, module_name: str, attribute: str) -> None:
        msg = f"Trying to access attribute '{attribute}' of '{module_name}'."
        msg += " But the module is not set up yet.\n"
        msg += " Call the 'setup' method first."
        super().__init__(msg)

    @staticmethod
    def check(instance: fr.modules.Module | fr.grid.GridBase,
              attribute: str) -> None:
        """
        Check if the module or grid is set up.

        Parameters
        ----------
        instance : Module | GridBase
            The instance to check.
        attribute : str
            The attribute being accessed.

        """
        if not instance.is_setup:
            # get the class name of the instance
            cls_name = instance.__class__.__name__
            raise NotSetUpError(cls_name, attribute)
