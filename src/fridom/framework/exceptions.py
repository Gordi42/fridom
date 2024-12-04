"""exceptions.py - Custom exceptions for the framework."""

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
