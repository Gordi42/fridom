"""formatting.py - Utilities for formatting data."""
import datetime

def humanize_length(value: float):
    """
    Format a length in human readable format [mm, cm, m, km].

    Parameters
    ----------
    `value` : `float`
        The length to format.

    Returns
    -------
    `str`
        The formatted length.
    """
    if value < 1e-2:
        return f"{value*1e3:.2f} mm"
    if value < 1:
        return f"{value*1e2:.2f} cm"
    if value < 1e3:
        return f"{value:.2f} m"
    return f"{value/1e3:.2f} km"

def humanize_time(value: float):
    """
    Format a time in human readable format.

    Parameters
    ----------
    `value` : `float`
        The time to format.

    Returns
    -------
    `str`
        The formatted time.
    """
    delta = datetime.timedelta(seconds=float(value))
    days = delta.days
    formatted_time = ""
    if days > 0:
        years, days = divmod(days, 365)
        if years > 0:
            formatted_time += f"{years}y "
        if days > 0:
            formatted_time += f"{days}d "

    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    microseconds = delta.microseconds % 1000

    if hours > 0 or days > 0:
        formatted_time += f"{hours:02d}:"
    if minutes > 0 or hours > 0 or days > 0:
        formatted_time += f"{minutes:02d}:"
    if seconds > 0 or minutes > 0 or hours > 0 or days > 0:
        formatted_time += f"{seconds:02d}s "
    if milliseconds > 0 or microseconds > 0:
        formatted_time += f"{milliseconds}"
        if microseconds > 0:
            formatted_time += f".{microseconds}"
        formatted_time += "ms"
    return formatted_time.strip()

def humanize_number(value: float, unit: str):
    """
    Format a number in human readable format.

    Description
    -----------
    This function formats a number in human readable format. The number is
    converted to a string with the appropriate unit (e.g., meters, seconds).

    Parameters
    ----------
    `value` : `float`
        The number to format.
    `unit` : `str`
        The unit of the number (e.g., meters, seconds).

    Returns
    -------
    `str`
        The formatted number with the appropriate unit.
    """
    if unit == "meters":
        return humanize_length(value)

    if unit == "seconds":
        return humanize_time(value)

    raise NotImplementedError(f"Unit '{unit}' not implemented.")
