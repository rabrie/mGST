import numpy as np
from math import ceil, isinf, log10
from numbers import Number
from typing import List, Optional, Tuple, Union

from qibocal.config import raise_error


def significant_digit(number: Number):
    """Computes the position of the first significant digit of a given number.

    Args:
        number (Number)

    Returns:
        int: position of the first significant digit or ``3`` if the given number
        is integer, ``inf`` or ``0``. Returns ``-1`` if ``number`` is ``None``.
    """
    if number is None:
        return -1

    position = (
        ceil(-log10(abs(np.real(number))))
        if not isinf(np.real(number)) and np.real(number) != 0
        else 3
    )
    if np.imag(number) != 0:
        position = max(
            position,
            ceil(-log10(abs(np.imag(number))))
            if not isinf(np.imag(number)) and np.imag(number) != 0
            else 3,
        )
    position = 3 if position < 1 else position
    return position


def number_to_str(
    value: Number,
    uncertainty: Optional[
        Union[Number, List[Number], Tuple[Number], np.ndarray]
    ] = None,
    precision: Optional[int] = None,
):
    """Converts a number into a string.

    Args:
        value (Number): the number to display
        uncertainty (Number or List[Number] or np.ndarray, optional): number or 2-element
        interval with the low and high uncertainties of the ``value``. Defaults to ``None``.
        precision (int, optional): nonnegative number of floating points of the displayed value.
        If ``None``, defaults to the first significant digit of ``uncertainty``
        or ``3`` if ``uncertainty`` is ``None``. Defaults to ``None``.

    Returns:
        str: The number expressed as a string, with the uncertainty if given.
    """
    
    if np.isnan(value):
        return f"--"
    
    if precision is not None:
        if isinstance(precision, int) is False:
            raise_error(
                TypeError,
                f"`precision` must be of type int. Got {type(precision)} instead.",
            )
        if precision < 0:
            raise_error(
                ValueError,
                f"`precision` cannot be negative. Got {precision}.",
            )

    # If uncertainty is not given, return the value with precision
    if uncertainty is None:
        precision = precision if precision is not None else 3
        return f"{value:.{precision}f}"

    if isinstance(uncertainty, Number):
        if precision is None:
            precision = significant_digit(uncertainty)
        return f"{value:.{precision}f} \u00B1 {uncertainty:.{precision}f}"
    

    
    if isinstance(uncertainty, (list, tuple, np.ndarray)) is False:
        raise_error(
            TypeError,
            f"`uncertainty` must be of type Iterable or a Number. Got {type(uncertainty)} instead.",
        )

    if len(uncertainty) != 2:
        raise_error(
            ValueError,
            f"`uncertainty` list must contain 2 elements. Got {len(uncertainty)} instead.",
        )

    # If the is a None uncertainty, return the value with precision
    if any(error is None for error in uncertainty):
        precision = precision if precision is not None else 3
        return f"{value:.{precision}f}"

    # If precision is None, get the first significant digit of the uncertainty
    if precision is None:
        precision = max(significant_digit(error) for error in uncertainty)

    # # Check if both uncertainties are equal up to precision
    # if np.round(uncertainty[0], precision) == np.round(uncertainty[1], precision):
    #     return f"{value:.{precision}f} \u00B1 {uncertainty[0]:.{precision}f}"

    return f"{value:.{precision}f} [{uncertainty[1]:.{precision}f},{uncertainty[0]:.{precision}f}]"
