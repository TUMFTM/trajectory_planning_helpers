import numpy as np
import math
from typing import Union


def normalize_psi(psi: Union[np.ndarray, float]) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    Normalize heading psi such that [-pi,pi[ holds as interval boundaries.

    .. inputs::
    :param psi:         array containing headings psi to be normalized.
    :type psi:          Union[np.ndarray, float]

    .. outputs::
    :return psi_out:    array with normalized headings psi.
    :rtype psi_out:     np.ndarray

    .. notes::
    len(psi) = len(psi_out)
    """

    # use modulo operator to remove multiples of 2*pi
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)

    # restrict psi to [-pi,pi[
    if type(psi_out) is np.ndarray:
        psi_out[psi_out >= math.pi] -= 2 * math.pi
        psi_out[psi_out < -math.pi] += 2 * math.pi

    else:
        if psi_out >= math.pi:
            psi_out -= 2 * math.pi
        elif psi_out < -math.pi:
            psi_out += 2 * math.pi

    return psi_out


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
