import numpy as np
import math
from typing import Union


def normalize_psi(psi: Union[np.ndarray, float]) -> np.ndarray:
    """
    Author:
    Alexander Heilmeier

    Description:
    Normalize heading psi such that [-pi,pi[ holds as interval boundaries.

    Inputs:
    psi:        array containing headings psi to be normalized.

    Outputs:
    psi_out:    array with normalized headings psi.

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
