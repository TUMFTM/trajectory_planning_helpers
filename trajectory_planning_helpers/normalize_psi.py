import numpy as np


def normalize_psi(psi: np.ndarray) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Normalize heading psi such that [-pi, pi[ holds as interval boundaries.

    Inputs:
    psi: array containing headings psi to be normalized.
    """

    # use modulo operator to remove multiples of 2*pi
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * np.pi)

    # restrict psi to [-pi, pi[
    if type(psi_out) is np.ndarray:
        psi_out[psi_out >= np.pi] -= 2 * np.pi
        psi_out[psi_out < -np.pi] += 2 * np.pi

    else:
        if psi_out >= np.pi:
            psi_out -= 2 * np.pi
        elif psi_out < -np.pi:
            psi_out += 2 * np.pi

    return psi_out


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
