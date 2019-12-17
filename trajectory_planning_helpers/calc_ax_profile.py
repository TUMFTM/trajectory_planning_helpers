import numpy as np


def calc_ax_profile(vx_profile: np.ndarray,
                    el_lengths: np.ndarray,
                    eq_length_output: bool = False) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    The function calculates the acceleration profile for a given velocity profile.

    .. inputs::
    :param vx_profile:          array containing the velocity profile used as a basis for the acceleration calculations.
    :type vx_profile:           np.ndarray
    :param el_lengths:          array containing the element lengths between every point of the velocity profile.
    :type el_lengths:           np.ndarray
    :param eq_length_output:    assumes zero acceleration for the last point of the acceleration profile and therefore
                                returns ax_profile with equal length to vx_profile.
    :type eq_length_output:     bool

    .. outputs::
    :return ax_profile:         acceleration profile calculated for the inserted vx_profile.
    :rtype ax_profile:          np.ndarray

    .. notes::
    case eq_length_output is True:
    len(vx_profile) = len(el_lengths) + 1 = len(ax_profile)

    case eq_length_output is False:
    len(vx_profile) = len(el_lengths) + 1 = len(ax_profile) + 1
    """

    # check inputs
    if vx_profile.size != el_lengths.size + 1:
        raise ValueError("Array size of vx_profile should be 1 element bigger than el_lengths!")

    # calculate longitudinal acceleration profile array numerically: (v_end^2 - v_beg^2) / 2*s
    if eq_length_output:
        ax_profile = np.zeros(vx_profile.size)
        ax_profile[:-1] = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)
    else:
        ax_profile = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)

    return ax_profile


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
