import numpy as np
import math
import trajectory_planning_helpers.calc_ax_profile


def calc_t_profile(vx_profile: np.ndarray,
                   el_lengths: np.ndarray,
                   t_start: float = 0.0,
                   ax_profile: np.ndarray = None) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    Calculate a temporal duration profile for a given trajectory.

    .. inputs::
    :param vx_profile:  array containing the velocity profile.
    :type vx_profile:   np.ndarray
    :param el_lengths:  array containing the element lengths between every point of the velocity profile.
    :type el_lengths:   np.ndarray
    :param t_start:     start time in seconds added to first array element.
    :type t_start:      float
    :param ax_profile:  acceleration profile fitting to the velocity profile.
    :type ax_profile:   np.ndarray

    .. outputs::
    :return t_profile:  time profile for the given velocity profile.
    :rtype t_profile:   np.ndarray

    .. notes::
    len(el_lengths) + 1 = len(t_profile)

    len(vx_profile) and len(ax_profile) must be >= len(el_lengths) as the temporal duration from one point to the next
    is only calculated based on the previous point.
    """

    # check inputs
    if vx_profile.size < el_lengths.size:
        raise ValueError("vx_profile and el_lenghts must have at least the same length!")

    if ax_profile is not None and ax_profile.size < el_lengths.size:
        raise ValueError("ax_profile and el_lenghts must have at least the same length!")

    # calculate acceleration profile if required
    if ax_profile is None:
        ax_profile = trajectory_planning_helpers.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile,
                                                                                 el_lengths=el_lengths,
                                                                                 eq_length_output=False)

    # calculate temporal duration of every step between two points
    no_points = el_lengths.size
    t_steps = np.zeros(no_points)

    for i in range(no_points):
        if not math.isclose(ax_profile[i], 0.0):
            t_steps[i] = (-vx_profile[i] + math.sqrt((math.pow(vx_profile[i], 2) + 2 * ax_profile[i] * el_lengths[i])))\
                         / ax_profile[i]

        else:  # ax == 0.0
            t_steps[i] = el_lengths[i] / vx_profile[i]

    # calculate temporal duration profile out of steps
    t_profile = np.insert(np.cumsum(t_steps), 0, 0.0) + t_start

    return t_profile


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
