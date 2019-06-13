import numpy as np
import trajectory_planning_helpers.calc_ax_profile


def calc_t_profile(vx_profile: np.ndarray,
                   el_lengths: np.ndarray,
                   t_start: float = 0.0,
                   ax_profile=None) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Calculate a temporal duration profile for a given trajectory.

    Inputs:
    vx_profile: array containing the velocity profile.
    el_lengths: array containing the element lengths between every point of the velocity profile.
    t_start: start time in s added to first array element.

    len(t_profile) == len(el_lengths) + 1.
    """

    # calculate acceleration profile if required (len(ax_profile) == len(vx_profile) - 1)
    if ax_profile is None:
        ax_profile = trajectory_planning_helpers.calc_ax_profile.calc_ax_profile(vx_profile=vx_profile,
                                                                                 el_lengths=el_lengths,
                                                                                 eq_length_output=False)

    # calculate temporal duration of every step between two points
    no_points = el_lengths.size
    t_steps = np.zeros(no_points)

    for i in range(no_points):
        if not np.isclose(ax_profile[i], 0.0):
            t_steps[i] = (-vx_profile[i] + np.sqrt((np.power(vx_profile[i], 2) + 2 * ax_profile[i] * el_lengths[i]))) \
                         / ax_profile[i]

        else:  # ax == 0.0
            t_steps[i] = el_lengths[i] / vx_profile[i]

    # calculate temporal duration profile out of steps
    t_profile = np.insert(np.cumsum(t_steps), 0, 0.0) + t_start

    return t_profile


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
