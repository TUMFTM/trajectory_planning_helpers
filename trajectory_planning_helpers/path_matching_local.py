import numpy as np
import trajectory_planning_helpers.angle3pt
from typing import Union


def path_matching_local(path: np.ndarray,
                        ego_position: np.ndarray,
                        consider_as_closed: bool = False,
                        s_tot: Union[float, None] = None) -> tuple:
    """
    author:
    Alexander Heilmeier

    .. description::
    Get the corresponding s coordinate and the displacement of the own vehicle in relation to a local path.

    .. inputs::
    :param path:                Unclosed path used to match ego position ([s, x, y]).
    :type path:                 np.ndarray
    :param ego_position:        Ego position of the vehicle ([x, y]).
    :type ego_position:         np.ndarray
    :param consider_as_closed:  If the path is closed in reality we can interpolate between last and first point. This
                                can be enforced by setting consider_as_closed = True.
    :type consider_as_closed:   bool
    :param s_tot:               Total length of path in m.
    :type s_tot:                Union[float, None]

    .. outputs::
    :return s_interp:           Interpolated s position of the vehicle in m.
    :rtype s_interp:            np.ndarray
    :return d_displ:            Estimated displacement from the trajectory in m.
    :rtype d_displ:             np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK INPUT ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if path.shape[1] != 3:
        raise ValueError("Inserted path must have 3 columns [s, x, y]!")

    if consider_as_closed and s_tot is None:
        print("WARNING: s_tot is not handed into path_matching_local function! Estimating s_tot on the basis of equal"
              "stepsizes")
        s_tot = path[-1, 0] + path[1, 0] - path[0, 0]  # assume equal stepsize

    # ------------------------------------------------------------------------------------------------------------------
    # SELF LOCALIZATION ON RACELINE ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get the nearest path point to ego position
    dists_to_cg = np.hypot(path[:, 1] - ego_position[0], path[:, 2] - ego_position[1])
    ind_min = np.argpartition(dists_to_cg, 1)[0]

    # get previous and following point on path
    if consider_as_closed:
        if ind_min == 0:
            ind_prev = dists_to_cg.shape[0] - 1
            ind_follow = 1

        elif ind_min == dists_to_cg.shape[0] - 1:
            ind_prev = ind_min - 1
            ind_follow = 0

        else:
            ind_prev = ind_min - 1
            ind_follow = ind_min + 1

    else:
        ind_prev = max(ind_min - 1, 0)
        ind_follow = min(ind_min + 1, dists_to_cg.shape[0] - 1)

    # get angle between selected point and neighbours: ang1 to previous point, ang2 to following point on path
    ang_prev = np.abs(trajectory_planning_helpers.angle3pt.angle3pt(path[ind_min, 1:3],
                                                                    ego_position,
                                                                    path[ind_prev, 1:3]))

    ang_follow = np.abs(trajectory_planning_helpers.angle3pt.angle3pt(path[ind_min, 1:3],
                                                                      ego_position,
                                                                      path[ind_follow, 1:3]))

    # extract neighboring points -> closest point and the point resulting in the larger angle
    if ang_prev > ang_follow:
        a_pos = path[ind_prev, 1:3]
        b_pos = path[ind_min, 1:3]
        s_curs = np.append(path[ind_prev, 0], path[ind_min, 0])
    else:
        a_pos = path[ind_min, 1:3]
        b_pos = path[ind_follow, 1:3]
        s_curs = np.append(path[ind_min, 0], path[ind_follow, 0])

    # adjust s if closed path shell be considered and we have the case of interpolation between last and first point
    if consider_as_closed:
        if ind_min == 0 and ang_prev > ang_follow:
            s_curs[1] = s_tot
        elif ind_min == dists_to_cg.shape[0] - 1 and ang_prev <= ang_follow:
            s_curs[1] = s_tot

    # interpolate between those points (linear) for better positioning
    no_interp_values = 11
    t_lin = np.linspace(0.0, 1.0, no_interp_values)  # set relative lengths that are evaluated for interpolation
    x_cg_interp = np.linspace(a_pos[0], b_pos[0], no_interp_values)
    y_cg_interp = np.linspace(a_pos[1], b_pos[1], no_interp_values)

    # get nearest of those interpolated points relative to ego position
    dists_to_cg = np.hypot(x_cg_interp - ego_position[0], y_cg_interp - ego_position[1])
    ind_min_interp = np.argpartition(dists_to_cg, 1)[0]
    t_lin_used = t_lin[ind_min_interp]

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE REQUIRED INFORMATION -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate current path length
    s_interp = np.interp(t_lin_used, (0.0, 1.0), s_curs)

    # get displacement between ego position and path (needed for lookahead distance)
    d_displ = dists_to_cg[ind_min_interp]

    return s_interp, d_displ


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
