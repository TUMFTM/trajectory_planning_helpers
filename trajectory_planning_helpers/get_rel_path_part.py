import numpy as np


def get_rel_path_part(path_cl: np.ndarray,
                      s_pos: float,
                      s_dist_back: float = 20.0,
                      s_dist_forw: float = 20.0,
                      bound_right_cl: np.ndarray = None,
                      bound_left_cl: np.ndarray = None) -> tuple:
    """
    author:
    Alexander Heilmeier

    .. description::
    This function returns the relevant part of a closed path (e.g. on the racetrack) on the basis of a given s position.
    The distances s_dist_forw and s_dist_backw are used to determine how much the path should reach forward and
    backward from this position.

    .. inputs::
    :param path_cl:         Closed path of which we want to extract the relevant part ([s, x, y]).
    :type path_cl:          np.ndarray
    :param s_pos:           s position of the vehicle in m (matched to the s coordinate of path_cl).
    :type s_pos:            float
    :param s_dist_back:     Backward distance in m from current s position. Including last point before that value!
    :type s_dist_back:      float
    :param s_dist_forw:     Forward distance in m from current s position. Including first point after that value!
    :type s_dist_forw:      float
    :param bound_right_cl:  Optional input: Right boundary ([x, y]) of path_cl. Every boundary point belongs to the path
                            point on the same index, i.e. they have the same number of points.
    :type bound_right_cl:   np.ndarray
    :param bound_left_cl:   Optional input: Right boundary ([x, y]) of path_cl. Every boundary point belongs to the path
                            point on the same index, i.e. they have the same number of points.
    :type bound_left_cl:    np.ndarray

    .. outputs::
    :return path_rel:           Relevant part of the path ([s, x, y]). Attention: s coordinate does not start at 0m!
    :rtype path_rel:            np.ndarray
    :return bound_right_rel:    Relevant part of right boundary ([x, y]). None if not inserted.
    :rtype bound_right_rel:     np.ndarray
    :return bound_left_rel:     Relevant part of left boundary ([x, y]). None if not inserted.
    :rtype bound_left_rel:      np.ndarray
    """

    # get s_tot into a variable
    s_tot = path_cl[-1, 0]

    # check distance input
    if s_dist_back + s_dist_forw >= s_tot:
        raise ValueError('Summed distance inputs are greater or equal to the total distance of the given path!')

    # check boundaries
    if bound_right_cl is not None and bound_right_cl.shape[0] != path_cl.shape[0]:
        raise ValueError('Inserted right boundary does not have the same number of points as the path!')

    if bound_left_cl is not None and bound_left_cl.shape[0] != path_cl.shape[0]:
        raise ValueError('Inserted left boundary does not have the same number of points as the path!')

    # cut s position if it exceeds the path length
    if s_pos >= s_tot:
        s_pos -= s_tot

    # set s boundaries
    s_min = s_pos - s_dist_back
    s_max = s_pos + s_dist_forw

    if s_min < 0.0:
        s_min += s_tot

    if s_max > s_tot:
        s_max -= s_tot

    # now the following holds: s_min -> [0.0; s_tot[ s_max -> ]0.0; s_tot]

    # get indices of according points
    # - 1 to include trajectory point before s_min
    idx_start = np.searchsorted(path_cl[:, 0], s_min, side="right") - 1
    # + 1 to include trajectory point after s_max when slicing
    idx_stop = np.searchsorted(path_cl[:, 0], s_max, side="left") + 1

    # catch case of reaching into the next lap
    if idx_start < idx_stop:
        # common case
        path_rel = path_cl[idx_start:idx_stop]

        if bound_right_cl is not None:
            bound_right_rel = bound_right_cl[idx_start:idx_stop]
        else:
            bound_right_rel = None

        if bound_left_cl is not None:
            bound_left_rel = bound_left_cl[idx_start:idx_stop]
        else:
            bound_left_rel = None

    else:
        # overlapping case
        # temporarily add s_tot to the part in the "next lap" for convenient interpolation afterwards
        path_rel_part2 = np.copy(path_cl[:idx_stop])
        path_rel_part2[:, 0] += s_tot

        # :-1 for first part to include last/first point of closed trajectory only once
        path_rel = np.vstack((path_cl[idx_start:-1], path_rel_part2))

        if bound_right_cl is not None:
            bound_right_rel = np.vstack((bound_right_cl[idx_start:-1], bound_right_cl[:idx_stop]))
        else:
            bound_right_rel = None

        if bound_left_cl is not None:
            bound_left_rel = np.vstack((bound_left_cl[idx_start:-1], bound_left_cl[:idx_stop]))
        else:
            bound_left_rel = None

    return path_rel, bound_right_rel, bound_left_rel


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
