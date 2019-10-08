import numpy as np


def get_rel_path_part(path_cl: np.ndarray,
                      s_pos: float,
                      s_dist_back: float = 20.0,
                      s_dist_forw: float = 20.0) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function returns the relevant part of a closed path (e.g. on the racetrack) on the basis of a given s position.
    The distances s_dist_forw and s_dist_backw are used to determine how much the path should reach forward and
    backward from this position.

    Inputs:
    path_cl:        Closed path of which we want to extract the relevant part ([s, x, y]).
    s_pos:          s position of the vehicle in m (matched to the s coordinate of path_cl).
    s_dist_back:    Backward distance in m from current s position. Including last point before that value!
    s_dist_forw:    Forward distance in m from current s position. Including first point after that value!

    Outputs:
    path_rel:       Relevant part of the path ([s, x, y]). Attention: s coordinate does not start at 0m!
    """

    # get s_tot into a variable
    s_tot = path_cl[-1, 0]

    # check distance input
    if s_dist_back + s_dist_forw >= s_tot:
        raise ValueError('Summed distance inputs are greater or equal to the total distance of the given path!')

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
    ind_start = np.searchsorted(path_cl[:, 0], s_min, side="right") - 1
    # + 1 to include trajectory point after s_max when slicing
    ind_stop = np.searchsorted(path_cl[:, 0], s_max, side="left") + 1

    # catch case of reaching into the next lap
    if ind_start < ind_stop:
        # common case
        path_rel = path_cl[ind_start:ind_stop]

    else:
        # overlapping case
        # temporarily add s_tot to the part in the "next lap" for convenient interpolation afterwards
        path_rel_part2 = np.copy(path_cl[:ind_stop])
        path_rel_part2[:, 0] += s_tot

        # :-1 for first part to include last/first point of closed trajectory only once
        path_rel = np.vstack((path_cl[ind_start:-1], path_rel_part2))

    return path_rel


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
