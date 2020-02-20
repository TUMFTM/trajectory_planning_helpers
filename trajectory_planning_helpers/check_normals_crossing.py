import numpy as np


def check_normals_crossing(track: np.ndarray,
                           normvec_normalized: np.ndarray,
                           horizon: int = 10) -> bool:
    """
    author:
    Alexander Heilmeier

    .. description::
    This function checks spline normals for crossings. Returns True if a crossing was found, otherwise False.

    .. inputs::
    :param track:               array containing the track [x, y, w_tr_right, w_tr_left] to check
    :type track:                np.ndarray
    :param normvec_normalized:  array containing normalized normal vectors for every track point
                                [x_component, y_component]
    :type normvec_normalized:   np.ndarray
    :param horizon:             determines the number of normals in forward and backward direction that are checked
                                against each normal on the line
    :type horizon:              int

    .. outputs::
    :return found_crossing:     bool value indicating if a crossing was found or not
    :rtype found_crossing:      bool

    .. notes::
    The checks can take a while if full check is performed. Inputs are unclosed.
    """

    # check input
    no_points = track.shape[0]

    if horizon >= no_points:
        raise ValueError("Horizon of %i points is too large for a track with %i points, reduce horizon!"
                         % (horizon, no_points))

    elif horizon >= no_points / 2:
        print("WARNING: Horizon of %i points makes no sense for a track with %i points, reduce horizon!"
              % (horizon, no_points))

    # initialization
    les_mat = np.zeros((2, 2))
    idx_list = list(range(0, no_points))
    idx_list = idx_list[-horizon:] + idx_list + idx_list[:horizon]

    # loop through all points of the track to check for crossings in their neighbourhoods
    for idx in range(no_points):

        # determine indices of points in the neighbourhood of the current index
        idx_neighbours = idx_list[idx:idx + 2 * horizon + 1]
        del idx_neighbours[horizon]
        idx_neighbours = np.array(idx_neighbours)

        # remove indices of normal vectors that are collinear to the current index
        is_collinear_b = np.isclose(np.cross(normvec_normalized[idx], normvec_normalized[idx_neighbours]), 0.0)
        idx_neighbours_rel = idx_neighbours[np.nonzero(np.invert(is_collinear_b))[0]]

        # check crossings solving an LES
        for idx_comp in list(idx_neighbours_rel):

            # LES: x_1 + lambda_1 * nx_1 = x_2 + lambda_2 * nx_2; y_1 + lambda_1 * ny_1 = y_2 + lambda_2 * ny_2;
            const = track[idx_comp, :2] - track[idx, :2]
            les_mat[:, 0] = normvec_normalized[idx]
            les_mat[:, 1] = -normvec_normalized[idx_comp]

            # solve LES
            lambdas = np.linalg.solve(les_mat, const)

            # we have a crossing within the relevant part if both lambdas lie between -w_tr_left and w_tr_right
            if -track[idx, 3] <= lambdas[0] <= track[idx, 2] \
                    and -track[idx_comp, 3] <= lambdas[1] <= track[idx_comp, 2]:
                return True  # found crossing

    return False


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
