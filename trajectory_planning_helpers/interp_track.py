import numpy as np
import math


def interp_track(track: np.ndarray,
                 stepsize: float) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    Interpolate track points linearly to a new stepsize.

    .. inputs::
    :param track:           track of the form [x, y, w_tr_right, w_tr_left].
    :type track:            np.ndarray
    :param stepsize:        desired stepsize after interpolation in m.
    :type stepsize:         float

    .. outputs::
    :return track_interp:   interpolated track [x, y, w_tr_right, w_tr_left].
    :rtype track_interp:    np.ndarray

    .. notes::
    track input and output are unclosed! track input must however be closable in the current form!
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LINEAR INTERPOLATION OF TRACK ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create closed track
    track_cl = np.vstack((track, track[0]))

    # calculate element lengths (euclidian distance)
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))

    # sum up total distance (from start) to every element
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)

    # calculate desired lenghts depending on specified stepsize (+1 because last element is included)
    no_points_interp_cl = math.ceil(dists_cum_cl[-1] / stepsize) + 1
    dists_interp_cl = np.linspace(0.0, dists_cum_cl[-1], no_points_interp_cl)

    # interpolate closed track points
    track_interp_cl = np.zeros((no_points_interp_cl, 4))
    track_interp_cl[:, 0] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 0])
    track_interp_cl[:, 1] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 1])
    track_interp_cl[:, 2] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 2])
    track_interp_cl[:, 3] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 3])

    return track_interp_cl[:-1]


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
