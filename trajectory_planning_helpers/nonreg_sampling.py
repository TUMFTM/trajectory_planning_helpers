import numpy as np
import trajectory_planning_helpers as tph


def nonreg_sampling(track: np.ndarray,
                    eps_kappa: float = 1e-3,
                    step_non_reg: int = 0) -> tuple:
    """
    author:
    Thomas Herrmann

    .. description::
    The non-regular sampling function runs through the curvature profile and determines straight and corner sections.
    During straight sections it reduces the amount of points by skipping them depending on the step_non_reg parameter.

    .. inputs::
    :param track:           [x, y, w_tr_right, w_tr_left] (always unclosed).
    :type track:            np.ndarray
    :param eps_kappa:       identify straights using this threshold in curvature in rad/m, i.e. straight if
                            kappa < eps_kappa
    :type eps_kappa:        float
    :param step_non_reg:    determines how many points are skipped in straight sections, e.g. step_non_reg = 3 means
                            every fourth point is used while three points are skipped
    :type step_non_reg:     int

    .. outputs::
    :return track_sampled:  [x, y, w_tr_right, w_tr_left] sampled track (always unclosed).
    :rtype track_sampled:   np.ndarray
    :return sample_idxs:    indices of points that are kept
    :rtype sample_idxs:     np.ndarray
    """

    # if stepsize is equal to zero simply return the input
    if step_non_reg == 0:
        return track, np.arange(0, track.shape[0])

    # calculate curvature (required to be able to differentiate straight and corner sections)
    path_cl = np.vstack((track[:, :2], track[0, :2]))
    coeffs_x, coeffs_y = tph.calc_splines.calc_splines(path=path_cl)[:2]
    kappa_path = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x,
                                                         coeffs_y=coeffs_y,
                                                         ind_spls=np.arange(0, coeffs_x.shape[0]),
                                                         t_spls=np.zeros(coeffs_x.shape[0]))[1]

    # run through the profile to determine the indices of the points that are kept
    idx_latest = step_non_reg + 1
    sample_idxs = [0]

    for idx in range(1, len(kappa_path)):
        if np.abs(kappa_path[idx]) >= eps_kappa or idx >= idx_latest:
            # keep this point
            sample_idxs.append(idx)
            idx_latest = idx + step_non_reg + 1

    return track[sample_idxs], np.array(sample_idxs)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
