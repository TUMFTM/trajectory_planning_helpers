import numpy as np


def filt_vel_profile(v_profile: np.ndarray, filt_window: int, closed: bool) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Documentation:
    Filter a given velocity profile using a moving average filter.

    Inputs:
    vel_profile:        velocity profile that should be filtered
    filt_window:        filter window size for moving average filter (must be odd)
    closed:             flag showing if the velocity profile can be considered as closable

    vel_profile input is unclosed!
    """

    # check if window width is odd
    if not filt_window % 2 == 1:
        raise IOError("Window width of moving average filter in velocity profile filtering must be odd!")

    # calculate half window width - 1
    w_window_half = int((filt_window - 1) / 2)

    # apply filter
    if closed:
        # temporarily add points in front of and behind v_profile
        v_profile_tmp = np.concatenate((v_profile[-w_window_half:], v_profile, v_profile[:w_window_half]), axis=0)

        # convolution filter used as a moving average filter and remove temporary points
        v_profile_filt = np.convolve(v_profile_tmp,
                                     np.ones(filt_window) / float(filt_window),
                                     mode="same")[w_window_half:-w_window_half]

    else:
        # implementation 1: include boundaries during filtering
        # no_points = v_profile.size
        # v_profile_filt = np.zeros(no_points)
        #
        # for i in range(no_points):
        #     if i < w_window_half:
        #         v_profile_filt[i] = np.average(v_profile[:i + w_window_half + 1])
        #
        #     elif i < no_points - w_window_half:
        #         v_profile_filt[i] = np.average(v_profile[i - w_window_half:i + w_window_half + 1])
        #
        #     else:
        #         v_profile_filt[i] = np.average(v_profile[i - w_window_half:])

        # implementation 2: start filtering at w_window_half and stop at -w_window_half
        v_profile_filt = np.copy(v_profile)
        v_profile_filt[w_window_half:-w_window_half] = np.convolve(v_profile,
                                                                   np.ones(filt_window) / float(filt_window),
                                                                   mode="same")[w_window_half:-w_window_half]

    return v_profile_filt


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
