import numpy as np


def calc_vel_profile_brake(ggv: np.ndarray, kappa: np.ndarray, el_lengths: np.ndarray, v_start: float,
                           mu: np.ndarray = None, decel_max: float = None, tire_model_exp: float = 2.0) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Documentation:
    Calculate brake (may also be emergency) velocity profile based on a local trajectory.

    Inputs:
    ggv:          ggv-diagram to be applied
    kappa:        curvature profile of given trajectory in rad/m
    el_lengths:   element lengths (distances between coordinates) of given trajectory
    v_start:      start velocity in m/s
    mu:           friction coefficients
    decel_max:    maximum deceleration to be applied (if set to "None", the max. based on ggv and kappa will be used)

    len(vx_profile) = len(kappa) = len(el_lengths) + 1
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK INPUT ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if decel_max is not None and not decel_max < 0.0:
        raise ValueError("Deceleration input must be negative!")

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = kappa.size

    # create velocity profile array and set initial speed
    vx_profile = np.zeros(no_points)
    vx_profile[0] = v_start

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0))

    # set mu if it is None
    if mu is None:
        mu = np.ones(no_points)

    # ------------------------------------------------------------------------------------------------------------------
    # PURE FORWARD SOLVER ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    for i in range(no_points - 1):
        # calculate required values
        ax_min_cur_tires = mu[i] * np.interp(vx_profile[i], ggv[:, 0], ggv[:, 3])
        ay_max_cur_tires = mu[i] * np.interp(vx_profile[i], ggv[:, 0], ggv[:, 4])
        ay_used_cur = np.power(vx_profile[i], 2) / radii[i]

        if ay_used_cur < ay_max_cur_tires:
            # car is able to stay on track -> decelerate with unused tire potential
            radicand = 1 - np.power(ay_used_cur / ay_max_cur_tires, tire_model_exp)
            ax_possible_cur_tires = ax_min_cur_tires * np.power(radicand, 1.0 / tire_model_exp)

            # check if ax_possible_cur_tires is more than allowed decel_max and reduce it if so
            if decel_max is not None and ax_possible_cur_tires < decel_max:
                ax_use = decel_max
            else:
                ax_use = ax_possible_cur_tires

            # calculate velocity in the next point based on ax_use
            radicand = np.power(vx_profile[i], 2) + 2 * ax_use * el_lengths[i]

            if radicand < 0.0:
                break
            else:
                vx_profile[i + 1] = np.sqrt(radicand)

        # if lateral acceleration is used completely do not apply any longitudinal deceleration
        else:
            vx_profile[i + 1] = vx_profile[i]

    return vx_profile


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
