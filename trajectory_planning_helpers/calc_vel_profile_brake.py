import numpy as np
import math
import trajectory_planning_helpers.calc_vel_profile


def calc_vel_profile_brake(ggv: np.ndarray,
                           kappa: np.ndarray,
                           el_lengths: np.ndarray,
                           v_start: float,
                           drag_coeff: float,
                           m_veh: float,
                           dyn_model_exp: float = 1.0,
                           mu: np.ndarray = None,
                           decel_max: float = None) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    modified by:
    Tim Stahl

    .. description::
    Calculate brake (may also be emergency) velocity profile based on a local trajectory.

    .. inputs::
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
    :type ggv:              np.ndarray
    :param kappa:           curvature profile of given trajectory in rad/m.
    :type kappa:            np.ndarray
    :param el_lengths:      element lengths (distances between coordinates) of given trajectory.
    :type el_lengths:       np.ndarray
    :param v_start:         start velocity in m/s.
    :type v_start:          float
    :param drag_coeff:      drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           vehicle mass in kg.
    :type m_veh:            float
    :param dyn_model_exp:   exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param mu:              friction coefficients.
    :type mu:               np.ndarray
    :param decel_max:       maximum deceleration to be applied (if set to "None", the max. based on ggv and kappa will
                            be used).
    :type decel_max:        float

    .. outputs::
    :return vx_profile:     calculated velocity profile using maximum deceleration of the car.
    :rtype vx_profile:      np.ndarray

    .. notes::
    len(kappa) = len(el_lengths) + 1 = len(mu) = len(vx_profile)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK INPUT ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if decel_max is not None and not decel_max < 0.0:
        raise ValueError("Deceleration input must be negative!")

    if mu is not None and kappa.size != mu.size:
        raise ValueError("kappa and mu must have the same length!")

    if kappa.size != el_lengths.size + 1:
        raise ValueError("kappa must have the length of el_lengths + 1!")

    if v_start < 0.0:
        v_start = 0.0
        print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0,2.0]!')

    if ggv.shape[1] != 3:
        raise ValueError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")

    if ggv[-1, 0] < v_start:
        raise ValueError("ggv has to cover the entire velocity range of the car (i.e. >= v_start)!")

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

        # calculate longitudinal acceleration
        ggv_mod = np.copy(ggv)
        ggv_mod[:, 1] *= -1.0  # use negative acceleration in x axis for forward deceleration
        ax_final = trajectory_planning_helpers.calc_vel_profile.calc_ax_poss(vx_start=vx_profile[i],
                                                                             radius=radii[i],
                                                                             ggv=ggv_mod,
                                                                             ax_max_machines=None,
                                                                             mu=mu[i],
                                                                             mode='decel_forw',
                                                                             dyn_model_exp=dyn_model_exp,
                                                                             drag_coeff=drag_coeff,
                                                                             m_veh=m_veh)

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER DESIRED MAXIMUM DECELERATION ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate equivalent longitudinal acceleration of drag force at the current speed
        ax_drag = -math.pow(vx_profile[i], 2) * drag_coeff / m_veh

        # consider desired maximum deceleration
        if decel_max is not None and ax_final < decel_max:
            # since this planner cannot use positive tire accelerations (this would require another interpolation of
            # ggv[:, 1]) to overcome drag we plan with the drag acceleration if it is greater (i.e. more negative) than
            # the desired maximum deceleration
            if ax_drag < decel_max:
                ax_final = ax_drag
            else:
                ax_final = decel_max

        # --------------------------------------------------------------------------------------------------------------
        # CALCULATE VELOCITY IN THE NEXT POINT -------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        radicand = math.pow(vx_profile[i], 2) + 2 * ax_final * el_lengths[i]

        if radicand < 0.0:
            # standstill is reached
            break
        else:
            vx_profile[i + 1] = math.sqrt(radicand)

    return vx_profile


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
