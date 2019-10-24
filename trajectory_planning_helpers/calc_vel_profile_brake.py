import numpy as np
import math
import warnings


def calc_vel_profile_brake(ggv: np.ndarray,
                           kappa: np.ndarray,
                           el_lengths: np.ndarray,
                           v_start: float,
                           mu: np.ndarray = None,
                           decel_max: float = None,
                           dyn_model_exp: float = 1.0,
                           drag_coeff: float = 0.85,
                           m_veh: float = 1160.0) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Documentation:
    Calculate brake (may also be emergency) velocity profile based on a local trajectory.

    Inputs:
    ggv:            ggv-diagram to be applied: [v, ax_max_machines, ax_max_tires, ax_min_tires, ay_max_tires].
                    ax_max_machines should be handed in without considering drag resistance!
    kappa:          curvature profile of given trajectory in rad/m.
    el_lengths:     element lengths (distances between coordinates) of given trajectory.
    v_start:        start velocity in m/s.
    mu:             friction coefficients.
    decel_max:      maximum deceleration to be applied (if set to "None", the max. based on ggv and kappa will be used).
    dyn_model_exp:  exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    drag_coeff:     drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    m_veh:          vehicle mass in kg.

    Outputs:
    vx_profile:     calculated velocity profile using maximum deceleration of the car.

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
        warnings.warn('Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if not 1.0 <= dyn_model_exp <= 2.0:
        warnings.warn('Exponent for the vehicle dynamics model should be in the range [1.0,2.0]!')

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

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER TIRE POTENTIAL --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        ax_max_tires = mu[i] * np.interp(vx_profile[i], ggv[:, 0], ggv[:, 3])
        ay_max_tires = mu[i] * np.interp(vx_profile[i], ggv[:, 0], ggv[:, 4])
        ay_used = math.pow(vx_profile[i], 2) / radii[i]

        radicand = 1 - math.pow(ay_used / ay_max_tires, dyn_model_exp)

        if radicand > 0.0:
            ax_avail_tires = ax_max_tires * math.pow(radicand, 1.0 / dyn_model_exp)
        else:
            ax_avail_tires = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER MACHINE LIMITATIONS ---------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # no limitations in braking case
        ax_avail_vehicle = ax_avail_tires

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER DRAG ------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate equivalent longitudinal acceleration of drag force at the current speed
        ax_drag = -math.pow(vx_profile[i], 2) * drag_coeff / m_veh
        ax_final = ax_avail_vehicle + ax_drag

        # --------------------------------------------------------------------------------------------------------------
        # CONSIDER DESIRED MAXIMUM DECELERATION ------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if decel_max is not None and ax_final < decel_max:
            # since this planner cannot use positive tire accelerations (this would require another interpolation of
            # ggv[:, 2]) to overcome drag we plan with the drag acceleration if it is greater (i.e. more negative) than
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
