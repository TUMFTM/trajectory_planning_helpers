import numpy as np
import math
import trajectory_planning_helpers.conv_filt


def calc_vel_profile(ggv: np.ndarray,
                     kappa: np.ndarray,
                     el_lengths: np.ndarray,
                     closed: bool,
                     dyn_model_exp: float,
                     drag_coeff: float,
                     m_veh: float,
                     mu: np.ndarray = None,
                     v_start: float = None,
                     v_end: float = None,
                     filt_window: int = None) -> np.ndarray:
    """
    Author:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Description:
    Calculates a velocity profile using the tire and motor limits as good as possible.

    Inputs:
    ggv:                ggv-diagram to be applied: [v, ax_max_machines, ax_max_tires, ax_min_tires, ay_max_tires].
                        ax_max_machines should be handed in without considering drag resistance!
    kappa:              curvature profile of given trajectory in rad/m (always unclosed).
    el_lengths:         element lengths (distances between coordinates) of given trajectory.
    closed:             flag to set if the velocity profile must be calculated for a closed or unclosed trajectory.
    mu:                 friction coefficients (always unclosed).
    v_start:            start velocity in m/s (used in unclosed case only).
    v_end:              end velocity in m/s (used in unclosed case only).
    filt_window:        filter window size for moving average filter (must be odd).
    dyn_model_exp:      exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    drag_coeff:         drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    m_veh:              vehicle mass in kg.

    All inputs must be inserted unclosed, i.e. kappa[-1] != kappa[0], even if closed is set True! (el_lengths is kind of
    closed if closed is True of course!)

    Outputs:
    vx_profile:         calculated velocity profile (always unclosed).

    case closed is True:
    len(kappa) = len(el_lengths) = len(mu) = len(vx_profile)

    case closed is False:
    len(kappa) = len(el_lengths) + 1 = len(mu) = len(vx_profile)
    """

    # check inputs
    if mu is not None and kappa.size != mu.size:
        raise ValueError("kappa and mu must have the same length!")

    if closed and kappa.size != el_lengths.size:
        raise ValueError("kappa and el_lengths must have the same length if closed!")

    elif not closed and kappa.size != el_lengths.size + 1:
        raise ValueError("kappa must have the length of el_lengths + 1 if unclosed!")

    if not closed and v_start is None:
        raise ValueError("v_start must be provided for the unclosed case!")

    if v_start is not None and v_start < 0.0:
        v_start = 0.0
        print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if v_end is not None and v_end < 0.0:
        v_end = 0.0
        print('WARNING: Input v_end was < 0.0. Using v_end = 0.0 instead!')

    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0,2.0]!')

    # ------------------------------------------------------------------------------------------------------------------
    # SPEED PROFILE CALCULATION (FB) -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1.0, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0.0))

    # set mu to one in case it is not set
    if mu is None:
        mu = np.ones(kappa.size)

    # call solver
    if not closed:
        vx_profile = __solver_fb_unclosed(ggv=ggv,
                                          radii=radii,
                                          el_lengths=el_lengths,
                                          mu=mu,
                                          v_start=v_start,
                                          v_end=v_end,
                                          dyn_model_exp=dyn_model_exp,
                                          drag_coeff=drag_coeff,
                                          m_veh=m_veh)

    else:
        vx_profile = __solver_fb_closed(ggv=ggv,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        mu=mu,
                                        dyn_model_exp=dyn_model_exp,
                                        drag_coeff=drag_coeff,
                                        m_veh=m_veh)

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if filt_window is not None:
        vx_profile = trajectory_planning_helpers.conv_filt.conv_filt(signal=vx_profile,
                                                                     filt_window=filt_window,
                                                                     closed=closed)

    return vx_profile


def __solver_fb_unclosed(ggv: np.ndarray,
                         radii: np.ndarray,
                         el_lengths: np.ndarray,
                         mu: np.ndarray,
                         v_start: float,
                         v_end: float = None,
                         dyn_model_exp: float = 1.0,
                         drag_coeff: float = 0.85,
                         m_veh: float = 1160.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # run through all the points and check for possible lateral acceleration
    mu_mean = np.mean(mu)
    ay_max_global = mu_mean * np.amin(np.abs(ggv[:, 4]))    # get first lateral acceleration estimate
    vx_profile = np.sqrt(ay_max_global * radii)             # get first velocity profile estimate

    ay_max_curr = mu * np.interp(vx_profile, ggv[:, 0], ggv[:, 4])
    vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

    # cut vx_profile to car's top speed
    vx_max = ggv[-1, 0]
    vx_profile[vx_profile > vx_max] = vx_max

    # consider v_start
    if vx_profile[0] > v_start:
        vx_profile[0] = v_start

    # calculate acceleration profile
    vx_profile = __solver_fb_acc_profile(ggv=ggv,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=False,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)

    # consider v_end
    if v_end is not None and vx_profile[-1] > v_end:
        vx_profile[-1] = v_end

    # calculate deceleration profile
    vx_profile = __solver_fb_acc_profile(ggv=ggv,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=True,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)

    return vx_profile


def __solver_fb_closed(ggv: np.ndarray,
                       radii: np.ndarray,
                       el_lengths: np.ndarray,
                       mu: np.ndarray,
                       dyn_model_exp: float = 1.0,
                       drag_coeff: float = 0.85,
                       m_veh: float = 1160.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

    # run through all the points and check for possible lateral acceleration
    mu_mean = np.mean(mu)
    ay_max_global = mu_mean * np.amin(np.abs(ggv[:, 4]))    # get first lateral acceleration estimate
    vx_profile = np.sqrt(ay_max_global * radii)             # get first velocity estimate (radii must be positive!)

    # do it two times to improve accuracy (because of velocity-dependent accelerations)
    for i in range(2):
        ay_max_curr = mu * np.interp(vx_profile, ggv[:, 0], ggv[:, 4])
        vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

    # cut vx_profile to car's top speed
    vx_max = ggv[-1, 0]
    vx_profile[vx_profile > vx_max] = vx_max

    """We need to calculate the speed profile for two laps to get the correct starting and ending velocity."""

    # double arrays
    vx_profile_double = np.concatenate((vx_profile, vx_profile), axis=0)
    radii_double = np.concatenate((radii, radii), axis=0)
    el_lengths_double = np.concatenate((el_lengths, el_lengths), axis=0)
    mu_double = np.concatenate((mu, mu), axis=0)

    # calculate acceleration profile
    vx_profile_double = __solver_fb_acc_profile(ggv=ggv,
                                                radii=radii_double,
                                                el_lengths=el_lengths_double,
                                                mu=mu_double,
                                                vx_profile=vx_profile_double,
                                                backwards=False,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh)

    # use second lap of acceleration profile
    vx_profile_double = np.concatenate((vx_profile_double[no_points:], vx_profile_double[no_points:]), axis=0)

    # calculate deceleration profile
    vx_profile_double = __solver_fb_acc_profile(ggv=ggv,
                                                radii=radii_double,
                                                el_lengths=el_lengths_double,
                                                mu=mu_double,
                                                vx_profile=vx_profile_double,
                                                backwards=True,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh)

    # use second lap of deceleration profile
    vx_profile = vx_profile_double[no_points:]

    return vx_profile


def __solver_fb_acc_profile(ggv: np.ndarray,
                            radii: np.ndarray,
                            el_lengths: np.ndarray,
                            mu: np.ndarray,
                            vx_profile: np.ndarray,
                            dyn_model_exp: float,
                            drag_coeff: float,
                            m_veh: float,
                            backwards: bool = False) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    vx_max = ggv[-1, 0]
    no_points = vx_profile.size

    # check for reversed direction -> modify vx_profile, radii and el_lengths
    if backwards:
        ggv_mod = ggv[:, [0, 1, 3, 4]]  # use negative acceleration in x axis if we are going backwards
        radii_mod = np.flipud(radii)
        el_lengths_mod = np.flipud(el_lengths)
        mu_mod = np.flipud(mu)
        vx_profile = np.flipud(vx_profile)
        mode = 'decel_backw'
    else:
        ggv_mod = ggv[:, [0, 1, 2, 4]]
        radii_mod = radii
        el_lengths_mod = el_lengths
        mu_mod = mu
        mode = 'accel_forw'

    # ------------------------------------------------------------------------------------------------------------------
    # SEARCH START POINTS FOR ACCELERATION PHASES ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    vx_diffs = np.diff(vx_profile)
    acc_inds = np.where(vx_diffs > 0.0)[0]                  # indices of points with positive acceleration
    if acc_inds.size != 0:
        # check index diffs -> we only need the first point of every acceleration phase
        acc_inds_diffs = np.diff(acc_inds)
        acc_inds_diffs = np.insert(acc_inds_diffs, 0, 2)    # first point is always a starting point
        acc_inds_rel = acc_inds[acc_inds_diffs > 1]         # starting point indices for acceleration phases
    else:
        acc_inds_rel = []                                   # if vmax is low and can be driven all the time

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE VELOCITY PROFILE ---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # cast np.array as a list
    acc_inds_rel = list(acc_inds_rel)

    # while we have indices remaining in the list
    while acc_inds_rel:
        # set index to first list element
        i = acc_inds_rel.pop(0)

        # start from current index and run until either the end of the lap or a termination criterion are reached
        while i < no_points - 1:

            ax_possible_cur = calc_ax_poss(vx_start=vx_profile[i],
                                           radius=radii_mod[i],
                                           ggv=ggv_mod,
                                           mu=mu_mod[i],
                                           mode=mode,
                                           dyn_model_exp=dyn_model_exp,
                                           drag_coeff=drag_coeff,
                                           m_veh=m_veh)

            vx_possible_next = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])

            if backwards:
                """
                We have to loop the calculation if we are in the backwards iteration (currently just once). This is 
                because we calculate the possible ax at a point i which does not necessarily fit for point i + 1 
                (which is i - 1 in the real direction). At point i + 1 (or i - 1 in real direction) we have a different 
                start velocity (vx_possible_next), radius and mu value while the absolute value of ax remains the same 
                in both directions.
                """

                # looping just once at the moment
                for j in range(1):
                    ax_possible_next = calc_ax_poss(vx_start=vx_possible_next,
                                                    radius=radii_mod[i + 1],
                                                    ggv=ggv_mod,
                                                    mu=mu_mod[i + 1],
                                                    mode=mode,
                                                    dyn_model_exp=dyn_model_exp,
                                                    drag_coeff=drag_coeff,
                                                    m_veh=m_veh)

                    vx_tmp = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_next * el_lengths_mod[i])

                    if vx_tmp < vx_possible_next:
                        vx_possible_next = vx_tmp
                    else:
                        break

            # save possible next velocity if it is smaller than the current value
            if vx_possible_next < vx_profile[i + 1]:
                vx_profile[i + 1] = vx_possible_next

            i += 1

            # break current acceleration phase if next speed would be higher than the maximum vehicle velocity or if we
            # are at the next acceleration phase start index
            if vx_possible_next > vx_max or (acc_inds_rel and i >= acc_inds_rel[0]):
                break

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # flip output vel_profile if necessary
    if backwards:
        vx_profile = np.flipud(vx_profile)

    return vx_profile


def calc_ax_poss(vx_start: float,
                 radius: float,
                 ggv: np.ndarray,
                 mu: float,
                 dyn_model_exp: float,
                 drag_coeff: float,
                 m_veh: float,
                 mode: str = 'accel_forw') -> float:
    """
    This function returns the possible longitudinal acceleration in the current step/point.

    Inputs:
    vx_start:           [m/s] velocity at current point
    radius:             [m] radius on which the car is currently driving
    ggv:                [v_mps, ax_max_machines_mps2, ax_max_tires_mps2, ay_max_tires_mps2] ggv diagram
    mu:                 [-] current friction value
    dyn_model_exp:      [-] exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    drag_coeff:         [m2*kg/m3] drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    m_veh:              [kg] vehicle mass
    mode:               [-] operation mode, can be 'accel_forw', 'decel_forw', 'decel_backw'
                            -> determines if machine limitations are considered and if ax should be considered negative
                            or positive during deceleration (for possible backwards iteration)

    Outputs:
    ax_final:           [m/s2] final acceleration from current point to next one
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check input
    if mode not in ['accel_forw', 'decel_forw', 'decel_backw']:
        raise ValueError("Unknown operation mode for calc_ax_poss!")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER TIRE POTENTIAL ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate possible and used accelerations (considering tires)
    ax_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 2])
    ay_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 3])
    ay_used = math.pow(vx_start, 2) / radius

    # during forward acceleration and backward deceleration ax_max_tires must be considered positive
    if mode in ['accel_forw', 'decel_backw']:
        ax_max_tires = math.fabs(ax_max_tires)
    else:
        ax_max_tires = -math.fabs(ax_max_tires)

    radicand = 1.0 - math.pow(ay_used / ay_max_tires, dyn_model_exp)

    if radicand > 0.0:
        ax_avail_tires = ax_max_tires * math.pow(radicand, 1.0 / dyn_model_exp)
    else:
        ax_avail_tires = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER MACHINE LIMITATIONS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # consider limitations imposed by electrical machines during forward acceleration
    if mode == 'accel_forw':
        # interpolate machine acceleration to be able to consider varying gear ratios, efficiencies etc.
        ax_max_machines = np.interp(vx_start, ggv[:, 0], ggv[:, 1])
        ax_avail_vehicle = min(ax_avail_tires, ax_max_machines)
    else:
        ax_avail_vehicle = ax_avail_tires

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER DRAG ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate equivalent longitudinal acceleration of drag force at the current speed
    ax_drag = -math.pow(vx_start, 2) * drag_coeff / m_veh

    # drag reduces the possible acceleration in the forward case and increases it in the backward case
    if mode in ['accel_forw', 'decel_forw']:
        ax_final = ax_avail_vehicle + ax_drag
        # attention: this value will now be negative in forward direction if tire is entirely used for cornering
    else:
        ax_final = ax_avail_vehicle - ax_drag

    return ax_final


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
