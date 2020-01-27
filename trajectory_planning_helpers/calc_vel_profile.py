import numpy as np
import math
import trajectory_planning_helpers.conv_filt


def calc_vel_profile(ax_max_machines: np.ndarray,
                     kappa: np.ndarray,
                     el_lengths: np.ndarray,
                     closed: bool,
                     drag_coeff: float,
                     m_veh: float,
                     ggv: np.ndarray = None,
                     loc_gg: np.ndarray = None,
                     v_max: float = None,
                     dyn_model_exp: float = 1.0,
                     mu: np.ndarray = None,
                     v_start: float = None,
                     v_end: float = None,
                     filt_window: int = None) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    modified by:
    Tim Stahl

    .. description::
    Calculates a velocity profile using the tire and motor limits as good as possible.

    .. inputs::
    :param ax_max_machines: longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines]. Velocity
                            in m/s, accelerations in m/s2. They should be handed in without considering drag resistance.
    :type ax_max_machines:  np.ndarray
    :param kappa:           curvature profile of given trajectory in rad/m (always unclosed).
    :type kappa:            np.ndarray
    :param el_lengths:      element lengths (distances between coordinates) of given trajectory.
    :type el_lengths:       np.ndarray
    :param closed:          flag to set if the velocity profile must be calculated for a closed or unclosed trajectory.
    :type closed:           bool
    :param drag_coeff:      drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           vehicle mass in kg.
    :type m_veh:            float
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
                            ATTENTION: Insert either ggv + mu (optional) or loc_gg!
    :type ggv:              np.ndarray
    :param loc_gg:          local gg diagrams along the path points: [[ax_max_0, ay_max_0], [ax_max_1, ay_max_1], ...],
                            accelerations in m/s2. ATTENTION: Insert either ggv + mu (optional) or loc_gg!
    :type loc_gg:           np.ndarray
    :param v_max:           Maximum longitudinal speed in m/s (optional if ggv is supplied, taking the minimum of the
                            fastest velocities covered by the ggv and ax_max_machines arrays then).
    :type v_max:            float
    :param dyn_model_exp:   exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param mu:              friction coefficients (always unclosed).
    :type mu:               np.ndarray
    :param v_start:         start velocity in m/s (used in unclosed case only).
    :type v_start:          float
    :param v_end:           end velocity in m/s (used in unclosed case only).
    :type v_end:            float
    :param filt_window:     filter window size for moving average filter (must be odd).
    :type filt_window:      int

    .. outputs::
    :return vx_profile:     calculated velocity profile (always unclosed).
    :rtype vx_profile:      np.ndarray

    .. notes::
    All inputs must be inserted unclosed, i.e. kappa[-1] != kappa[0], even if closed is set True! (el_lengths is kind of
    closed if closed is True of course!)

    case closed is True:
    len(kappa) = len(el_lengths) = len(mu) = len(vx_profile)

    case closed is False:
    len(kappa) = len(el_lengths) + 1 = len(mu) = len(vx_profile)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INPUT CHECKS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check if either ggv (and optionally mu) or loc_gg are handed in
    if (ggv is not None or mu is not None) and loc_gg is not None:
        raise ValueError("Either ggv and optionally mu OR loc_gg must be supplied, not both (or all) of them!")

    if ggv is None and loc_gg is None:
        raise ValueError("Either ggv or loc_gg must be supplied!")

    # check shape of loc_gg
    if loc_gg is not None:
        if loc_gg.ndim != 2:
            raise ValueError("loc_gg must have two dimensions!")

        if loc_gg.shape[0] != kappa.size:
            raise ValueError("Length of loc_gg and kappa must be equal!")

        if loc_gg.shape[1] != 2:
            raise ValueError("loc_gg must consist of two columns: [ax_max, ay_max]!")

    # check shape of ggv
    if ggv is not None and ggv.shape[1] != 3:
        raise ValueError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")

    # check size of mu
    if mu is not None and kappa.size != mu.size:
        raise ValueError("kappa and mu must have the same length!")

    # check size of kappa and element lengths
    if closed and kappa.size != el_lengths.size:
        raise ValueError("kappa and el_lengths must have the same length if closed!")

    elif not closed and kappa.size != el_lengths.size + 1:
        raise ValueError("kappa must have the length of el_lengths + 1 if unclosed!")

    # check start and end velocities
    if not closed and v_start is None:
        raise ValueError("v_start must be provided for the unclosed case!")

    if v_start is not None and v_start < 0.0:
        v_start = 0.0
        print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if v_end is not None and v_end < 0.0:
        v_end = 0.0
        print('WARNING: Input v_end was < 0.0. Using v_end = 0.0 instead!')

    # check dyn_model_exp
    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0, 2.0]!')

    # check shape of ax_max_machines
    if ax_max_machines.shape[1] != 2:
        raise ValueError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

    # check v_max
    if v_max is None:
        if ggv is None:
            raise ValueError("v_max must be supplied if ggv is None!")
        else:
            v_max = min(ggv[-1, 0], ax_max_machines[-1, 0])

    else:
        # check if ggv covers velocity until v_max
        if ggv is not None and ggv[-1, 0] < v_max:
            raise ValueError("ggv has to cover the entire velocity range of the car (i.e. >= v_max)!")

        # check if ax_max_machines covers velocity until v_max
        if ax_max_machines[-1, 0] < v_max:
            raise ValueError("ax_max_machines has to cover the entire velocity range of the car (i.e. >= v_max)!")

    # ------------------------------------------------------------------------------------------------------------------
    # BRINGING GGV OR LOC_GG INTO SHAPE FOR EQUAL HANDLING AFTERWARDS --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """For an equal/easier handling of every case afterwards we bring all cases into a form where the local ggv is made
    available for every waypoint, i.e. [ggv_0, ggv_1, ggv_2, ...] -> we have a three dimensional array p_ggv (path_ggv)
    where the first dimension is the waypoint, the second is the velocity and the third is the two acceleration columns
    -> DIM = NO_WAYPOINTS_CLOSED x NO_VELOCITY ENTRIES x 3"""

    # CASE 1: ggv supplied -> copy it for every waypoint
    if ggv is not None:
        p_ggv = np.repeat(np.expand_dims(ggv, axis=0), kappa.size, axis=0)
        op_mode = 'ggv'

    # CASE 2: local gg diagram supplied -> add velocity dimension (artificial velocity of 10.0 m/s)
    else:
        p_ggv = np.expand_dims(np.column_stack((np.ones(loc_gg.shape[0]) * 10.0, loc_gg)), axis=1)
        op_mode = 'loc_gg'

    # ------------------------------------------------------------------------------------------------------------------
    # SPEED PROFILE CALCULATION (FB) -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1.0, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0.0))

    # call solver
    if not closed:
        vx_profile = __solver_fb_unclosed(p_ggv=p_ggv,
                                          ax_max_machines=ax_max_machines,
                                          v_max=v_max,
                                          radii=radii,
                                          el_lengths=el_lengths,
                                          mu=mu,
                                          v_start=v_start,
                                          v_end=v_end,
                                          dyn_model_exp=dyn_model_exp,
                                          drag_coeff=drag_coeff,
                                          m_veh=m_veh,
                                          op_mode=op_mode)

    else:
        vx_profile = __solver_fb_closed(p_ggv=p_ggv,
                                        ax_max_machines=ax_max_machines,
                                        v_max=v_max,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        mu=mu,
                                        dyn_model_exp=dyn_model_exp,
                                        drag_coeff=drag_coeff,
                                        m_veh=m_veh,
                                        op_mode=op_mode)

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if filt_window is not None:
        vx_profile = trajectory_planning_helpers.conv_filt.conv_filt(signal=vx_profile,
                                                                     filt_window=filt_window,
                                                                     closed=closed)

    return vx_profile


def __solver_fb_unclosed(p_ggv: np.ndarray,
                         ax_max_machines: np.ndarray,
                         v_max: float,
                         radii: np.ndarray,
                         el_lengths: np.ndarray,
                         v_start: float,
                         drag_coeff: float,
                         m_veh: float,
                         op_mode: str,
                         mu: np.ndarray = None,
                         v_end: float = None,
                         dyn_model_exp: float = 1.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # handle mu
    if mu is None:
        mu = np.ones(radii.size)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)

    # run through all the points and check for possible lateral acceleration
    if op_mode == 'ggv':
        # in ggv mode all ggvs are equal -> we can use the first one
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])   # get first lateral acceleration estimate
        vx_profile = np.sqrt(ay_max_global * radii)         # get first velocity profile estimate

        ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
        vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

    else:
        # in loc_gg mode all ggvs consist of a single line due to the missing velocity dependency, mu is None in this
        # case
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)        # get first velocity profile estimate

    # cut vx_profile to car's top speed
    vx_profile[vx_profile > v_max] = v_max

    # consider v_start
    if vx_profile[0] > v_start:
        vx_profile[0] = v_start

    # calculate acceleration profile
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
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
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=True,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)

    return vx_profile


def __solver_fb_closed(p_ggv: np.ndarray,
                       ax_max_machines: np.ndarray,
                       v_max: float,
                       radii: np.ndarray,
                       el_lengths: np.ndarray,
                       drag_coeff: float,
                       m_veh: float,
                       op_mode: str,
                       mu: np.ndarray = None,
                       dyn_model_exp: float = 1.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

    # handle mu
    if mu is None:
        mu = np.ones(no_points)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)

    # run through all the points and check for possible lateral acceleration
    if op_mode == 'ggv':
        # in ggv mode all ggvs are equal -> we can use the first one
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])   # get first lateral acceleration estimate
        vx_profile = np.sqrt(ay_max_global * radii)         # get first velocity estimate (radii must be positive!)

        ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
        vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

    else:
        # in loc_gg mode all ggvs consist of a single line due to the missing velocity dependency, mu is None in this
        # case
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)        # get first velocity estimate (radii must be positive!)

    # cut vx_profile to car's top speed
    vx_profile[vx_profile > v_max] = v_max

    """We need to calculate the speed profile for two laps to get the correct starting and ending velocity."""

    # double arrays
    vx_profile_double = np.concatenate((vx_profile, vx_profile), axis=0)
    radii_double = np.concatenate((radii, radii), axis=0)
    el_lengths_double = np.concatenate((el_lengths, el_lengths), axis=0)
    mu_double = np.concatenate((mu, mu), axis=0)
    p_ggv_double = np.concatenate((p_ggv, p_ggv), axis=0)

    # calculate acceleration profile
    vx_profile_double = __solver_fb_acc_profile(p_ggv=p_ggv_double,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
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
    vx_profile_double = __solver_fb_acc_profile(p_ggv=p_ggv_double,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
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


def __solver_fb_acc_profile(p_ggv: np.ndarray,
                            ax_max_machines: np.ndarray,
                            v_max: float,
                            radii: np.ndarray,
                            el_lengths: np.ndarray,
                            mu: np.ndarray,
                            vx_profile: np.ndarray,
                            drag_coeff: float,
                            m_veh: float,
                            dyn_model_exp: float = 1.0,
                            backwards: bool = False) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = vx_profile.size

    # check for reversed direction
    if backwards:
        radii_mod = np.flipud(radii)
        el_lengths_mod = np.flipud(el_lengths)
        mu_mod = np.flipud(mu)
        vx_profile = np.flipud(vx_profile)
        mode = 'decel_backw'
    else:
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
                                           ggv=p_ggv[i],
                                           ax_max_machines=ax_max_machines,
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
                                                    ggv=p_ggv[i + 1],
                                                    ax_max_machines=ax_max_machines,
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
            if vx_possible_next > v_max or (acc_inds_rel and i >= acc_inds_rel[0]):
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
                 ax_max_machines: np.ndarray = None,
                 mode: str = 'accel_forw') -> float:
    """
    This function returns the possible longitudinal acceleration in the current step/point.

    .. inputs::
    :param vx_start:        [m/s] velocity at current point
    :type vx_start:         float
    :param radius:          [m] radius on which the car is currently driving
    :type radius:           float
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
    :type ggv:              np.ndarray
    :param mu:              [-] current friction value
    :type mu:               float
    :param dyn_model_exp:   [-] exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param drag_coeff:      [m2*kg/m3] drag coefficient incl. all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           [kg] vehicle mass
    :type m_veh:            float
    :param ax_max_machines: longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines]. Velocity
                            in m/s, accelerations in m/s2. They should be handed in without considering drag resistance.
                            Can be set None if using one of the decel modes.
    :type ax_max_machines:  np.ndarray
    :param mode:            [-] operation mode, can be 'accel_forw', 'decel_forw', 'decel_backw'
                            -> determines if machine limitations are considered and if ax should be considered negative
                            or positive during deceleration (for possible backwards iteration)
    :type mode:             str

    .. outputs::
    :return ax_final:       [m/s2] final acceleration from current point to next one
    :rtype ax_final:        float
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check inputs
    if mode not in ['accel_forw', 'decel_forw', 'decel_backw']:
        raise ValueError("Unknown operation mode for calc_ax_poss!")

    if mode == 'accel_forw' and ax_max_machines is None:
        raise ValueError("ax_max_machines is required if operation mode is accel_forw!")

    if ggv.ndim != 2 or ggv.shape[1] != 3:
        raise ValueError("ggv must have two dimensions and three columns [vx, ax_max, ay_max]!")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER TIRE POTENTIAL ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate possible and used accelerations (considering tires)
    ax_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 1])
    ay_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 2])
    ay_used = math.pow(vx_start, 2) / radius

    # during forward acceleration and backward deceleration ax_max_tires must be considered positive, during forward
    # deceleration it must be considered negative
    if mode in ['accel_forw', 'decel_backw'] and ax_max_tires < 0.0:
        print("WARNING: Inverting sign of ax_max_tires because it should be positive but was negative!")
        ax_max_tires *= -1.0
    elif mode == 'decel_forw' and ax_max_tires > 0.0:
        print("WARNING: Inverting sign of ax_max_tires because it should be negative but was positve!")
        ax_max_tires *= -1.0

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
        ax_max_machines_tmp = np.interp(vx_start, ax_max_machines[:, 0], ax_max_machines[:, 1])
        ax_avail_vehicle = min(ax_avail_tires, ax_max_machines_tmp)
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
