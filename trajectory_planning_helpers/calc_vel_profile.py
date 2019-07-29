import numpy as np
import math
import trajectory_planning_helpers.conv_filt
import warnings


def calc_vel_profile(ggv: np.ndarray,
                     kappa: np.ndarray,
                     el_lengths: np.ndarray,
                     closed: bool,
                     mu: np.ndarray = None,
                     v_start: float = None,
                     v_end: float = None,
                     filt_window: int = None,
                     dyn_model_exp: float = 1.0) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Documentation:
    Calculates a velocity profile using the tire and motor limits as good as possible. This function is for

    Inputs:
    ggv:                ggv-diagram to be applied.
    kappa:              curvature profile of given trajectory in rad/m (always unclosed).
    el_lengths:         element lengths (distances between coordinates) of given trajectory.
    closed:             flag to set if the velocity profile must be calculated for a closed or unclosed trajectory.
    mu:                 friction coefficients (always unclosed).
    v_start:            start velocity in m/s (used in unclosed case only).
    v_end:              end velocity in m/s (used in unclosed case only).
    filt_window:        filter window size for moving average filter (must be odd).
    dyn_model_exp:      exponent used in the vehicle dynamics model (usual range [1.0,2.0]).

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
        warnings.warn('Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if v_end is not None and v_end < 0.0:
        v_end = 0.0
        warnings.warn('Input v_end was < 0.0. Using v_end = 0.0 instead!')

    if not 1.0 <= dyn_model_exp <= 2.0:
        warnings.warn('Exponent for the vehicle dynamics model should be in the range [1.0,2.0]!')

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
                                          dyn_model_exp=dyn_model_exp)

    else:
        vx_profile = __solver_fb_closed(ggv=ggv,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        mu=mu,
                                        dyn_model_exp=dyn_model_exp)

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if filt_window is not None:
        vx_profile = trajectory_planning_helpers.conv_filt.conv_filt(signal=vx_profile,
                                                                     filt_window=filt_window,
                                                                     closed=closed)

    return vx_profile


def __solver_fb_unclosed(ggv: np.ndarray, radii: np.ndarray, el_lengths: np.ndarray, mu: np.ndarray, v_start: float,
                         v_end: float = None, dyn_model_exp: float = 1.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

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
                                         rev_dir=False,
                                         dyn_model_exp=dyn_model_exp)

    # consider v_end
    if v_end is not None and vx_profile[-1] > v_end:
        vx_profile[-1] = v_end

    # calculate deceleration profile
    vx_profile = __solver_fb_acc_profile(ggv=ggv,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         rev_dir=True,
                                         dyn_model_exp=dyn_model_exp)

    return vx_profile


def __solver_fb_closed(ggv: np.ndarray, radii: np.ndarray, el_lengths: np.ndarray, mu: np.ndarray,
                       dyn_model_exp: float = 1.0) -> np.ndarray:

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
                                                dyn_model_exp=dyn_model_exp,
                                                rev_dir=False)

    # use second lap of acceleration profile
    vx_profile_double = np.concatenate((vx_profile_double[no_points:], vx_profile_double[no_points:]), axis=0)

    # calculate deceleration profile
    vx_profile_double = __solver_fb_acc_profile(ggv=ggv,
                                                radii=radii_double,
                                                el_lengths=el_lengths_double,
                                                mu=mu_double,
                                                vx_profile=vx_profile_double,
                                                dyn_model_exp=dyn_model_exp,
                                                rev_dir=True)

    # use second lap of deceleration profile
    vx_profile = vx_profile_double[no_points:]

    return vx_profile


def __solver_fb_acc_profile(ggv: np.ndarray, radii: np.ndarray, el_lengths: np.ndarray, mu: np.ndarray,
                            vx_profile: np.ndarray, rev_dir: bool = False, dyn_model_exp: float = 1.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    vx_max = ggv[-1, 0]
    no_points = vx_profile.size

    # check for reverse direction -> modify ggv (exchange second and third column), vx_profile, radii and el_lengths
    if rev_dir:
        ggv_mod = np.copy(ggv)
        ggv_mod[:, [2, 3]] = ggv_mod[:, [3, 2]]
        ggv_mod[:, 2] = np.abs(ggv_mod[:, 2])
        ggv_mod[:, 3] = -np.abs(ggv_mod[:, 3])

        radii_mod = np.flipud(radii)
        el_lengths_mod = np.flipud(el_lengths)
        mu_mod = np.flipud(mu)
        vx_profile = np.flipud(vx_profile)
    else:
        ggv_mod = ggv
        radii_mod = radii
        el_lengths_mod = el_lengths
        mu_mod = mu

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

            ax_possible_cur = __calc_ax_poss(vx_start=vx_profile[i],
                                             radius=radii_mod[i],
                                             ggv=ggv_mod,
                                             mu=mu_mod[i],
                                             rev_dir=rev_dir,
                                             dyn_model_exp=dyn_model_exp)

            vx_possible_next = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])

            if rev_dir:
                """
                We have to loop the calculation if we are in the backwards iteration (currently just once). This is 
                because we calculate the possible ax at a point i which does not necessarily fit for point i + 1 
                (which is i - 1 in the real direction). At point i + 1 (or i - 1 in real direction) we have a different 
                start velocity (vx_possible_next), radius and mu value while the absolute value of ax remains the same 
                in both directions.
                """

                # looping just once at the moment
                for j in range(1):
                    ax_possible_next = __calc_ax_poss(vx_start=vx_possible_next,
                                                      radius=radii_mod[i + 1],
                                                      ggv=ggv_mod,
                                                      mu=mu_mod[i + 1],
                                                      rev_dir=rev_dir,
                                                      dyn_model_exp=dyn_model_exp)

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
    if rev_dir:
        vx_profile = np.flipud(vx_profile)

    return vx_profile


def __calc_ax_poss(vx_start: float, radius: float, ggv: np.ndarray, mu: float, rev_dir: bool,
                   dyn_model_exp: float = 1.0) -> float:
    """This function returns the possible longitudinal acceleration in the current step/point."""

    # consider that mu > 1.0 does not scale positive longitudinal acceleration
    mu_ax = mu

    if not rev_dir and mu_ax > 1.0:
        mu_ax = 1.0

    # calculate possible and used accelerations (considering tires)
    ax_max_cur_tires = mu_ax * np.interp(vx_start, ggv[:, 0], ggv[:, 2])
    ay_max_cur_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 4])
    ay_used_cur = math.pow(vx_start, 2) / radius

    radicand = 1 - math.pow(ay_used_cur / ay_max_cur_tires, dyn_model_exp)

    if radicand > 0.0:
        ax_possible_cur_tires = ax_max_cur_tires * math.pow(radicand, 1.0 / dyn_model_exp)
    else:
        ax_possible_cur_tires = 0.0

    # consider limitation imposed by electrical machines in positive acceleration (only forward direction)
    if not rev_dir:
        ax_max_cur_machines = np.interp(vx_start, ggv[:, 0], ggv[:, 1])
        ax_possible_cur = min(ax_max_cur_machines, ax_possible_cur_tires)
    else:
        ax_possible_cur = ax_possible_cur_tires

    return ax_possible_cur


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
