import numpy as np


def calc_vel_profile(ggv: np.ndarray, kappa: np.ndarray, el_lengths: np.ndarray, closed: bool,
                     mu: np.ndarray = None, v_start: float = None, v_end: float = None,
                     filt_window: int = None, tire_model_exp: float = 2.0) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Documentation:
    Calculates a velocity profile using the tire and motor limits as good as possible. This function is for

    Inputs:
    ggv:                ggv-diagram to be applied
    kappa:              curvature profile of given trajectory in rad/m
    el_lengths:         element lengths (distances between coordinates) of given trajectory
    closed:             flag to set if the velocity profile must be calculated for a closed or unclosed trajectory
    mu:                 friction coefficients
    v_start:            start velocity in m/s (used in unclosed case only)
    v_end:              end velocity in m/s (used in unclosed case only)
    filt_window:        filter window size for moving average filter (must be odd)
    tire_model_exp:     exponent used in the dynamics model

    case closed:
    len(kappa) = len(el_lengths) = len(mu)

    case unclosed:
    len(kappa) = len(el_lengths) + 1 = len(mu)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SPEED PROFILE CALCULATION (FB) -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0))

    # set mu to one in case it is not set
    if mu is None:
        mu = np.ones(kappa.size)

    # call solver
    if not closed:
        vx_profile = solver_fb_unclosed(ggv=ggv,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        mu=mu,
                                        v_start=v_start,
                                        v_end=v_end,
                                        tire_model_exp=tire_model_exp)

    else:
        vx_profile = solver_fb_closed(ggv=ggv,
                                      radii=radii,
                                      el_lengths=el_lengths,
                                      mu=mu,
                                      tire_model_exp=tire_model_exp)

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # convolution filter used as a moving average filter
    if filt_window is not None:

        # check if window width is odd
        if not filt_window % 2 == 1:
            raise IOError("Window width of moving average filter in velocity profile generation must be odd!")

        # apply filter
        vx_profile = np.convolve(vx_profile,
                                 np.ones(filt_window) / float(filt_window),
                                 mode="same")

    return vx_profile


def solver_fb_unclosed(ggv: np.ndarray, radii: np.ndarray, el_lengths: np.ndarray, mu: np.ndarray, v_start: float,
                       v_end: float, tire_model_exp: float = 2.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

    # run through all the points and check for possible lateral acceleration
    mu_mean = np.mean(mu)
    ay_max_global = mu_mean * np.amin(np.abs(ggv[:, 4]))    # get first lateral acceleration estimate
    vx_profile = np.sqrt(ay_max_global * radii)             # get first velocity profile estimate

    for i in range(no_points):
        ay_max_curr = mu[i] * np.interp(vx_profile[i], ggv[:, 0], ggv[:, 4])
        vx_profile[i] = np.sqrt(ay_max_curr * radii[i])

    # cut vx_profile to car's top speed
    vx_max = ggv[-1, 0]
    vx_profile[vx_profile > vx_max] = vx_max

    # consider v_start
    if vx_profile[0] > v_start:
        vx_profile[0] = v_start

    # calculate acceleration profile
    vx_profile = solver_fb_acc_profile(ggv=ggv,
                                       radii=radii,
                                       el_lengths=el_lengths,
                                       mu=mu,
                                       vx_profile=vx_profile,
                                       rev_dir=False,
                                       tire_model_exp=tire_model_exp)

    # consider v_end
    if vx_profile[-1] > v_end:
        vx_profile[-1] = v_end

    # calculate deceleration profile
    vx_profile = solver_fb_acc_profile(ggv=ggv,
                                       radii=radii,
                                       el_lengths=el_lengths,
                                       mu=mu,
                                       vx_profile=vx_profile,
                                       rev_dir=True,
                                       tire_model_exp=tire_model_exp)

    return vx_profile


def solver_fb_closed(ggv: np.ndarray, radii: np.ndarray, el_lengths: np.ndarray, mu: np.ndarray,
                     tire_model_exp: float = 2.0) -> np.ndarray:

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

    # run through all the points and check for possible lateral acceleration
    mu_mean = np.mean(mu)
    ay_max_global = mu_mean * np.amin(np.abs(ggv[:, 4]))    # get first lateral acceleration estimate
    vx_profile = np.sqrt(ay_max_global * radii)             # get first velocity estimate (radii must be positive!)

    for i in range(no_points):
        # do it two times to improve accuracy (because of velocity-dependent accelerations)

        for j in range(2):
            # get proper possible lateral acceleration for velocity estimate
            ay_max_curr = mu[i] * np.interp(vx_profile[i], ggv[:, 0], ggv[:, 4])
            vx_profile[i] = np.sqrt(ay_max_curr * radii[i])

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
    vx_profile_double = solver_fb_acc_profile(ggv=ggv,
                                              radii=radii_double,
                                              el_lengths=el_lengths_double,
                                              mu=mu_double,
                                              vx_profile=vx_profile_double,
                                              tire_model_exp=tire_model_exp,
                                              rev_dir=False)

    # use second lap of acceleration profile
    vx_profile_double = np.concatenate((vx_profile_double[no_points:], vx_profile_double[no_points:]), axis=0)

    # calculate deceleration profile
    vx_profile_double = solver_fb_acc_profile(ggv=ggv,
                                              radii=radii_double,
                                              el_lengths=el_lengths_double,
                                              mu=mu_double,
                                              vx_profile=vx_profile_double,
                                              tire_model_exp=tire_model_exp,
                                              rev_dir=True)

    # use second lap of deceleration profile
    vx_profile = vx_profile_double[no_points:]

    return vx_profile


def solver_fb_acc_profile(ggv: np.ndarray, radii: np.ndarray, el_lengths: np.ndarray, mu: np.ndarray,
                          vx_profile: np.ndarray, rev_dir: bool = False, tire_model_exp: float = 2.0) -> np.ndarray:

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
            # consider that mu > 1.0 does not scale positive longitudinal acceleration
            mu_ax = mu_mod[i]

            if not rev_dir and mu_ax > 1.0:
                mu_ax = 1.0

            # calculate possible and used accelerations (considering tires)
            ax_max_cur_tires = mu_ax * np.interp(vx_profile[i], ggv_mod[:, 0], ggv_mod[:, 2])
            ay_max_cur_tires = mu_mod[i] * np.interp(vx_profile[i], ggv_mod[:, 0], ggv_mod[:, 4])
            ay_used_cur = np.power(vx_profile[i], 2) / radii_mod[i]

            radicand = 1 - np.power(ay_used_cur / ay_max_cur_tires, tire_model_exp)

            if radicand > 0.0:
                ax_possible_cur_tires = ax_max_cur_tires * np.power(radicand, 1.0 / tire_model_exp)
            else:
                ax_possible_cur_tires = 0.0

            # consider limitation imposed by electrical machines in positive acceleration (only forward direction)
            ax_max_cur_machines = np.interp(vx_profile[i], ggv_mod[:, 0], ggv_mod[:, 1])

            if not rev_dir:
                ax_possible_cur = min(ax_max_cur_machines, ax_possible_cur_tires)
            else:
                ax_possible_cur = ax_possible_cur_tires

            vx_possible_next = np.sqrt(np.power(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])

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


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
