import numpy as np
import trajectory_planning_helpers as tph


def iqp_handler(reftrack: np.ndarray,
                normvectors: np.ndarray,
                A: np.ndarray,
                kappa_bound: float,
                w_veh: float,
                print_debug: bool,
                plot_debug: bool,
                stepsize_interp: float,
                iters_min: int = 3,
                curv_error_allowed: float = 0.01) -> tuple:

    """
    author:
    Alexander Heilmeier

    .. description::
    This function handles the iterative call of the quadratic optimization problem (minimum curvature) during
    trajectory optimization. The interface to this function was kept as similar as possible to the interface of
    opt_min_curv.py.

    The basic idea is to repeatedly call the minimum curvature optimization while we limit restrict the solution space
    for an improved validity (the linearization for the optimization problems is the problem here). After every step
    we update the reference track on the basis of the solution for the next iteration to increase the validity of the
    linearization. Since the optimization problem is based on the assumption of equal stepsizes we have to interpolate
    the track in every iteration.

    Please refer to our paper for further information:
    Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann
    Minimum Curvature Trajectory Planning and Control for an Autonomous Racecar
    DOI: 10.1080/00423114.2019.1631455

    .. inputs::
    :param reftrack:            array containing the reference track, i.e. a reference line and the according track
                                widths to the right and to the left [x, y, w_tr_right, w_tr_left] (unit is meter, must
                                be unclosed!)
    :type reftrack:             np.ndarray
    :param normvectors:         normalized normal vectors for every point of the reference track [x, y]
                                (unit is meter, must be unclosed!)
    :type normvectors:          np.ndarray
    :param A:                   linear equation system matrix for splines (applicable for both, x and y direction)
                                -> System matrices have the form a_i, b_i * t, c_i * t^2, d_i * t^3
                                -> see calc_splines.py for further information or to obtain this matrix
    :type A:                    np.ndarray
    :param kappa_bound:         curvature boundary to consider during optimization.
    :type kappa_bound:          float
    :param w_veh:               vehicle width in m. It is considered during the calculation of the allowed deviations
                                from the reference line.
    :type w_veh:                float
    :param print_debug:         bool flag to print debug messages.
    :type print_debug:          bool
    :param plot_debug:          bool flag to plot the curvatures that are calculated based on the original linearization
                                and on a linearization around the solution.
    :type plot_debug:           bool
    :param stepsize_interp:     stepsize in meters which is used for an interpolation after the spline approximation.
                                This stepsize determines the steps within the optimization problem.
    :type stepsize_interp:      float
    :param iters_min:           number if minimum iterations of the IQP (termination criterion).
    :type iters_min:            int
    :param curv_error_allowed:  allowed curvature error in rad/m between the original linearization and the
                                linearization around the solution (termination criterion).
    :type curv_error_allowed:   float

    .. outputs::
    :return alpha_mincurv_tmp:  solution vector of the optimization problem containing the lateral shift in m for every
                                point.
    :rtype alpha_mincurv_tmp:   np.ndarray
    :return reftrack_tmp:       reference track data [x, y, w_tr_right, w_tr_left] as it was used in the final iteration
                                of the IQP.
    :rtype reftrack_tmp:        np.ndarray
    :return normvectors_tmp:    normalized normal vectors as they were used in the final iteration of the IQP [x, y].
    :rtype normvectors_tmp:     np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # IQP (ITERATIVE QUADRATIC PROGRAMMING) ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set initial data
    reftrack_tmp = reftrack
    normvectors_tmp = normvectors
    A_tmp = A

    # loop
    iter_cur = 0

    while True:
        iter_cur += 1

        # calculate intermediate solution and catch sum of squared curvature errors
        alpha_mincurv_tmp, curv_error_max_tmp = tph.opt_min_curv.\
            opt_min_curv(reftrack=reftrack_tmp,
                         normvectors=normvectors_tmp,
                         A=A_tmp,
                         kappa_bound=kappa_bound,
                         w_veh=w_veh,
                         print_debug=print_debug,
                         plot_debug=plot_debug)

        # print some progress information
        if print_debug:
            print("Minimum curvature IQP: iteration %i, curv_error_max: %.4frad/m" % (iter_cur, curv_error_max_tmp))

        # restrict solution space to improve validity of the linearization during the first steps
        if iter_cur < iters_min:
            alpha_mincurv_tmp *= iter_cur * 1.0 / iters_min

        # check termination criterions: minimum number of iterations and curvature error
        if iter_cur >= iters_min and curv_error_max_tmp <= curv_error_allowed:
            if print_debug:
                print("Finished IQP!")
            break

        # --------------------------------------------------------------------------------------------------------------
        # INTERPOLATION FOR EQUAL STEPSIZES ----------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        refline_tmp, _, _, _, spline_inds_tmp, t_values_tmp = tph.create_raceline.\
            create_raceline(refline=reftrack_tmp[:, :2],
                            normvectors=normvectors_tmp,
                            alpha=alpha_mincurv_tmp,
                            stepsize_interp=stepsize_interp)[:6]

        # calculate new track boundaries on the basis of the intermediate alpha values and interpolate them accordingly
        reftrack_tmp[:, 2] -= alpha_mincurv_tmp
        reftrack_tmp[:, 3] += alpha_mincurv_tmp

        ws_track_tmp = tph.interp_track_widths.interp_track_widths(w_track=reftrack_tmp[:, 2:4],
                                                                   spline_inds=spline_inds_tmp,
                                                                   t_values=t_values_tmp,
                                                                   incl_last_point=False)

        # create new reftrack
        reftrack_tmp = np.column_stack((refline_tmp, ws_track_tmp))

        # --------------------------------------------------------------------------------------------------------------
        # CALCULATE NEW SPLINES ON THE BASIS OF THE INTERPOLATED REFERENCE TRACK ---------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate new splines
        refline_tmp_cl = np.vstack((reftrack_tmp[:, :2], reftrack_tmp[0, :2]))

        coeffs_x_tmp, coeffs_y_tmp, A_tmp, normvectors_tmp = tph.calc_splines.\
            calc_splines(path=refline_tmp_cl,
                         use_dist_scaling=False)

    return alpha_mincurv_tmp, reftrack_tmp, normvectors_tmp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
