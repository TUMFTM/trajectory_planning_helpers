import numpy as np
import math
import trajectory_planning_helpers.normalize_psi


def calc_head_curv_an(coeffs_x: np.ndarray,
                      coeffs_y: np.ndarray,
                      ind_spls: np.ndarray,
                      t_spls: np.ndarray,
                      calc_curv: bool = True,
                      calc_dcurv: bool = False) -> tuple:
    """
    author:
    Alexander Heilmeier
    Marvin Ochsenius (dcurv extension)

    .. description::
    Analytical calculation of heading psi, curvature kappa, and first derivative of the curvature dkappa
    on the basis of third order splines for x- and y-coordinate.

    .. inputs::
    :param coeffs_x:    coefficient matrix of the x splines with size (no_splines x 4).
    :type coeffs_x:     np.ndarray
    :param coeffs_y:    coefficient matrix of the y splines with size (no_splines x 4).
    :type coeffs_y:     np.ndarray
    :param ind_spls:    contains the indices of the splines that hold the points for which we want to calculate heading/curv.
    :type ind_spls:     np.ndarray
    :param t_spls:      containts the relative spline coordinate values (t) of every point on the splines.
    :type t_spls:       np.ndarray
    :param calc_curv:   bool flag to show if curvature should be calculated as well (kappa is set 0.0 otherwise).
    :type calc_curv:    bool
    :param calc_dcurv:  bool flag to show if first derivative of curvature should be calculated as well.
    :type calc_dcurv:   bool

    .. outputs::
    :return psi:        heading at every point.
    :rtype psi:         float
    :return kappa:      curvature at every point.
    :rtype kappa:       float
    :return dkappa:     first derivative of curvature at every point (if calc_dcurv bool flag is True).
    :rtype dkappa:      float

    .. notes::
    len(ind_spls) = len(t_spls) = len(psi) = len(kappa) = len(dkappa)
    """

    # check inputs
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise ValueError("Coefficient matrices must have the same length!")

    if ind_spls.size != t_spls.size:
        raise ValueError("ind_spls and t_spls must have the same length!")

    if not calc_curv and calc_dcurv:
        raise ValueError("dkappa cannot be calculated without kappa!")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE HEADING ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate required derivatives
    x_d = coeffs_x[ind_spls, 1] \
          + 2 * coeffs_x[ind_spls, 2] * t_spls \
          + 3 * coeffs_x[ind_spls, 3] * np.power(t_spls, 2)

    y_d = coeffs_y[ind_spls, 1] \
          + 2 * coeffs_y[ind_spls, 2] * t_spls \
          + 3 * coeffs_y[ind_spls, 3] * np.power(t_spls, 2)

    # calculate heading psi (pi/2 must be substracted due to our convention that psi = 0 is north)
    psi = np.arctan2(y_d, x_d) - math.pi / 2
    psi = trajectory_planning_helpers.normalize_psi.normalize_psi(psi)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE CURVATURE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if calc_curv:
        # calculate required derivatives
        x_dd = 2 * coeffs_x[ind_spls, 2] \
               + 6 * coeffs_x[ind_spls, 3] * t_spls

        y_dd = 2 * coeffs_y[ind_spls, 2] \
               + 6 * coeffs_y[ind_spls, 3] * t_spls

        # calculate curvature kappa
        kappa = (x_d * y_dd - y_d * x_dd) / np.power(np.power(x_d, 2) + np.power(y_d, 2), 1.5)

    else:
        kappa = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE FIRST DERIVATIVE OF CURVATURE --------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if calc_dcurv:
        # calculate required derivatives
        x_ddd = 6 * coeffs_x[ind_spls, 3]

        y_ddd = 6 * coeffs_y[ind_spls, 3]

        # calculate first derivative of curvature dkappa
        dkappa = ((np.power(x_d, 2) + np.power(y_d, 2)) * (x_d * y_ddd - y_d * x_ddd) -
                  3 * (x_d * y_dd - y_d * x_dd) * (x_d * x_dd + y_d * y_dd)) / \
                 np.power(np.power(x_d, 2) + np.power(y_d, 2), 3)

        return psi, kappa, dkappa

    else:

        return psi, kappa


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
