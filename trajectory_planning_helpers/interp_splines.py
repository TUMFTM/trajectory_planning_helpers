import numpy as np
import trajectory_planning_helpers as tph


def interp_splines(coeffs_x: np.ndarray,
                   coeffs_y: np.ndarray,
                   spline_lengths: np.ndarray = None,
                   incl_last_point: bool = False,
                   stepsize_approx: float = 1.0) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Interpolate points on one or more splines with third order. The last point (i.e. t = 0)
    can be included if option is set accordingly. The algorithm keeps stepsize_approx as good as possible.
    ws_track can be inserted optionally and should contain [w_tr_right, w_tr_left].

    Inputs:
    coeffs_x: coefficient matrix of the x splines with size no_splines x 4.
    coeffs_y: coefficient matrix of the y splines with size no_splines x 4.
    spline_lengths: array containing the lengths of the inserted splines with size no_splines x 1.
    incl_last_point: flag to set if last point should be kept or removed before return.
    stepsize_approx: desired stepsize of the points after interpolation.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check if coeffs_x and coeffs_y have exactly two dimensions and raise error otherwise
    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise ValueError("Coefficient matrices do not have two dimensions!")

    # get the total distance up to the end of every spline (i.e. cumulated distances)
    if spline_lengths is None:
        spline_lengths = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x,
                                                                     coeffs_y=coeffs_y,
                                                                     quickndirty=False)

    dists_cum = np.cumsum(spline_lengths)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate number of interpolation points and distances (+1 because last point is included at first)
    no_interp_points = (np.ceil(dists_cum[-1] / stepsize_approx)).astype(int) + 1
    dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

    # create arrays to save the values
    raceline_interp = np.zeros((no_interp_points, 2))       # raceline coords (x, y) array
    spline_inds = np.zeros(no_interp_points, dtype=int)     # save the spline index to which a point belongs
    t_values = np.zeros(no_interp_points)                   # save t values

    # loop through all the elements and create steps with stepsize_approx
    j = 0

    for i in range(no_interp_points - 1):
        # find the spline that hosts the current interpolation point
        j = np.argmax(dists_interp[i] < dists_cum)
        spline_inds[i] = j

        # get spline t value depending on the progress within the current element
        if j > 0:
            t_values[i] = (dists_interp[i] - dists_cum[j - 1]) / spline_lengths[j]
        else:
            if spline_lengths.ndim == 0:
                t_values[i] = dists_interp[i] / spline_lengths
            else:
                t_values[i] = dists_interp[i] / spline_lengths[0]

        # calculate coords
        raceline_interp[i, 0] = coeffs_x[j, 0]\
                                + coeffs_x[j, 1] * t_values[i]\
                                + coeffs_x[j, 2] * np.power(t_values[i], 2) \
                                + coeffs_x[j, 3] * np.power(t_values[i], 3)

        raceline_interp[i, 1] = coeffs_y[j, 0]\
                                + coeffs_y[j, 1] * t_values[i]\
                                + coeffs_y[j, 2] * np.power(t_values[i], 2) \
                                + coeffs_y[j, 3] * np.power(t_values[i], 3)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE LAST POINT IF REQUIRED (t = 1.0) -----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if incl_last_point:
        t_values[-1] = 1.0

        raceline_interp[-1, 0] = coeffs_x[j, 0] + coeffs_x[j, 1] + coeffs_x[j, 2] + coeffs_x[j, 3]
        raceline_interp[-1, 1] = coeffs_y[j, 0] + coeffs_y[j, 1] + coeffs_y[j, 2] + coeffs_y[j, 3]

    else:
        raceline_interp = raceline_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]
        dists_interp = dists_interp[:-1]

    return raceline_interp, spline_inds, t_values, dists_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
