import numpy as np


def calc_splines(path: np.ndarray) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Calculate third order splines for x- and y-coordinates of a given path. Coefficient matrices have the form
    a_i, b_i * t, c_i * t^2, d_i * t^3.

    Inputs:
    path: array containing the points of the path. IMPORTANT: the path is assumed to be closed by stacking id 0 after
    the last point!
    """

    # close path
    path_cl = np.vstack((path, path[0]))

    # save number of points without closure
    no_points_uncl = path.shape[0]

    # set up linear equation systems (format a_x * x_les = b_x) with x_les containing the desired spline parameters
    # *4 because of 4 parameters in cubic spline
    a = np.zeros((no_points_uncl * 4, no_points_uncl * 4))
    b_x = np.zeros((no_points_uncl * 4, 1))
    b_y = np.zeros((no_points_uncl * 4, 1))

    # create template for a array entries
    template_a = np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0],      # a_i = x_i
                 [1, 1, 1, 1, 0, 0, 0, 0],      # a_i + b_i + c_i + d_i = x_i+1
                 [0, 1, 2, 3, 0, -1, 0, 0],     # b_i + 2c_i + 3d_i - b_i+1 = 0
                 [0, 0, 2, 6, 0, 0, -2, 0]])    # 2c_i + 6d_i - 2c_i+1 = 0)

    for i in range(no_points_uncl):
        j = i * 4

        if i != no_points_uncl - 1:
            a[j: j + 4, j: j + 8] = template_a
            b_x[j: j + 4] = \
                [[path_cl[i, 0]],
                 [path_cl[i + 1, 0]],
                 [0],
                 [0]]
            b_y[j: j + 4] = \
                [[path_cl[i, 1]],
                 [path_cl[i + 1, 1]],
                 [0],
                 [0]]
        else:  # the rest is considered afterwards (boundary conditions)
            a[j: j + 2, j: j + 4] = \
                [[1, 0, 0, 0],
                 [1, 1, 1, 1]]
            b_x[j: j + 2] = \
                [[path_cl[i, 0]],
                 [path_cl[i + 1, 0]]]
            b_y[j: j + 2] = \
                [[path_cl[i, 1]],
                 [path_cl[i + 1, 1]]]

    # gradient boundary conditions (for a closed spline)
    a[-2, 1] = 1
    a[-2, -3:] = [-1, -2, -3]
    # b_x[-2] = 0
    # b_y[-2] = 0

    # curvature boundary conditions (for a closed spline)
    a[-1, 2] = 2
    a[-1, -2:] = [-2, -6]
    # b_x[-1] = 0
    # b_y[-1] = 0

    # call solver
    x_les = np.squeeze(np.linalg.solve(a, b_x))  # squeeze removes single-dimensional entries
    y_les = np.squeeze(np.linalg.solve(a, b_y))

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_points_uncl, 4))
    coeffs_y = np.reshape(y_les, (no_points_uncl, 4))

    # get normal vector (second coefficient of cubic splines is relevant for the gradient)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    norm_factors = np.expand_dims(norm_factors, axis=1)  # second dimension must be inserted for next step
    normvec_normalized = norm_factors * normvec

    return normvec_normalized, a, coeffs_x, coeffs_y


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
