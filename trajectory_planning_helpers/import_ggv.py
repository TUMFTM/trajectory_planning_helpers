import numpy as np


def import_ggv(ggv_import_path: str) -> np.ndarray:
    """
    Author:
    Alexander Heilmeier

    Description:
    This function imports our vehicle ggv diagram consisting of the columns
    [vx, ax_max_emotors, ax_max_tires, ay_max_tires]

    Inputs:
    ggv_import_path:    Path to the ggv csv file.

    Outputs:
    ggv:                ggv diagram.
    """

    # load ggv
    with open(ggv_import_path, "rb") as fh:
        ggv = np.loadtxt(fh, delimiter=",")

    # expand dimension in case of a single row
    if ggv.ndim == 1:
        ggv = np.expand_dims(ggv, 0)

    # check dimensions of ggv diagram
    if ggv.shape[1] != 4:
        raise ValueError("ggv diagram must consist of the four columns [vx, ax_max_emotors, ax_max_tires,"
                         " ay_max_tires]!")

    # check ggv
    invalid_1 = ggv[:, 0] < 0.0        # assure velocities > 0.0
    invalid_2 = ggv[:, 1:] > 20.0      # assure valid maximum accelerations
    invalid_3 = ggv[:, 1] < 0.0        # assure positive accelerations
    invalid_4 = ggv[:, 2] < 0.0        # assure positive accelerations
    invalid_5 = ggv[:, 3] < 0.0        # assure positive accelerations

    if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3) or np.any(invalid_4) or np.any(invalid_5):
        raise ValueError("ggv diagram seems unreasonable!")

    return ggv


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
