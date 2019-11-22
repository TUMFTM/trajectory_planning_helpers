import numpy as np


def import_ggv(ggv_import_path: str) -> np.ndarray:
    """
    Author:
    Alexander Heilmeier

    Description:
    This function imports our vehicle ggv diagram consisting of the columns
    [vx, ax_max_emotors, ax_max_tires, ax_min_tires, ay_max_tires]

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

    # check ggv
    bool_1 = np.abs(ggv[:, 1:]) > 20.0
    bool_2 = ggv[:, 1] < 0.0
    bool_3 = ggv[:, 2] < 0.0
    bool_4 = ggv[:, 3] > 0.0
    bool_5 = ggv[:, 4] < 0.0

    if np.any(bool_1) or np.any(bool_2) or np.any(bool_3) or np.any(bool_4) or np.any(bool_5):
        raise ValueError("ggv diagram seems unreasonable!")

    return ggv


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
