import numpy as np


def import_veh_dyn_info(ggv_import_path: str = None,
                        ax_max_machines_import_path: str = None) -> tuple:
    """
    author:
    Alexander Heilmeier

    .. description::
    This function imports the required vehicle dynamics information from several files: The vehicle ggv diagram
    ([vx, ax_max, ay_max], velocity in m/s, accelerations in m/s2) and the ax_max_machines array containing the
    longitudinal acceleration limits by the electrical motors ([vx, ax_max_machines], velocity in m/s, acceleration in
    m/s2).

    .. inputs::
    :param ggv_import_path:             Path to the ggv csv file.
    :type ggv_import_path:              str
    :param ax_max_machines_import_path: Path to the ax_max_machines csv file.
    :type ax_max_machines_import_path:  str

    .. outputs::
    :return ggv:                        ggv diagram
    :rtype ggv:                         np.ndarray
    :return ax_max_machines:            ax_max_machines array
    :rtype ax_max_machines:             np.ndarray
    """

    # GGV --------------------------------------------------------------------------------------------------------------
    if ggv_import_path is not None:

        # load csv
        with open(ggv_import_path, "rb") as fh:
            ggv = np.loadtxt(fh, comments='#', delimiter=",")

        # expand dimension in case of a single row
        if ggv.ndim == 1:
            ggv = np.expand_dims(ggv, 0)

        # check columns
        if ggv.shape[1] != 3:
            raise ValueError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")

        # check values
        invalid_1 = ggv[:, 0] < 0.0     # assure velocities > 0.0
        invalid_2 = ggv[:, 1:] > 20.0   # assure valid maximum accelerations
        invalid_3 = ggv[:, 1] < 0.0     # assure positive accelerations
        invalid_4 = ggv[:, 2] < 0.0     # assure positive accelerations

        if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3) or np.any(invalid_4):
            raise ValueError("ggv seems unreasonable!")

    else:
        ggv = None

    # AX_MAX_MACHINES --------------------------------------------------------------------------------------------------
    if ax_max_machines_import_path is not None:

        # load csv
        with open(ax_max_machines_import_path, "rb") as fh:
            ax_max_machines = np.loadtxt(fh, comments='#',  delimiter=",")

        # expand dimension in case of a single row
        if ax_max_machines.ndim == 1:
            ax_max_machines = np.expand_dims(ax_max_machines, 0)

        # check columns
        if ax_max_machines.shape[1] != 2:
            raise ValueError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

        # check values
        invalid_1 = ax_max_machines[:, 0] < 0.0     # assure velocities > 0.0
        invalid_2 = ax_max_machines[:, 1] > 20.0    # assure valid maximum accelerations
        invalid_3 = ax_max_machines[:, 1] < 0.0     # assure positive accelerations

        if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3):
            raise ValueError("ax_max_machines seems unreasonable!")

    else:
        ax_max_machines = None

    return ggv, ax_max_machines


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
