import numpy as np
import trajectory_planning_helpers.normalize_psi


def calc_normal_vectors(psi_vel: np.ndarray) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Use heading to provide normalized normal vectors.

    Inputs:
    psi_vel: array containing the heading of every point (north up, [-pi; pi[)
    """

    # remove second dimension
    if psi_vel.ndim == 2:
        psi_vel_ = np.squeeze(psi_vel)
    else:
        psi_vel_ = np.copy(psi_vel)

    # remap psi_vel to x-axis
    psi_vel_ -= np.pi / 2
    psi_vel_ = trajectory_planning_helpers.normalize_psi.normalize_psi(psi_vel_)

    # inverse atan2
    tangvec = np.ones((psi_vel_.size, 2))

    for i in range(psi_vel_.size):
        if -np.pi/2 < psi_vel_[i] < np.pi/2:
            tangvec[i, 1] = np.tan(psi_vel_[i])  # x is set 1, y is set accordingly
        elif np.isclose(psi_vel_[i], np.pi/2):
            tangvec[i, 0] = 0.0
            tangvec[i, 1] = 1.0
        elif np.isclose(psi_vel_[i], -np.pi/2):
            tangvec[i, 0] = 0.0
            tangvec[i, 1] = -1.0
        else:
            tangvec[i, 0] = -1.0
            tangvec[i, 1] = -np.tan(psi_vel_[i])  # x is set -1, y is set accordingly

    # normalize tangent vector
    lengths = 1 / np.sqrt(np.power(tangvec[:, 0], 2) + np.power(tangvec[:, 1], 2))
    tangvec_normalized = tangvec * lengths[:, np.newaxis]

    # find normal vector
    normvec_normalized = np.stack((-tangvec_normalized[:, 1], tangvec_normalized[:, 0]), axis=1)

    return normvec_normalized


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
