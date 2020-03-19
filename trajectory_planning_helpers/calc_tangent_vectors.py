import numpy as np
import math
import trajectory_planning_helpers.normalize_psi


def calc_tangent_vectors(psi: np.ndarray) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    Use heading to calculate normalized (i.e. unit length) tangent vectors.

    .. inputs::
    :param psi:                     array containing the heading of every point (north up, range [-pi,pi[).
    :type psi:                      np.ndarray

    .. outputs::
    :return tangvec_normalized:     unit length tangent vectors for every point [x, y].
    :rtype tangvec_normalized:      np.ndarray

    .. notes::
    len(psi) = len(tangvec_normalized)
    """

    psi_ = np.copy(psi)

    # remap psi_vel to x-axis
    psi_ += math.pi / 2
    psi_ = trajectory_planning_helpers.normalize_psi.normalize_psi(psi_)

    # get normalized tangent vectors
    tangvec_normalized = np.zeros((psi_.size, 2))
    tangvec_normalized[:, 0] = np.cos(psi_)
    tangvec_normalized[:, 1] = np.sin(psi_)

    return tangvec_normalized


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    psi_test = np.array([0.0, math.pi/4, math.pi/2, math.pi, -math.pi, -math.pi/2])
    print("Result:\n", calc_tangent_vectors(psi=psi_test))
