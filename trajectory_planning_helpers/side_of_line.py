import numpy as np


def side_of_line(a: np.ndarray,
                 b: np.ndarray,
                 z: np.ndarray) -> np.ndarray:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Function determines if a point z is on the left or right side of a line from a to b. It is based on the z component
    orientation of the cross product, see question on
    https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line

    Inputs:
    points a, b, z

    Outputs:
    position: 0.0 = on line, 1.0 = left side, -1.0 = right side
    """

    # calculate side
    side = np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]))

    return side


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
