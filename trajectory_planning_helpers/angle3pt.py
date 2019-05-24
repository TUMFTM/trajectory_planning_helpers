import math
from typing import Union
import numpy as np


def angle3pt(a: Union[tuple, np.ndarray], b: Union[tuple, np.ndarray], c: Union[tuple, np.ndarray]) -> float:
    """
    Created by:
    Tim Stahl

    Documentation: Calculates angle by turning from a to c around b.

    Inputs:
    points coordinates a, b, c.
    """

    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

    if ang > math.pi:
        ang -= 2 * math.pi
    elif ang <= -math.pi:
        ang += 2 * math.pi

    return ang


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
