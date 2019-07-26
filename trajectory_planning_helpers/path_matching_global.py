import numpy as np
import trajectory_planning_helpers.path_matching_local
import typing


def path_matching_global(path_cl: np.ndarray,
                         ego_position: np.ndarray,
                         s_expected: typing.Union[float, None] = None,
                         s_range: float = 20.0) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    Get the corresponding s coordinate and the displacement of the own vehicle in relation to the global path.

    Inputs:
    path_cl:        Closed path used to match ego position ([s, x, y]).
    ego_position:   Ego position of the vehicle ([x, y]).
    s_expected:     Expected s position of the vehicle in m.
    s_range:        Range around expected s position of the vehicle to search for the match in m.

    Outputs:
    s_interp:       Interpolated s position of the vehicle in m. The following holds: s_interp in range [0.0,s_tot[.
    d_displ:        Estimated displacement from the trajectory in m.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # GET RELEVANT PART OF TRAJECTORY FOR EXPECTED S -------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get s_tot into a variable
    s_tot = path_cl[-1, 0]

    if s_expected is not None:
        # get relevant part of trajectory
        if s_expected >= s_tot:
            s_expected -= s_tot

        # set values
        s_min = s_expected - s_range
        s_max = s_expected + s_range

        # check for overlapping of two laps
        if s_min < 0.0:
            s_min += s_tot

        if s_max > s_tot:
            s_max -= s_tot

        # now the following holds: s_min -> [0.0; s_tot[ s_max -> ]0.0; s_tot]

        # get indices of according points
        # - 1 to include trajectory point before s_min
        ind_start = np.searchsorted(path_cl[:, 0], s_min, side="right") - 1
        # + 1 to include trajectory point after s_max when slicing
        ind_stop = np.searchsorted(path_cl[:, 0], s_max, side="left") + 1

        # catch case of reaching into the next lap
        if ind_start < ind_stop:  # common case
            path_rel = path_cl[ind_start:ind_stop]

        else:  # overlapping case
            # temporarily add s_tot to the part in the "next lap" for convenient interpolation afterwards
            path_rel_part2 = np.copy(path_cl[:ind_stop])
            path_rel_part2[:, 0] += s_tot

            # :-1 for first part to include last/first point of closed trajectory only once
            path_rel = np.vstack((path_cl[ind_start:-1], path_rel_part2))

        # path must not be considered closed specifically as it is continuous and unclosed by construction
        consider_as_closed = False

    else:
        path_rel = path_cl[:-1]

        # path is unclosed to keep every point unique but must be considered closed to get proper matching between
        # last and first point
        consider_as_closed = True

    # ------------------------------------------------------------------------------------------------------------------
    # USE PATH MATCHING FUNCTION ON RELEVANT PART ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get s_interp and d_displ
    s_interp, d_displ = trajectory_planning_helpers.path_matching_local.\
        path_matching_local(path=path_rel,
                            ego_position=ego_position,
                            consider_as_closed=consider_as_closed,
                            s_tot=s_tot)

    # cut length if bigger than s_tot
    if s_interp >= s_tot:
        s_interp -= s_tot

    # now the following holds: s_interp -> [0.0; s_tot[

    return s_interp, d_displ


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
