# Description
This repository provides some helper functions we frequently use in our trajectory planning software stack at FTM/TUM.
Many of the functions are based on third order splines because we use them as a basis for our path planning.
Please keep in mind that some of the functions are designed to work on a closed (race-) track and might therefore
not work properly on a common street network.

# List of components
* `angle3pt`: Calculates angle by turning from a to c around b.
* `calc_ax_profile`: Calculate the longitudinal acceleration profile for a given velocity profile.
* `calc_head_curv_an`: Analytical curvature calculation on the basis of third order splines.
* `calc_head_curv_num`: Numerical curvature calculation.
* `calc_normal_vectors`: Calculate normalized normal vectors on the basis of headings psi.
* `calc_spline_lengths`: Calculate spline lengths.
* `calc_splines`: Calculate splines for a (closable) path.
* `calc_t_profile`: Calculate the temporal duration profile for a given velocity profile.
* `calc_vel_profile`: Calculate velocity profile on the basis of a forward/backward solver.
* `calc_vel_profile_brake`: Calculate velocity profile on the basis of a pure forward solver.
* `conv_filt`: Filter a given signal using a 1D convolution (moving average) filter.
* `import_ggv`: Import the ggv diagram containing vehicle handling limits from a text file.
* `interp_splines`: Interpolate splines to get points with a desired stepsize.
* `normalize_psi`: Normalize heading psi such that the interval [-pi, pi[ holds.
* `path_matching_global`: Match own vehicle position to a global (i.e. closed) path.
* `path_matching_local`: Match own vehicle position to a local (i.e. unclosed) path.
* `progressbar`: Commandline progressbar (to be called in a for loop).
* `side_of_line`: Function determines if a point is on the left or right side of a line.
* `spline_approximation`: Function used to obtain a smoothed track on the basis of a spline approximation.

Contact persons: [Alexander Heilmeier](mailto:alexander.heilmeier@tum.de), [Tim Stahl](mailto:stahl@ftm.mw.tum.de).
