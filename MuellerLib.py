import numpy as np

def MuellerRetarder(delta, theta):
    """
    Returns the Mueller matrix of a retarder with delta as the phase difference and theta as the orientation angle.
    Both in terms of degrees !!!!
    """
    delta_rad = np.deg2rad(delta)
    theta_rad = np.deg2rad(theta)

    c_delta = np.cos(delta_rad)
    s_delta = np.sin(delta_rad)
    c2_theta = np.cos(2 * theta_rad)
    s2_theta = np.sin(2 * theta_rad)

    M = np.array([
        [1, 0, 0, 0],
        [0, c2_theta**2 + s2_theta**2 * c_delta, c2_theta * s2_theta * (1 - c_delta), s2_theta * s_delta],
        [0, c2_theta * s2_theta * (1 - c_delta), s2_theta**2 + c2_theta**2 * c_delta, -c2_theta * s_delta],
        [0, -s2_theta * s_delta, c2_theta * s_delta, c_delta]
    ])
    return M

def MuellerLinearPolarizer(theta):
    """
    Returns the Mueller matrix of a linear polarizer with orientation angle theta.
    In terms of degrees !!!!
    """
    theta_rad = np.deg2rad(theta)
    c2_theta = np.cos(2 * theta_rad)
    s2_theta = np.sin(2 * theta_rad)

    M = 0.5 * np.array([
        [1, c2_theta, s2_theta, 0],
        [c2_theta, c2_theta**2, c2_theta * s2_theta, 0],
        [s2_theta, c2_theta * s2_theta, s2_theta**2, 0],
        [0, 0, 0, 0]
    ])
    return M