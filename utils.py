import numpy as np


def eulers_2_rot_matrix(x):
    """
    Transforms a set of Euler angles into a rotation matrix.

    Args:
        x (numpy.ndarray): Input vector of Euler angles [gamma_x, beta_y, alpha_z]
                          which are ZYX Eulers angles in radians.

    Returns:
        numpy.ndarray: The rotation matrix.
    """
    gamma_x = x[0]
    beta_y = x[1]
    alpha_z = x[2]

    return rotz(alpha_z) @ roty(beta_y) @ rotx(gamma_x)


def rotx(t):
    """
    Rotation about X axis.

    Args:
        t (float): Angle in radians.

    Returns:
        numpy.ndarray: The rotation matrix about the X axis.
    """
    ct = np.cos(t)
    st = np.sin(t)

    return np.array([[1, 0, 0],
                     [0, ct, -st],
                     [0, st, ct]])


def roty(t):
    """
    Rotation about Y axis.

    Args:
        t (float): Angle in radians.

    Returns:
    numpy.ndarray: The rotation matrix about the Y axis.
    """
    ct = np.cos(t)
    st = np.sin(t)

    return np.array([[ct, 0, st],
                     [0, 1, 0],
                     [-st, 0, ct]])


def rotz(t):
    """
    Rotation about Z axis.

    Args:
        t (float): Angle in radians.

    Returns:
    numpy.ndarray: The rotation matrix about the Z axis.
    """
    ct = np.cos(t)
    st = np.sin(t)

    return np.array([[ct, -st, 0],
                     [st, ct, 0],
                     [0, 0, 1]])