"""
Spectral windows.
"""

import numpy as np


def BH92(M: int):
    """
    Blackman-Harris -92dB window with M samples.
    """
    z = np.arange(0, M) * 2 * np.pi / M
    return (
        0.35875
        - 0.48829 * np.cos(z)
        + 0.14128 * np.cos(2 * z)
        - 0.01168 * np.cos(3 * z)
    )


def Nuttall4(M: int):
    """
    Nuttall4 window with M samples, as defined in [Nuttall1981,Heinzel2002]
    """
    z = np.arange(0, M) * 2 * np.pi / M
    return (
        0.3125 - 0.46875 * np.cos(z) + 0.1875 * np.cos(2 * z) - 0.03125 * np.cos(3 * z)
    )
