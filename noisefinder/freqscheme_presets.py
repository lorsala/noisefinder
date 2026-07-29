"""
Predefined FreqScheme configurations.

This module provides factory functions that return ready-to-use
FreqScheme instances for common analysis scenarios (e.g. lpfScheme()).
"""

import numpy as np

from .freqscheme import FreqScheme
from .specwindows import BH92


def lpfScheme(Lmax, fmax, fs, optimalolap=True):
    """
    LPF frequency scheme. See `Phys. Rev. Lett. 120, 061101 (2018)
    <https://doi.org/10.1103/PhysRevLett.120.061101>`_.
    It uses dft_idx=4 for the first bin, and 8 for the following ones.

    Parameters
    ------------
    Lmax: int
        Number of datapoints in the longest segment.
    fmax : float
        Maximum frequency. If None, it defaults to Nyquist.
    fs: float
        Sampling frequency [Hz]
    optimalolap: bool
        If True, adjust segment overlap to maximize data coverage.
    """
    fmax = fs / 2 if fmax is None else fmax  # set fmax
    fmax = fs / 2 if fmax > fs / 2 else fmax  # set fmax, max Nyquist
    dft_idx_0 = 4
    dft_idx_i = 8
    r = 3 / 5
    Ls = [Lmax]
    dft_idxs = [dft_idx_0]
    aa = 2
    while True:
        tmpL = np.floor(r ** (aa - 2) * Lmax)
        tmpf = dft_idx_i / (tmpL / fs)
        if tmpf > fmax:
            break
        Ls.append(tmpL)
        dft_idxs.append(dft_idx_i)
        aa += 1

    return FreqScheme(
        fs=fs,
        olapmax=0.50,
        dft_idxs=np.asarray(dft_idxs, dtype=int),
        Ls=np.asarray(Ls, dtype=int),
        win=BH92,
        optimalolap=optimalolap
    )
