"""
Predefined FreqScheme configurations.

This module provides factory functions that return ready-to-use
FreqScheme instances for common analysis scenarios (e.g. lpfScheme()).
"""

import numpy as np

from .freqscheme import FreqScheme
from .specwindows import BH92


def lpfScheme(Lmax, fmax, fs, olapmax=0.50, optimalolap=True):
    """
    LPF frequency scheme. See `Phys. Rev. Lett. 120, 061101 (2018)
    <https://doi.org/10.1103/PhysRevLett.120.061101>`_.
    It uses dft_idx=4 for the first bin, and 8 for the following ones.

    Parameters
    ------------
    Lmax: int
        Number of datapoints in the longest segment.
    fmax : float, None
        Maximum frequency. If None, it defaults to Nyquist.
    fs: float
        Sampling frequency [Hz]
    optimalolap: bool
        If True, adjust segment overlap to maximize data coverage.
    """

    if not isinstance(Lmax, (int, np.integer)):
        raise TypeError(f"'Lmax' must be an integer, got {type(Lmax).__name__}")
    if Lmax <= 1:
        raise ValueError(f"'Lmax' must be > 1, got {Lmax}")  

    fmax = fs / 2 if fmax is None else fmax  # set fmax

    if fmax > fs / 2:
        raise ValueError(
            f"'fmax' ({fmax}) must be lower than the Nyquist frequency fs/2 ({fs / 2})"
        )

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
        olapmax=olapmax,
        dft_idxs=np.asarray(dft_idxs, dtype=int),
        Ls=np.asarray(Ls, dtype=int),
        win=BH92,
        optimalolap=optimalolap,
        name='lpf'
    )

def wosaScheme(nperseg, fs, win, olapmax=0.50, optimalolap=True):
    """
    WOSA (Welch overlapping segment average) frequency scheme.

    Parameters
    -------------

    nperseg: int
        Number of datapoints per DFT segment.
    win: Callable
        Spectral window function.
    fs: float
        Sampling frequency.
    olapmax: float
        Maximum overlap.
    optimalolap: bool
        If True, reduces overlap to maximixe data coverage.
    """

    if not isinstance(nperseg, (int, np.integer)):
        raise TypeError(f"'nperseg' must be an integer, got {type(nperseg).__name__}")
    if nperseg <= 1:
        raise ValueError(f"'nperseg' must be > 1, got {nperseg}")

    if not callable(win):
        raise TypeError(f"'win' must be callable, got {type(win).__name__}")

    if not isinstance(olapmax, float):
        raise TypeError(f"'olapmax' must be a float, got {type(olapmax).__name__}")
    if not (0.0 < olapmax < 1.0):
        raise ValueError(f"'olapmax' must be between 0 and 1, got {olapmax}")

    freqs = np.fft.rfftfreq(n=nperseg, d=1 / fs)
    Ls = np.full(len(freqs), nperseg, dtype=int)
    dft_idxs = np.arange(0, len(freqs), 1)

    return FreqScheme(
        fs=fs,
        olapmax=olapmax,
        dft_idxs=dft_idxs,
        Ls=Ls,
        win=win,
        optimalolap=optimalolap,
        name="wosa"
    )