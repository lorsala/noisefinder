"""
Contains definition of class CPSDresults
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True, kw_only=True)
class CPSDresults:
    """Contains the one-sided Cross Power Spectral Density (CPSD) matrix.
    Evaluates cross-coherences and multiple coherence R2 from the CPSD
    matrix, computed automatically at initialization via `__post_init__`.
    """

    CPSD: np.ndarray
    """One-sided Cross Power Spectral Density matrix, with shape
        ``(n_freqs, n_ts, n_ts)``."""
    freqs: np.ndarray
    """Frequency vector, in Hz, corresponding to the CPSD estimates."""
    navs: np.ndarray
    """Number of averages (segments) used to estimate the CPSD at
        each frequency."""
    fs: float
    """Sampling frequency, in Hz."""
    Ls: list[int]
    """Segment lengths (number of samples) used for each block."""
    dft_idxs: list[int]
    """DFT bin indices corresponding to the estimated frequencies."""
    Ltot: int
    """Total number of samples in the analyzed signal."""
    matp: int
    """Number of timeseries."""
    detrend_c: bool
    """If true, subtracts mean before DFT estimation."""
    olapmax: float
    """Maximum segment overlap fraction used in the estimation."""
    ofs_L: list[int]
    """Offset(s), in samples, for each segment."""
    win: Callable[[int], list[float]]
    """Spectral window function."""
    periodograms: list[np.ndarray]
    """Periodograms computed for each segment, used as the basis for
        the CPSD estimate."""
    PSD: np.ndarray = field(init=False)
    """Power Spectral Density, computed as the real part of the
        diagonal of `CPSD`. Set automatically in `__post_init__`, not
        passed at initialization."""
    cohere: np.ndarray = field(init=False)
    """Cross-coherence matrix. Set automatically in `__post_init__`,
        not passed at initialization."""
    MSC: np.ndarray = field(init=False)
    """Magnitude-Squared Coherence, computed as ``abs(cohere) ** 2``.
        Set automatically in `__post_init__`, not passed at
        initialization."""
    R2: np.ndarray = field(init=False)
    """Multiple coherence (squared), computed from `CPSD` and `navs`.
        Set automatically in `__post_init__`, not passed at
        initialization."""

    def __post_init__(self) -> None:
        """Compute derived quantities (cohere, MSC, R2, PSD) from CPSD.
        """
        cohere_tmp = self._evalcohere()
        object.__setattr__(self, "cohere", cohere_tmp)
        MSC_tmp = np.abs(self.cohere) ** 2
        object.__setattr__(self, "MSC", MSC_tmp)
        R2_tmp = self._evalR2()
        object.__setattr__(self, "R2", R2_tmp)
        PSD_tmp = np.real(np.diagonal(self.CPSD, axis1=1, axis2=2)).T
        object.__setattr__(self, "PSD", PSD_tmp)

    def _evalcohere(self):
        """Evaluate the cross-coherence matrix from the CPSD.

        Whitens each cross-spectral matrix `Sij` by its diagonal to
        obtain the normalized coherence.

        Returns
        -------
        np.ndarray
            Cross-coherence matrix, with the same shape as `CPSD`.
        """
        cohere = [
            np.diag(np.diag(Sij) ** (-1 / 2)) @ Sij @ np.diag(np.diag(Sij) ** (-1 / 2))
            for Sij in self.CPSD
        ]
        return np.asarray(cohere)

    def _evalR2(self):
        """Evaluate the multiple coherence R2 for each frequency.

        For each frequency, computes R2 from the CPSD matrix `Sij`
        provided there are enough averages (`nn >= Sij.shape[0]`);
        otherwise returns NaN for that frequency.

        Returns
        -------
        np.ndarray
            Multiple coherence R2 values, one per frequency.
        """
        R2 = []
        for Sij, nn in zip(self.CPSD, self.navs):
            if nn < Sij.shape[0]:
                R2.append(np.nan)
            else:
                R2.append(np.real(1 - 1 / (np.linalg.inv(Sij)[0, 0] * Sij[0, 0])))
        return np.asarray(R2)
