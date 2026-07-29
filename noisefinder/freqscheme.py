from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True, kw_only=True)
class FreqScheme:
    """
    Collect parameters related to the frequency scheme, with no PSD calculation.
    """

    fs: float
    """Sampling frequency [Hz]."""

    dft_idxs: np.ndarray
    """Array of DFT indices."""

    Ls: np.ndarray
    """Array of segment lengths."""

    olapmax: float
    """Maximum overlap, will be adjusted to maximize coverage."""

    optimalolap: bool
    """If True, adjust segment overlap to maximize data coverage."""

    win: Callable[[int], list[float]]
    """Spectral window."""


    def __post_init__(self) -> None:
        """Ensure freqscheme is valid"""
        if not np.all(self.Ls > 0):
            msg = "All Ls must be greater than 0."
            raise ValueError(msg)
        if not np.all(self.dft_idxs > 0):
            msg = "All dft_idxs must be greater than 0."
            raise ValueError(msg)
        if self.olapmax < 0 or self.olapmax > 1:
            msg = "Overlap fraction must be between 0 and 1."
            raise ValueError(msg)

    @property
    def freqs(self):
        """
        Compute frequencies based on DFT indexes and stretch lengths.
        """
        freqs = self.dft_idxs / (self.Ls / self.fs)
        return freqs
