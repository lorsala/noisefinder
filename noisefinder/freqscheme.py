from dataclasses import dataclass
from typing import Callable
from numpy.typing import ArrayLike

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
    name: str = "noname"
    """Scheme name. Optional but highly suggested."""

    def __post_init__(self):
        # fs
        if not isinstance(self.fs, (int, float)):
            raise TypeError(f"'fs' must be a number, got {type(self.fs).__name__}")
        if self.fs <= 0:
            raise ValueError(f"'fs' must be positive, got {self.fs}")

        # dft_idxs / Ls: shared checks
        for name in ("dft_idxs", "Ls"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"'{name}' must be a numpy.ndarray, got {type(arr).__name__}")
            if not np.issubdtype(arr.dtype, np.integer):
                raise TypeError(f"'{name}' must have an integer dtype, got {arr.dtype}")
            if arr.size == 0:
                raise ValueError(f"'{name}' must not be empty")
        if np.any(self.Ls < 1):
            raise ValueError(f"'{name}' must contain only values >= 1, got min {self.Ls.min()}")
        if np.any(self.dft_idxs < 0):
            raise ValueError(f"'{name}' must contain only values >= 0, got min {self.dft_idxs.min()}")
        if len(self.dft_idxs) != len(self.Ls):
            raise ValueError("'dft_idxs' and 'Ls' must have the same length")

        # olapmax
        if not isinstance(self.olapmax, (float)):
            raise TypeError(f"'olapmax' must be a float, got {type(self.olapmax).__name__}")
        if not (0.0 < self.olapmax < 1.0):
            raise ValueError(f"'olapmax' must be between 0 and 1, got {self.olapmax}")

        # optimalolap
        if not isinstance(self.optimalolap, bool):
            raise TypeError(
                f"'optimalolap' must be a bool, got {type(self.optimalolap).__name__}"
            )

        # win
        if not callable(self.win):
            raise TypeError(f"'win' must be callable, got {type(self.win).__name__}")

    @property
    def freqs(self):
        """
        Compute frequencies based on DFT indexes and stretch lengths.
        """
        freqs = self.dft_idxs / (self.Ls / self.fs)
        return freqs
