"""
Defines DataSet class, containing data.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, kw_only=True)
class DataSet:
    """
    Contains synchronously sampled timeseries data.
    Stores a set of time series as a list of arrays and derives a
    combined data matrix along with related dimensional metadata.
    """

    ts: list[np.ndarray]
    """List of synchronously sampled time series, one array per
        channel/signal."""
    datamat: np.ndarray = field(init=False, repr=False)
    """Combined data matrix built from `ts`. Set automatically
        (not passed at initialization)."""
    matp: int = field(init=False)
    """Number of channels/signals in the dataset (i.e. number of
        elements in `ts`, or number of columns of `datamat`). Set
        automatically (not passed at initialization)."""
    Ltot: int = field(init=False)
    """Total number of samples in each time series. Set
        automatically (not passed at initialization)."""

    def __post_init__(self) -> None:
        self._sanitycheck()

        datamat = np.asarray(self.ts)
        object.__setattr__(self, "datamat", datamat)
        object.__setattr__(self, "matp", datamat.shape[0])
        object.__setattr__(self, "Ltot", datamat.shape[1])

    def _sanitycheck(self):
        if not isinstance(self.ts, list):
            raise TypeError(f"'ts' must be a list, got {type(self.ts).__name__}")

        if len(self.ts) == 0:
            raise ValueError("'ts' must contain at least one time series")

        for i, arr in enumerate(self.ts):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"ts[{i}] must be a numpy.ndarray, got {type(arr).__name__}"
                )
            if arr.ndim != 1:
                raise ValueError(
                    f"ts[{i}] must be 1-D, got shape {arr.shape}"
                )

        # length check
        lengths = [arr.size for arr in self.ts]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All time series in 'ts' must have equal length, got lengths {lengths}"
            )