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
    datamat: np.ndarray = field(init=False)
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
        if datamat.ndim == 1:
            datamat = np.reshape(datamat, (1, datamat.size))
        object.__setattr__(self, "datamat", datamat)
        object.__setattr__(self, "matp", datamat.shape[0])
        object.__setattr__(self, "Ltot", datamat.shape[1])

    def _sanitycheck(self):
        if not isinstance(self.ts, list):
            msg = "input timeseries must be given as a list"
            raise ValueError(msg)
        if not np.all([ts_tmp.shape == self.ts[0].shape for ts_tmp in self.ts]):
            msg = "lists have different amount of data"
            raise ValueError(msg)
