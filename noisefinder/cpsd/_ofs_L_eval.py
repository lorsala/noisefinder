import numpy as np


def _ofs_L_eval(Ltot, Ls, olapmax, optimalolap=True):
    """
    Return array of number of non-overlapping points,
    at multiple frequencies.

    Parameters
    -------------

    Ltot: float
        Total number of datapoints.
    Ls: np.ndarray
        Frequency-dependent segment length for PSD estimation.
    olapmax: float
        Maximum segment overlap.
    optimalolap: bool
        If True, adjust segment overlap to maximize data coverage.
    """

    if olapmax<0 or olapmax>1:
        msg = "Overlap must be between 0 and 1."
        raise ValueError(msg)

    if not optimalolap:
        nonolapL = np.ceil(Ls * (1 - olapmax))
    else:
        tmpM = np.floor(Ltot / (Ls * (1 - olapmax)) - olapmax / (1 - olapmax))
        with np.errstate(divide="ignore", invalid="ignore"):
            nonolapL = np.where(tmpM <= 1, Ltot, np.floor((Ltot - Ls) / (tmpM - 1)))
    return nonolapL
