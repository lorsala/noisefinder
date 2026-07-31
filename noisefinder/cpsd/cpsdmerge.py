import numpy as np

from .._dataset import DataSet
from .cpsdresults import CPSDresults

def CPSDmerge(CPSD1, CPSD2):
    """
    Merges two CPSDresults classes, with a weighted average based on 
    the number of periodograms available. 
    Obviously, the CPSDs must be evaluated at the same frequencies.

    Parameters
    -------------

    CPSD1: noisefinder.CPSDresults

    CPSD2: noisefinder.CPSDresults
    """

    # a few checks
    if not np.all(CPSD1.fs == CPSD2.fs):
        msg = "Sampling frequency fs must be the same."
        raise ValueError(msg)
    if not np.all(CPSD1.Ls == CPSD2.Ls):
        msg = "Segment lengths Ls must be the same."
        raise ValueError(msg)
    if not np.all(CPSD1.matp == CPSD2.matp):
        msg = "Number of synchronous timeseries matp must be the same."
        raise ValueError(msg)
    if not np.all(CPSD1.dft_idxs == CPSD2.dft_idxs):
        msg = "DFT indexes dft_idxs must be the same."
        raise ValueError(msg)
    if not np.all(CPSD1.detrend_c == CPSD2.detrend_c):
        msg = "Parameter detrend_c must be the same."
        raise ValueError(msg)

    # set CPSD. note that nan_to_num sets nans to zero so that we can sum even if PSD is nondefined (e.g. navs=0)
    CPSD_new = (
        np.nan_to_num(CPSD1.CPSD) * CPSD1.navs[:, None, None]
        + np.nan_to_num(CPSD2.CPSD) * CPSD2.navs[:, None, None]
    ) / (CPSD1.navs[:, None, None] + CPSD2.navs[:, None, None])
    navs_new = CPSD1.navs + CPSD2.navs
    periodograms_new = [
        (
            per1
            if len(per2) == 0
            else per2 if len(per1) == 0 else np.concatenate((per1, per2), axis=0)
        )
        for per1, per2 in zip(CPSD1.periodograms, CPSD2.periodograms)
    ]
    Ltotnew = CPSD1.Ltot + CPSD2.Ltot

    CPSDout = CPSDresults(
        CPSD=CPSD_new,
        freqs=CPSD1.freqs,
        navs=navs_new,
        periodograms=periodograms_new,
        fs=CPSD1.fs,
        Ls=CPSD1.Ls,
        dft_idxs=CPSD1.dft_idxs,
        Ltot=Ltotnew,
        matp=CPSD1.matp,
        detrend_c=CPSD1.detrend_c,
        olapmax=[CPSD1.olapmax, CPSD2.olapmax],
        ofs_L=None,
        win=CPSD1.win,
    )

    return CPSDout
