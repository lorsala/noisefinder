"""
Contains functions for CPSD evaluation.
Not to be used for WOSA, which is optimized with FFT.
"""

import numpy as np

from .._dataset import DataSet
from .cpsdresults import CPSDresults
from ._ofs_L_eval import _ofs_L_eval


def CPSDevaluate(ts, freqscheme, detrend_c=False):
    """
    Evaluates the one-sided Cross Power Spectral Density (CPSD) matrix.
    The user must initialize with a frequency scheme, such as :class:`noisefinder.freqscheme_presets.lpfScheme`.
    This function evaluates the CPSD matrix at the given frequencies
    with the given stretch lengths, averaging over periodograms.

    Parameters
    -------------

    ts: list[np.ndarray]
        List of synchronous timeseries. Use list also for a single ts.
    freqscheme: noisefinder.FreqScheme
        Frequency scheme.

    Returns
    --------
    CPSDout: noisefinder.cpsd.CPSDresults
        CPSD measurement.

    """

    dataset = DataSet(ts=ts)
    datamat = dataset.datamat

    win = freqscheme.win
    fs = freqscheme.fs

    ofs_L = _ofs_L_eval(Ltot=dataset.Ltot, Ls=freqscheme.Ls, 
        olapmax=freqscheme.olapmax, optimalolap=freqscheme.optimalolap)

    CPSD = []
    navs = []
    periodograms = []
    for tmpL, tmp_dft_idx, tmp_ofs_L in zip(freqscheme.Ls, freqscheme.dft_idxs, ofs_L):

        tmpCPSD, tmpnavs, tmpperiodograms = _evalCPSD_1freq(
            datamat=datamat,
            tmpL=tmpL,
            tmp_dft_idx=tmp_dft_idx,
            fs=fs,
            win=win,
            tmp_ofs_L=tmp_ofs_L,
            detrend_c=detrend_c,
        )

        CPSD.append(tmpCPSD)
        navs.append(tmpnavs)
        periodograms.append(tmpperiodograms)

    CPSD = np.asarray(CPSD)
    navs = np.asarray(navs)

    CPSDout = CPSDresults(
        CPSD=CPSD,
        freqs=freqscheme.freqs,
        navs=navs,
        fs=freqscheme.fs,
        Ls=freqscheme.Ls,
        dft_idxs=freqscheme.dft_idxs,
        Ltot=dataset.Ltot,
        matp=dataset.matp,
        detrend_c=detrend_c,
        olapmax=freqscheme.olapmax,
        ofs_L=ofs_L,
        win=freqscheme.win,
        periodograms=periodograms,
    )

    return CPSDout


def _evalCPSD_1freq(datamat, tmpL, tmp_dft_idx, fs, win, tmp_ofs_L, detrend_c):
    """Compute the CPSD matrix at a single frequency bin.

    Splits the data into segments of length `tmpL`, applies a window,
    evaluates the DFT at the frequency corresponding to `tmp_dft_idx`,
    and averages the resulting periodograms to estimate the one-sided
    Cross Power Spectral Density (CPSD) matrix at that single
    frequency.

    Note
    ----
    `datamat` can be multidimensional (``p x N``), in which case the
    full ``p x p`` CPSD matrix is evaluated at the given frequency.

    Parameters
    ----------
    datamat : np.ndarray
        Synchronous timeseries, with shape ``(p, N)`` where `p` is
        the number of channels and `N` the number of samples.
    tmpL : int
        Segment length, in samples, used for spectral estimation.
    tmp_dft_idx : int
        DFT bin index at which to evaluate the spectral estimate.
    fs : float
        Sampling frequency, in Hz.
    win : Callable[[int], np.ndarray]
        Spectral window function; takes the segment length and
        returns the window coefficients.
    tmp_ofs_L : int
        Offset, in samples, between the start of consecutive segments
        (i.e. ``tmpL - tmp_ofs_L`` is the overlap in samples).
    detrend_c : bool
        If ``True``, subtract the mean of each segment before
        computing the DFT.

    Returns
    -------
    tmpCPSD : np.ndarray
        One-sided CPSD matrix at the given frequency bin, with shape
        ``(p, p)``. All ``nan`` if no full segment fits in `datamat`.
    tmpnavs : int
        Number of averaged segments. ``0`` if no full segment fits in
        `datamat`.
    periodograms : np.ndarray
        One-sided periodograms of each segment at the given
        frequency, with shape ``(n_segments, p)``. Empty (shape
        ``(0, p)``) if no full segment fits in `datamat`.
    """
    ndim, npoints = datamat.shape

    winpt = win(tmpL)  # spectral window
    # DFT coefficients at the requested frequency bin
    p = -2j * np.pi * np.arange(tmpL) / tmpL
    C = np.exp(tmp_dft_idx * p)

    startpoints = np.arange(0, npoints - tmpL + 1, tmp_ofs_L, dtype=int)
    tmpnavs = len(startpoints)

    if tmpnavs == 0:
        return (
            np.full((ndim, ndim), np.nan, dtype=complex),
            0,
            np.empty((0, ndim), dtype=complex),
        )

    # stack all segments at once: shape (n_segments, ndim, tmpL)
    segments = np.stack(
        [datamat[:, sp : sp + tmpL] for sp in startpoints], axis=0
    )
    if detrend_c:
        segments = segments - np.mean(segments, axis=2, keepdims=True)
    segments = segments * winpt  # apply window (broadcasts over last axis)

    periodograms = segments @ C  # (n_segments, ndim) complex periodograms

    # CPSD[i, j] = sum over segments of periodogram_i * conj(periodogram_j)
    Amat = np.einsum("ki,kj->ij", periodograms, periodograms.conj())

    wins2 = winpt @ winpt
    tmpCPSD = 2.0 * Amat / tmpnavs / fs / wins2  # one-sided CPSD matrix

    periodograms = periodograms * np.sqrt(2.0 / fs / wins2)  # one-sided periodograms

    return tmpCPSD, tmpnavs, periodograms


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
