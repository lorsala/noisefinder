"""
Contains functions for CPSD evaluation with WOSA scheme.
"""

import numpy as np
from .._dataset import DataSet
from .cpsdresults import CPSDresults
from ._ofs_L_eval import _ofs_L_eval


def CPSDevaluateWOSA(ts, nperseg, win, fs, olapmax, detrend_c=False, optimalolap=True):
    """
    Evaluates the one-sided Cross Power Spectral Density (CPSD) matrix,
    with WOSA frequency scheme.

    Parameters
    -------------

    ts: list[np.ndarray]
        List of synchronous timeseries. Use list also for a single ts.
    nperseg: int
        Number of datapoints per DFT segment.
    win: Callable
        Spectral window function.
    fs: float
        Sampling frequency.
    olapmax:
        Maximum overlap.
    detrend_c: bool 
        If true, subtract mean before calculating DFT.
        Defaults to False.
    optimalolap: bool
        If True, reduces overlap to maximixe data coverage.
    """

    dataset = DataSet(ts=ts)
    datamat = dataset.datamat

    ofs_L = _ofs_L_eval(Ltot=dataset.Ltot, Ls=nperseg, olapmax=olapmax, optimalolap=optimalolap)

    CPSD = []
    navs = []
    periodograms = []
    CPSD, navs, freqs, periodograms = _evalWOSACPSD(
        datamat=datamat,
        tmpL=nperseg,
        tmp_ofs_L=ofs_L,
        fs=fs,
        win=win,
        detrend_c=detrend_c,
    )
    Ls = freqs**0 * nperseg
    dft_idxs = np.arange(0, len(freqs), 1)


    CPSDout = CPSDresults(
        CPSD=CPSD,
        freqs=freqs,
        navs=navs,
        fs=fs,
        Ls=Ls,
        dft_idxs=dft_idxs,
        Ltot=dataset.Ltot,
        matp=dataset.matp,
        detrend_c=detrend_c,
        olapmax=olapmax,
        ofs_L=ofs_L,
        win=win,
        periodograms=periodograms,
    )

    return CPSDout


def _evalWOSACPSD(datamat, tmpL, fs, win, tmp_ofs_L, detrend_c):
    """Compute the CPSD matrix using the Welch/WOSA averaging scheme.

    Splits the data into overlapping or non-overlapping segments of
    length `tmpL`, applies a window, computes the DFT of each segment,
    and averages the resulting periodograms to estimate the one-sided
    Cross Power Spectral Density (CPSD) matrix.

    Note
    ----
    `datamat` can be multidimensional (``p x N``), in which case the
    full ``p x p`` CPSD matrix is evaluated at each frequency.

    Parameters
    ----------
    datamat : np.ndarray
        Synchronous timeseries, with shape ``(p, N)`` where `p` is
        the number of channels and `N` the number of samples.
    tmpL : int
        Segment length, in samples, used for spectral estimation.
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
        One-sided CPSD matrix, with shape ``(nfreqs, p, p)``. All
        zeros if no full segment fits in `datamat`.
    tmpnavs : np.ndarray
        Number of averaged segments, broadcast to shape ``(nfreqs,)``
        (constant across frequency for this WOSA scheme). All zeros
        if no full segment fits in `datamat`.
    freqs : np.ndarray
        Frequency vector, in Hz, with shape ``(nfreqs,)``.
    periodograms : np.ndarray
        One-sided periodograms of each segment, with shape
        ``(n_segments, p, nfreqs)``. Empty (shape ``(0, p, nfreqs)``)
        if no full segment fits in `datamat`.
    """
    ndim, npoints = datamat.shape
    freqs = np.fft.rfftfreq(n=tmpL, d=1 / fs)
    nfreqs = len(freqs)
    winpt = win(tmpL)  # spectral window

    startpoints = np.arange(0, npoints - tmpL + 1, tmp_ofs_L, dtype=int)
    tmpnavs = len(startpoints)

    if tmpnavs == 0:
        return (
            np.zeros((nfreqs, ndim, ndim), dtype=complex),
            np.zeros(nfreqs),
            freqs,
            np.empty((0, ndim, nfreqs), dtype=complex),
        )

    # stack all segments at once: shape (n_segments, ndim, tmpL)
    segments = np.stack(
        [datamat[:, sp : sp + tmpL] for sp in startpoints], axis=0
    )
    if detrend_c:
        segments = segments - np.mean(segments, axis=2, keepdims=True)
    segments = segments * winpt  # apply window (broadcasts over last axis)

    # one FFT call for all segments/channels at once
    periodograms = np.fft.rfft(segments, axis=2)  # (n_segments, ndim, nfreqs)

    # CPSD[f, i, j] = sum over segments of periodogram_i * conj(periodogram_j)
    Amat = np.einsum("kif,kjf->fij", periodograms, periodograms.conj())

    wins2 = winpt @ winpt
    tmpCPSD = 2.0 * Amat / tmpnavs / fs / wins2  # one-sided CPSD matrix

    periodograms = periodograms * np.sqrt(2.0 / fs / wins2)  # one-sided periodograms

    tmpnavs_arr = np.full(nfreqs, tmpnavs)

    return tmpCPSD, tmpnavs_arr, freqs, periodograms