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

    ofs_L = _ofs_L_eval(Ltot=dataset.Ltot, Ls=freqscheme.Ls, 
        olapmax=freqscheme.olapmax, optimalolap=freqscheme.optimalolap)

    if freqscheme.name == "wosa": 
        # WOSA is much more efficient with FFT, so we have a separate function
        if not np.all(freqscheme.Ls==freqscheme.Ls[0]):
            raise ValueError("In WOSA scheme, all nperseg must be equal.")
        if not np.all(ofs_L==ofs_L[0]):
            raise ValueError("In WOSA scheme, all ofs_L must be equal.")
        if not dataset.Ltot > freqscheme.Ls[0]:
            raise ValueError("In WOSA scheme, nperseg must be greater than total length.")
        CPSD, navs, periodograms = _evalWOSACPSD(
            datamat=datamat,
            tmpL=freqscheme.Ls[0],
            tmp_ofs_L=ofs_L[0],
            fs=freqscheme.fs,
            win=freqscheme.win,
            detrend_c=detrend_c,
        )

    else:
        results = [
            _evalCPSD_1freq(
                datamat=datamat,
                tmpL=tmpL,
                tmp_dft_idx=tmp_dft_idx,
                fs=freqscheme.fs,
                win=freqscheme.win,
                tmp_ofs_L=tmp_ofs_L,
                detrend_c=detrend_c,
            )
            for tmpL, tmp_dft_idx, tmp_ofs_L
            in zip(freqscheme.Ls, freqscheme.dft_idxs, ofs_L)
        ]

        CPSD, navs, periodograms = map(list, zip(*results))
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

    if (tmp_dft_idx==0 or (tmp_dft_idx == tmpL // 2 and tmpL % 2 == 0)): #DC or Nyquist
        tmpCPSD = 1.0 * Amat / tmpnavs / fs / wins2
        periodograms = periodograms * np.sqrt(1.0 / fs / wins2)
    else: # non-DC and non-Nyquist: one-sided
        tmpCPSD = 2.0 * Amat / tmpnavs / fs / wins2  # one-sided CPSD matrix
        periodograms = periodograms * np.sqrt(2.0 / fs / wins2)  # one-sided periodograms

    return tmpCPSD, tmpnavs, periodograms


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
    tmpCPSD = Amat / tmpnavs / fs / wins2  # two-sided CPSD matrix
    periodograms = periodograms * np.sqrt(1.0 / fs / wins2)  # two-sided periodograms

    tmpCPSD[1:-1] *= 2.0 # one-sided CPSD matrix
    periodograms[:, :, 1:-1] *= np.sqrt(2.0) #one-sided periodograms
    if tmpL % 2 != 0: # double last bin only for odd tmpL
        tmpCPSD[-1] *= 2.0 
        periodograms[:, :, -1] *= np.sqrt(2.0)
    

    tmpnavs_arr = np.full(nfreqs, tmpnavs)

    return tmpCPSD, tmpnavs_arr, periodograms