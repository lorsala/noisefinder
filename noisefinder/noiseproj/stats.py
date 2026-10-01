"""
Multi-frequency noise projection.
"""

import numpy as np

from . import stats_onebin


def PSDresidual_qnt(mfnp, q):
    """Compute confidence intervals for the residual noise PSD.

    For each frequency in `mfnp`, computes the quantiles of the
    residual Power Spectral Density remaining after noise projection.

    Parameters
    ----------
    mfnp : noisefinder.noiseproj
        Multi-frequency noise projection results, as returned by
        :func:`NoiseProj`.
    q : float
        Lower-tail probability.

    Returns
    -------
    np.ndarray
        Residual noise PSD confidence intervals, with shape
        ``(3, n_freqs)``: lower bound, median, and upper bound for
        each frequency.
    """

    q = np.atleast_1d(q)
    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    PSDres_qnt = [stats_onebin.PSDresidual_onebin_qnt(sfnp, q) for sfnp in mfnp.sfnp_arr]
    return np.asarray(PSDres_qnt).T


def ASDresidual_qnt(mfnp, q):
    """Compute confidence intervals for the residual noise ASD.

    Derives the Amplitude Spectral Density confidence intervals as the
    square root of the residual PSD confidence intervals (see
    :func:`PSDresidual_qnt`).

    Parameters
    ----------
    mfnp : noiseproj
        Multi-frequency noise projection results, as returned by
        :func:`NoiseProj`.
    q : float
        Lower-tail probability.

    Returns
    -------
    np.ndarray
        Residual noise ASD confidence intervals, with shape
        ``(3, n_freqs)``: lower bound, median, and upper bound for
        each frequency.
    """
    PSDres_qnt = PSDresidual_qnt(mfnp, q)
    ASDres_qnt = np.sqrt(PSDres_qnt)
    return ASDres_qnt


def alpha_qnt(mfnp, q):
    """Compute confidence intervals of the projected susceptibilities
    for every frequency bin.

    Bins where decorrelation was not possible (``mfnp.sfnp_arr[i] is None``)
    are filled with ``nan``.

    Parameters
    ----------
    mfnp : noiseproj
        Multi-frequency noise projection results. ``mfnp.sfnp_arr`` holds
        one ``noiseproj_onebin`` per bin, or ``None`` where ``navs == 0``.
    q : float or array_like of float
        Lower-tail probabilities, each in [0, 1].

    Returns
    -------
    alpre_qnt : np.ndarray, shape (nf, r, nq)
        Quantiles of the real part, ``nan`` on invalid bins.
    alpim_qnt : np.ndarray, shape (nf, r, nq), or None
        Quantiles of the imaginary part, ``nan`` on invalid bins.
        ``None`` if the projection is real-valued.

    Raises
    ------
    ValueError
        If no bin could be decorrelated.
    """
    q = np.atleast_1d(q)
    nq = q.size
    nf = len(mfnp.sfnp_arr)

    valid = next((s for s in mfnp.sfnp_arr if s is not None), None)
    if valid is None:
        msg = "No frequency bin could be decorrelated."
        raise ValueError(msg)

    r = valid.r
    is_complex = valid.case == "complex"

    alpre_qnt = np.full((nf, r, nq), np.nan)
    alpim_qnt = np.full((nf, r, nq), np.nan) if is_complex else None

    for ffi, sfnp in enumerate(mfnp.sfnp_arr):
        if sfnp is None:
            continue
        re_i, im_i = stats_onebin.alpha_onebin_qnt(sfnp, q)
        alpre_qnt[ffi] = re_i
        if is_complex:
            alpim_qnt[ffi] = im_i

    return alpre_qnt, alpim_qnt


def alpha_RVS(mfnp, size):
    """Draw random samples of the susceptibilities for every frequency bin.

    For each bin, samples from the joint posterior of the
    susceptibilities (see :func:`stats_onebin.alpha_onebin_RVS`).
    Samples of different bins are independent. Bins where
    decorrelation was not possible (``mfnp.sfnp_arr[i] is None``) are
    filled with ``nan``.

    Parameters
    ----------
    mfnp : noiseproj.NoiseProjResults
        Multi-frequency noise projection results, as returned by
        :func:`run_noiseproj`.
    size : int
        Number of samples to draw per frequency bin. Must be a positive
        integer.

    Returns
    -------
    np.ndarray
        Samples of the susceptibilities, with shape ``(nf, size, r)``.
        Complex-valued if the projection is complex, real-valued
        otherwise. ``nan`` on invalid bins.

    Raises
    ------
    ValueError
        If no bin could be decorrelated, or if `size` is not positive.
    TypeError
        If `size` is not an integer.
    """
    stats_onebin._check_size(size)

    nf = len(mfnp.sfnp_arr)

    valid = next((s for s in mfnp.sfnp_arr if s is not None), None)
    if valid is None:
        msg = "No frequency bin could be decorrelated."
        raise ValueError(msg)

    r = valid.r
    if valid.case == "complex":
        dtype, fill = complex, complex(np.nan, np.nan)
    else:
        dtype, fill = float, np.nan

    alphas_rvs = np.full((nf, size, r), fill, dtype=dtype)
    for ffi, sfnp in enumerate(mfnp.sfnp_arr):
        if sfnp is None:
            continue
        alphas_rvs[ffi] = stats_onebin.alpha_onebin_RVS(sfnp, size)

    return alphas_rvs


def R2contrib_qnt(mfnp, q):
    """Compute confidence intervals for the R2 contribution.

    For each frequency in `mfnp`, computes the quantiles of the
    contribution to the multiple coherence R2 attributable to noise
    projection.

    Parameters
    ----------
    mfnp : noiseproj
        Multi-frequency noise projection results, as returned by
        :func:`NoiseProj`.
    q : float
        Lower-tail probability.

    Returns
    -------
    np.ndarray
        R2 contribution confidence intervals, with shape
        ``(3, n_freqs)``: lower bound, median, and upper bound for
        each frequency.
    """

    R2contrib_qnt = [stats_onebin.R2contrib_onebin_qnt(sfnp, q) for sfnp in mfnp.sfnp_arr]
    return np.asarray(R2contrib_qnt).T
