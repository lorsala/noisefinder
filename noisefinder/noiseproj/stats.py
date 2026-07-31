"""
Multi-frequency noise projection.
"""

from dataclasses import dataclass
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
    """Compute confidence intervals for the projected susceptibilities.

    For each frequency in `mfnp`, computes the quantiles of the real
    and imaginary parts of the susceptibilities estimated by noise
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
    alpre_qnt : np.ndarray
        Confidence intervals for the real part of the susceptibilities,
        with shape ``(3, n_freqs)``.
    alpim_qnt : np.ndarray
        Confidence intervals for the imaginary part of the
        susceptibilities, with shape ``(3, n_freqs)``.
    """

    alp_qnt = [stats_onebin.alpha_onebin_qnt(sfnp, q) for sfnp in mfnp.sfnp_arr]
    alpre_qnt, alpim_qnt = zip(*alp_qnt) if alp_qnt else ((), ())

    return np.asarray(alpre_qnt), np.asarray(alpim_qnt)


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
