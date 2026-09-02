"""
Single-frequency noise projection.
"""

from typing import Any
import warnings
from dataclasses import dataclass
import numbers


import numpy as np
import scipy.stats as st

from ..cpsd import stats_onebin as cpsd_stats_onebin



def PSDresidual_onebin_RVS(sfnp, size):
    """Draw random samples from the residual PSD distribution.

    Parameters
    ----------
    sfnp : noiseproj_onebin
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`.
    size : int
        Number of samples to draw.

    Returns
    -------
    np.ndarray
        Random samples drawn from the residual PSD posterior
        distribution, with length `size`.

    Raises
    ------
    ValueError
        If `sfnp` is ``None`` (i.e. decorrelation was not possible
        because the number of averages was not greater than the
        number of timeseries).
    """
    if sfnp is None:
        raise ValueError("Number of averages must be greater than number of ts.")

    PSDres_rvs = sfnp.PSDres_dist.rvs(size=size)
    return PSDres_rvs


def alpha_onebin_RVS(sfnp, size):
    """Draw random samples from the susceptibilities distribution.

    For the real case, samples directly from the multivariate
    distribution. For the complex case, samples the real and
    imaginary parts jointly (stacked as a real Student-t multivariate)
    and recombines them into complex susceptibilities.

    Parameters
    ----------
    sfnp : noiseproj_onebin
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`.
    size : int
        Number of samples to draw.

    Returns
    -------
    np.ndarray
        Random samples of the susceptibilities, with shape
        ``(size, r)``. Complex-valued if `sfnp.case` is
        ``"complex"``, real-valued otherwise.

    Raises
    ------
    ValueError
        If `sfnp` is ``None`` (i.e. decorrelation was not possible
        because the number of averages was not greater than the
        number of timeseries).
    """
    if sfnp is None:
        raise ValueError("Number of averages must be greater than number of ts.")

    if sfnp.case == "real":
        alphas_rvs = sfnp.alphasmultivar_dist.rvs(size=size)
    elif sfnp.case == "complex":
        realrvsStud = sfnp.alphasmultivar_dist.rvs(size=size)
        alphas_rvs = realrvsStud[:, : sfnp.r] + 1.0j * realrvsStud[:, sfnp.r :]
    return alphas_rvs


def PSDresidual_onebin_qnt(sfnp, q):
    """Compute the confidence interval of the residual noise PSD.

    Parameters
    ----------
    sfnp : noiseproj_onebin or None
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`. If ``None`` (decorrelation not possible
        for this frequency), a `UserWarning` is issued and a
        ``[nan, nan, nan]`` interval is returned instead of raising.
    q : float
        Lower-tail probability.

    Warns
    -----
    UserWarning
        If `sfnp` is ``None``, since decorrelation could not be
        performed for this frequency.
    """

    q = np.atleast_1d(q)
    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    if sfnp is None:
        warnings.warn(
            "Number of averages must be greater than number of ts.", UserWarning
        )
        return [np.nan, np.nan, np.nan]

    PSDres_qnt = [ sfnp.PSDres_dist.ppf(qtmp) for qtmp in q]
    return PSDres_qnt


def ASDresidual_onebin_qnt(sfnp, q):
    """Compute the confidence interval of the residual noise ASD.

    Derives the Amplitude Spectral Density confidence interval as the
    square root of the residual PSD confidence interval (see
    :func:`PSDresidual_onebin_qnt`).

    Parameters
    ----------
    sfnp : noiseproj_onebin or None
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`. If ``None``, propagates a
        ``[nan, nan, nan]`` interval (see :func:`PSDresidual_onebin_qnt`).
    q : float
        Lower-tail probability

    """

    PSDres_qnt = PSDresidual_onebin_qnt(sfnp, q)
    ASDres_qnt = np.sqrt(PSDres_qnt)
    return ASDres_qnt


def alpha_onebin_qnt(sfnp, q):
    """Compute confidence intervals of the projected susceptibilities.

    Parameters
    ----------
    sfnp : noiseproj_onebin
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`. Must not be ``None``: frequency bins where
        decorrelation was not possible are the caller's responsibility
        to skip (see :func:`alpha_qnt`).
    q : float or array_like of float
        Lower-tail probabilities, each in [0, 1].

    Returns
    -------
    alpre_qnt : np.ndarray, shape (r, nq)
        Quantiles of the real part of each susceptibility.
    alpim_qnt : np.ndarray, shape (r, nq), or None
        Quantiles of the imaginary part of each susceptibility.
        ``None`` if ``sfnp.case == "real"``.

    Raises
    ------
    ValueError
        If `sfnp` is ``None``, or if any element of `q` is outside [0, 1].
    """
    if sfnp is None:
        msg = (
            "sfnp is None: decorrelation was not possible for this bin "
            "(number of averages must exceed the number of ts). "
            "Skip these bins in the caller."
        )
        raise ValueError(msg)

    q = np.atleast_1d(q)
    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    alpre_qnt = np.asarray([sfnp.alpre_dist[i].ppf(q) for i in range(sfnp.r)])

    if sfnp.case == "complex":
        alpim_qnt = np.asarray([sfnp.alpim_dist[i].ppf(q) for i in range(sfnp.r)])
    else:
        alpim_qnt = None

    return alpre_qnt, alpim_qnt


def R2contrib_onebin_qnt(sfnp, q):
    """Compute the confidence interval of the contribution to R2.

    Estimates the expected multiple coherence `R2e` from the CPSD
    matrix, then derives the confidence interval of its contribution
    via :func:`stats_onebin.R2quantiles_one`.

    Parameters
    ----------
    sfnp : noiseproj_onebin or None
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`. If ``None`` (decorrelation not possible
        for this frequency), a `UserWarning` is issued and a
        ``[nan, nan, nan]`` interval is returned instead of raising.
    q : float
        Lower-tail probability.

    Warns
    -----
    UserWarning
        If `sfnp` is ``None``, since decorrelation could not be
        performed for this frequency.
    """

    q = np.atleast_1d(q)
    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    if sfnp is None:
        warnings.warn(
            "Number of averages must be greater than number of ts.", UserWarning
        )
        return [np.nan, np.nan, np.nan]

    R2exp = np.real(1 - 1 / (np.linalg.inv(sfnp.CPSDmat)[0, 0] * sfnp.CPSDmat[0, 0]))
    R2contrib_qnt = cpsd_stats_onebin.R2posterior_qnt_onebin(
        R2exp=R2exp, navs=sfnp.navs, p=sfnp.r + 1, q=q
    )

    return R2contrib_qnt
