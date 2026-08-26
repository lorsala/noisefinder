"""
Module with backend posterior calculation, 
and confidence interval calculation.
Relies on distributions in scipy.stats.
"""

import warnings

import numpy as np
import scipy.special as sp
import scipy.stats as st
import scipy.integrate
import mpmath

from typing import Literal


def _loghyp2f1_eff(
    a, b, c, z,
    hyp2f1_mode: Literal["lowres", "varres", "highres"] = "varres",
):
    """
    Returns log(hyp2f1). If res_mode='lowres', it may be inaccurate.
    However, high resolution takes a long time.
    To be improved in the future.

    Parameters
    -------------

    a,b,c,z: float
        See hyp2f1 parameters.
    hyp2f1_mode : str
        Resolution mode. One of ``"lowres"``, ``"varres"``, or ``"highres"``.
    """

    if hyp2f1_mode == "lowres":
        res = np.log(sp.hyp2f1(a, b, c, z))
        return res
    elif hyp2f1_mode == "highres":
        res = mpmath.log(mpmath.hyp2f1(a, b, c, z, maxterms=1e6))
        return res
    elif hyp2f1_mode == "varres":
        res = np.log(sp.hyp2f1(a, b, c, z))
        if np.isinf(res) or np.isnan(res):
            res = mpmath.log(mpmath.hyp2f1(a, b, c, z, maxterms=1e6))
        return res
    else:
        msg = "hyp2f1_mode must be 'lowres', 'highres', or 'varres'."
        raise ValueError(msg)


def PSDposterior_dist_onebin(PSDexp, navs):
    """
    Posterior of the PSD distribution, as instance of stats.invgamma.
    This assumes Jeffrey's prior. See paper.

    Parameters
    -------------

    PSDexp: float
        Experimental, measured PSD.
    navs: int
        Number of averaged windows.
    """
    return st.invgamma(a=navs, scale=PSDexp * navs)

def PSDposterior_onebin(PSDexp, PSDth, navs):
    """
    Posterior of the PSD distribution, evaluated at PSDth.
    This assumes Jeffrey's prior. See paper.

    Parameters
    -------------

    PSDexp: float
        Experimental, measured PSD.
    PSDth: float
        Theoretical PSD axis, for pdf evaluation.
    navs: int
        Number of averaged windows.
    """
    return PSDposterior_dist_onebin(PSDexp,navs).pdf(x=PSDth)


def PSDposterior_qnt_onebin(expPSD, navs, q):
    """
    Posterior quantiles of the PSD distribution, as instance of stats.invgamma.
    This assumes Jeffrey's prior. See paper.

    Parameters
    -------------

    expPSD: float
        Experimental, measured PSD.
    navs: int
        Number of averaged windows.
    q: list[float],np.ndarray
        Lower-tail probability.
    """
    q = np.atleast_1d(q)
    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    PSDq = [navs * expPSD / sp.gammainccinv(navs, qtmp) for qtmp in q]
    return np.asarray(PSDq)


def R2posterior_onebin(R2th, R2exp, navs, p):
    """
    Posterior distribution of the R2 distribution, evaluated at R2th.
    This assumes flat prior. See paper.
    Note that hyp2f1 diverges for R2-->1 and navs-->inf,
    a future release should treat this better.

    Parameters
    -------------

    R2th: np.ndarray
        Theoretical R2, independent variable for PDF calculation.
    R2exp: np.ndarray
        Experimental R2, dependent variable for PDF calculation.
    navs: int
        Number of averaged windows.
    p: int
        Number of timeseries.
    """

    if not np.isfinite(R2exp) or navs < p:
        warnings.warn(
            "Invalid input: can't evaluate posterior if R2exp is not finite and navs is not >= p; "
            "returning NaN."
        )
        return np.full_like(R2th, np.nan, dtype=float)

    
    logpdf = []
    for R2thtmp in R2th:
        logpdf.append(
            (navs) * np.log(1 - R2thtmp)
            + _loghyp2f1_eff(navs, navs, p - 1, R2exp * R2thtmp)
        )
    maxlogpdf = max(logpdf) - np.log(1e50)  # just to avoid overflow
    logpdf = [float(lp - maxlogpdf) for lp in logpdf]
    PDF = np.exp(logpdf)
    PDF = PDF / scipy.integrate.trapezoid(x=R2th, y=PDF)
    return PDF


def MSCposterior_onebin(MSCth, MSCexp, navs):
    """
    Posterior distribution of the MSC distribution, evaluated at MSCth.
    This assumes flat prior. See paper.
    Note that hyp2f1 diverges for MSC-->1 and navs-->inf,
    a future release should treat this better.

    Parameters
    -------------

    MSCth: np.ndarray
        Theoretical MSC, independent variable for PDF calculation.
    MSCexp: np.ndarray
        Experimental MSC, dependent variable for PDF calculation.
    navs: int
        Number of averaged windows.
    """
    return R2posterior_onebin(R2th=MSCth, R2exp=MSCexp, navs=navs, p=2)


def MSCposterior_qnt_onebin(MSCexp, navs, q):
    """
    Posterior quantiles of the MSC distribution, evaluated at MSCth.
    This assumes flat prior. See paper.
    Note that hyp2f1 diverges for MSC-->1 and navs-->inf,
    a future release should treat this better.

    Parameters
    -------------

    MSCexp: np.ndarray
        Experimental MSC, dependent variable for PDF calculation.
    navs: int
        Number of averaged windows.
    q: list[float],np.ndarray
        Lower-tail probability
    """
    q = np.atleast_1d(q)

    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    if not np.isfinite(MSCexp) or navs < 2:
        warnings.warn(
            "Invalid input: MSCexp must be finite and navs must be >= 2; "
            "returning NaN."
        )
        return np.full_like(q, np.nan, dtype=float)

    if navs > 1000 and MSCexp > 0.5:
        warnings.warn(
            f"navs={navs:d}, evaluation can take a long time for MSC and R2, especially if there's high correlation."
        )
    quantile = _distquantile(pdffunc=MSCposterior_onebin, valexp=MSCexp, q=q, navs=navs)
    return np.asarray(quantile)


def R2posterior_qnt_onebin(R2exp, navs, q, p):
    """
    Posterior quantiles of the MSC distribution, evaluated at MSCth.
    This assumes flat prior. See paper.
    Note that hyp2f1 diverges for MSC-->1 and navs-->inf,
    a future release should treat this better.

    Parameters
    -------------

    R2exp: np.ndarray
        Experimental MSC, dependent variable for PDF calculation.
    navs: int
        Number of averaged windows.
    q: list[float],np.ndarray
        Lower-tail probability
    """
    q = np.atleast_1d(q)

    if np.any((q < 0) | (q > 1)):
        msg = "q must be between 0 and 1."
        raise ValueError(msg)

    if not np.isfinite(R2exp) or navs < p:
        warnings.warn(
            "Invalid input: can't evaluate posterior if R2exp is not finite and navs is not >= p; "
            "returning NaN."
        )
        return np.full_like(q, np.nan, dtype=float)

    if navs > 1000 and R2exp > 0.5:
        warnings.warn(
            f"navs={navs:d}, evaluation can take a long time for MSC and R2, especially if there's high correlation."
        )
    quantile = _distquantile(pdffunc=R2posterior_onebin, valexp=R2exp, q=q, navs=navs, p=p)
    return np.asarray(quantile)


def _distquantile(pdffunc, valexp, q, **kwargs):
    """Compute the quantile of a pdf function at a given q.

    Parameters
    ----------
    pdffunc : callable
        Probability density function to evaluate, with signature
        ``pdffunc(x, valexp, **kwargs) -> array``.
    valexp : float
        Expected value passed to `pdffunc`.
    q : list[float],np.ndarray
        Lower-tail probability.
    **kwargs
        Additional keyword arguments passed directly to `pdffunc`
        (e.g. model-specific parameters for the chosen pdf).

    Returns
    -------
    float
        Quantile corresponding to the lower tail probability q.
    """
    x = np.linspace(1e-6, 1 - 1e-6, 500)
    PDF = pdffunc(x, valexp, **kwargs)
    CDF = scipy.integrate.cumulative_trapezoid(x=x, y=PDF)
    CDF = CDF / CDF[-1]
    PPF = [ x[np.argmax(CDF > qtmp)] for qtmp in q]
    return PPF