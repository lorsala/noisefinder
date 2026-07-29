"""
Single-frequency noise projection.
"""

from typing import Any
import warnings
from dataclasses import dataclass
import numbers


import numpy as np
import scipy.stats as st

from .cpsd import stats_methods



def run_noiseproj_sf(CPSDmat: np.ndarray, nav: float, case="complex"):
    """Perform single-frequency noise projection (decorrelation).

    Computes the frozen posterior distributions of residual noise
    power and susceptibilities for a single frequency, by decorrelating
    the channels of the CPSD matrix. For multiple frequencies, use
    :func:`NoiseProj`, which builds an instance of :class:`noiseproj`.

    Parameters
    ----------
    CPSDmat : np.ndarray
        Single-frequency Cross Power Spectral Density matrix, square,
        with shape ``(p, p)``.
    nav : float
        Number of averaged periodograms. Must satisfy
        ``nav > p - 1`` (i.e. greater than the number of timeseries
        minus one) for decorrelation to be possible.
    case : str, optional
        Prior assumed on the susceptibilities: ``"complex"`` or
        ``"real"``. Defaults to ``"complex"``.

    Returns
    -------
    noiseprojsf.noiseproj_sf or None
        Instance of :class:`noiseprojsf.noiseproj_sf` containing the
        single-frequency noise projection results. Returns ``None``
        if `nav` is not large enough to allow decorrelation (a
        `UserWarning` is issued in that case).

    Raises
    ------
    ValueError
        If `CPSDmat` is not 2D, not square, if `case` is not
        ``"real"`` or ``"complex"``, or if there is only one
        timeseries (nothing to decorrelate).
    TypeError
        If `nav` is not a real number (e.g. if an array is passed
        instead).

    Warns
    -----
    UserWarning
        If `nav` is not greater than the number of timeseries minus
        one, decorrelation is not performed and ``None`` is returned.
    """

    if not CPSDmat.ndim == 2:
        msg = "CPSD matrix must be 2D (single-frequency CPSD matrix)."
        raise ValueError(msg)
    if not CPSDmat.shape[0] == CPSDmat.shape[1]:
        msg = "CPSD matrix must be square."
        raise ValueError(msg)
    if not isinstance(nav, numbers.Real):
        msg = "nav must be a real number, not an array."
        raise TypeError(msg)
    if case not in ("real","complex"):
        msg = "Case needs to be either real or complex."
        raise ValueError(msg)

    r = CPSDmat.shape[0] - 1

    if not r > 0:
        msg = "Just one timeseries, nothing to decorrelate."
        raise ValueError(msg)
    if not nav > r:
        warnings.warn(
            f"Number of averages must be greater than number of time series. nav={nav}, r={r}. Can't decorrelate this one.",
            UserWarning,
        )
        sfnp = None
        return sfnp

    # distinguish real and complex methods
    if case == "real":
        CPSDmat = np.real(CPSDmat)
        sfnp = _ExecuteNoiseProjectionReal(CPSDmat=CPSDmat, nav=nav, r=r, case=case)
    elif case == "complex":
        sfnp = _ExecuteNoiseProjectionComplex(CPSDmat=CPSDmat, nav=nav, r=r, case=case)

    return sfnp


def _ExecuteNoiseProjectionReal(CPSDmat, nav, r, case):
    """
    Execute noise projection for real susceptibilities.

    Parameters:
    -----------
    CPSDmat: np.ndarray
        CPSD matrix.
    nav:
        Number of averaged windows for spectral estimation.
    r:
        Number of decorrelated timeseries.
    case:
        Must be `"real"`

    Returns:
    ---------
    Instance of noisefinder.noiseproj_sf
    """
    if case != "real":
        msg = "Case must be real for real-susceptibility evaluation"
        raise ValueError(msg)
    if not CPSDmat.ndim==2:
        msg = "CPSD matrix must be bi-dimensional"
        raise ValueError(msg)
    if not CPSDmat.shape[0]==CPSDmat.shape[1]:
        msg = "CPSD matrix must be square."
        raise ValueError(msg)
    if not CPSDmat.shape[0]-1==r:
        msg = "Value of r is inconsistent with CPSD dimension."
        raise ValueError(msg)


    schur = np.real(1 / np.linalg.inv(CPSDmat)[0, 0])
    A = CPSDmat * nav
    A1y = A[0, 1:]
    Ayy = A[1:, 1:]

    nu = 2 * nav - r
    PSDres_dist = st.invgamma(a=nav - r / 2, scale=schur * nav)
    muvector = A1y @ np.linalg.inv(Ayy)
    covmatrix = np.linalg.inv(Ayy) * schur * nav / nu
    alpre_dist = [
        st.t(loc=mu, scale=std, df=nu)
        for mu, std in zip(muvector, np.sqrt(np.diag(covmatrix)))
    ]
    alpim_dist = None
    alphasmultivar_dist = st.multivariate_t(
        loc=muvector, shape=covmatrix, df=nu, allow_singular=True
    )
    sfnp = NoiseProjSfResults(
        CPSDmat=CPSDmat,
        nav=nav,
        r=r,
        nu=nu,
        case=case,
        PSDres_dist=PSDres_dist,
        alpre_dist=alpre_dist,
        alpim_dist=alpim_dist,
        alphasmultivar_dist=alphasmultivar_dist,
    )
    return sfnp


def _ExecuteNoiseProjectionComplex(CPSDmat, nav, r, case):
    """
    Execute noise projection for complex susceptibilities.

    Parameters:
    -----------
    CPSDmat: np.ndarray
        CPSD matrix.
    nav:
        Number of averaged windows for spectral estimation.
    r:
        Number of decorrelated timeseries.
    case:
        Must be `"complex"`

    Returns:
    ---------
    Instance of noisefinder.noiseproj_sf
    """
    if case != "complex":
        msg = "Case must be complex for complex-susceptibility evaluation"
        raise ValueError(msg)
    if not CPSDmat.ndim==2:
        msg = "CPSD matrix must be bi-dimensional"
        raise ValueError(msg)
    if not CPSDmat.shape[0]==CPSDmat.shape[1]:
        msg = "CPSD matrix must be square."
        raise ValueError(msg)
    if not CPSDmat.shape[0]-1==r:
        msg = "Value of r is inconsistent with CPSD dimension."
        raise ValueError(msg)

    schur = np.real(1 / np.linalg.inv(CPSDmat)[0, 0])
    A = CPSDmat * nav
    A1y = A[0, 1:]
    Ayy = A[1:, 1:]

    nu = 2 * nav - 2 * r
    PSDres_dist = st.invgamma(a=nav - r, scale=schur * nav)
    mucomplex = A1y @ np.linalg.inv(Ayy)
    muvector = np.block([np.real(mucomplex), np.imag(mucomplex)])
    covmatrix = (
        np.linalg.inv(
            np.block([[np.real(Ayy), np.imag(Ayy)], [-np.imag(Ayy), np.real(Ayy)]])
        )
        * schur
        * nav
        / nu
    )
    alphas_dist = [
        st.t(loc=mu, scale=std, df=nu)
        for mu, std in zip(muvector, np.sqrt(np.diag(covmatrix)))
    ]
    alpre_dist = [alphas_dist[i] for i in range(r)]
    alpim_dist = [alphas_dist[i + r] for i in range(r)]
    alphasmultivar_dist = st.multivariate_t(
        loc=muvector, shape=covmatrix, df=nu, allow_singular=True
    )
    sfnp = NoiseProjSfResults(
        CPSDmat=CPSDmat,
        nav=nav,
        r=r,
        nu=nu,
        case=case,
        PSDres_dist=PSDres_dist,
        alpre_dist=alpre_dist,
        alpim_dist=alpim_dist,
        alphasmultivar_dist=alphasmultivar_dist,
    )
    return sfnp


@dataclass(frozen=True, kw_only=True)
class NoiseProjSfResults:
    """Single-frequency noise projection (decorrelation) results.

    Contains the frozen posterior distributions of residual noise
    power and susceptibilities, for a single frequency, as computed
    by :func:`NoiseProjSf`.
    """

    CPSDmat: np.ndarray
    """Single-frequency Cross Power Spectral Density matrix used to
        compute this result, square, with shape ``(p, p)``."""
    nav: float
    """Number of averaged periodograms used for this frequency."""
    r: int
    """Number of susceptibilities (i.e. ``p - 1``, where `p` is the
        number of timeseries)."""
    nu: int
    """Degrees of freedom of the underlying posterior distributions."""
    case: str
    """Prior assumed on the susceptibilities: ``"complex"`` or
        ``"real"``."""
    PSDres_dist: Any
    """Frozen `scipy.stats` distribution of the residual noise PSD."""
    alpre_dist: list[Any]
    """Frozen `scipy.stats` distributions of the real part of each
        susceptibility, one per susceptibility (length `r`)."""
    alpim_dist: list[Any]
    """Frozen `scipy.stats` distributions of the imaginary part of
        each susceptibility, one per susceptibility (length `r`).
        Only meaningful when `case` is ``"complex"``."""
    alphasmultivar_dist: list[Any]
    """Frozen `scipy.stats` multivariate distribution of the
        susceptibilities, jointly."""


def sfPSDresidualRVS(sfnp, size):
    """Draw random samples from the residual PSD distribution.

    Parameters
    ----------
    sfnp : noiseproj_sf
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


def sfalphaRVS(sfnp, size):
    """Draw random samples from the susceptibilities distribution.

    For the real case, samples directly from the multivariate
    distribution. For the complex case, samples the real and
    imaginary parts jointly (stacked as a real Student-t multivariate)
    and recombines them into complex susceptibilities.

    Parameters
    ----------
    sfnp : noiseproj_sf
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


def sfPSDresidual_qnt(sfnp, q):
    """Compute the confidence interval of the residual noise PSD.

    Parameters
    ----------
    sfnp : noiseproj_sf or None
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


def sfASDresidual_qnt(sfnp, q):
    """Compute the confidence interval of the residual noise ASD.

    Derives the Amplitude Spectral Density confidence interval as the
    square root of the residual PSD confidence interval (see
    :func:`sfPSDresidual_qnt`).

    Parameters
    ----------
    sfnp : noiseproj_sf or None
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`. If ``None``, propagates a
        ``[nan, nan, nan]`` interval (see :func:`sfPSDresidual_qnt`).
    q : float
        Lower-tail probability

    """

    PSDres_qnt = sfPSDresidual_qnt(sfnp, q)
    ASDres_qnt = np.sqrt(PSDres_qnt)
    return ASDres_qnt


def sfalpProj_qnt(sfnp, q):
    """Compute confidence intervals of the projected susceptibilities.

    Parameters
    ----------
    sfnp : noiseproj_sf or None
        Single-frequency noise projection results, as returned by
        :func:`NoiseProjSf`. If ``None`` (decorrelation not possible
        for this frequency), a `UserWarning` is issued and
        ``nan``-filled intervals are returned instead of raising.
    q : float
        Lower-tail probability.

    Returns
    -------
    alpre_qnt : np.ndarray or list of float
        Quantile for the real part of each
        susceptibility.
    alpim_qnt : np.ndarray or list
        Quantile for the imaginary part of each
        susceptibility.

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
        return [np.nan], [np.nan]

    alpre_qnt = np.asarray(
        [
            [ sfnp.alpre_dist[i].ppf(qtmp) for qtmp in q]
            for i in range(sfnp.r)
        ]
    )

    if sfnp.case == "complex":
        alpim_qnt = np.asarray(
            [
                [ sfnp.alpim_dist[i].ppf(qtmp) for qtmp in q]
                for i in range(sfnp.r)
            ]
        )
    elif sfnp.case == "real":
        alpim_qnt = [None]

    return alpre_qnt, alpim_qnt


def sfR2contrib_qnt(sfnp, q):
    """Compute the confidence interval of the contribution to R2.

    Estimates the expected multiple coherence `R2e` from the CPSD
    matrix, then derives the confidence interval of its contribution
    via :func:`stats_methods.R2quantiles_one`.

    Parameters
    ----------
    sfnp : noiseproj_sf or None
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
    R2contrib_qnt = stats_methods.R2posterior_qnt_onebin(
        R2exp=R2exp, navs=sfnp.nav, p=sfnp.r + 1, q=q
    )

    return R2contrib_qnt
