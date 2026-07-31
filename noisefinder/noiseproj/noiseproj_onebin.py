"""
Single-frequency noise projection.
"""

from typing import Any
import warnings
from dataclasses import dataclass
import numbers


import numpy as np
import scipy.stats as st



def _run_noiseproj_onebin(CPSDmat: np.ndarray, navs: float, case="complex"):
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
    navs : float
        Number of averaged periodograms. Must satisfy
        ``navs > p - 1`` (i.e. greater than the number of timeseries
        minus one) for decorrelation to be possible.
    case : str, optional
        Prior assumed on the susceptibilities: ``"complex"`` or
        ``"real"``. Defaults to ``"complex"``.

    Returns
    -------
    noiseproj_onebin.NoiseProjSfResults or None
        Instance of :class:`noiseproj_onebin.NoiseProjSfResults` containing the
        single-frequency noise projection results. Returns ``None``
        if `navs` is not large enough to allow decorrelation (a
        `UserWarning` is issued in that case).

    Raises
    ------
    ValueError
        If `CPSDmat` is not 2D, not square, if `case` is not
        ``"real"`` or ``"complex"``, or if there is only one
        timeseries (nothing to decorrelate).
    TypeError
        If `navs` is not a real number (e.g. if an array is passed
        instead).

    Warns
    -----
    UserWarning
        If `navs` is not greater than the number of timeseries minus
        one, decorrelation is not performed and ``None`` is returned.
    """

    _noiseproj_onebin_sanity(CPSDmat=CPSDmat,navs=navs,case=case)

    r = CPSDmat.shape[0] - 1

    if not r > 0:
        msg = "Just one timeseries, nothing to decorrelate."
        raise ValueError(msg)
    if not navs > r:
        warnings.warn(
            f"Number of averages must be greater than number of time series. navs={navs}, r={r}. Can't decorrelate this one.",
            UserWarning,
        )
        sfnp = None
        return sfnp

    # distinguish real and complex methods
    if case == "real":
        CPSDmat = np.real(CPSDmat)
        sfnp = _ExecuteNoiseProjectionReal(CPSDmat=CPSDmat, navs=navs, r=r, case=case)
    elif case == "complex":
        sfnp = _ExecuteNoiseProjectionComplex(CPSDmat=CPSDmat, navs=navs, r=r, case=case)

    return sfnp

def _noiseproj_onebin_sanity(CPSDmat: np.ndarray, navs: float, case="complex"):
    # --- CPSDmat: 2-D square matrix ---
    if not isinstance(CPSDmat, np.ndarray):
        raise TypeError(f"'CPSDmat' must be a numpy.ndarray, got {type(CPSDmat).__name__}")
    if CPSDmat.ndim != 2:
        raise ValueError(f"'CPSDmat' must be 2-D, got shape {CPSDmat.shape}")
    if CPSDmat.shape[0] != CPSDmat.shape[1]:
        raise ValueError(f"'CPSDmat' must be square, got shape {CPSDmat.shape}")

    # --- navs: positive integer (allowed as a number, e.g. float(3.0)) ---
    if not isinstance(navs, (int, float, np.integer, np.floating)) or isinstance(navs, bool):
        raise TypeError(f"'navs' must be a number, got {type(navs).__name__}")
    if navs <= 0:
        raise ValueError(f"'navs' must be positive, got {navs}")
    if navs != int(navs):
        raise ValueError(f"'navs' must be an integer value, got {navs}")

    # --- case: "real" or "complex" ---
    if case not in ("real", "complex"):
        raise ValueError(f"'case' must be 'real' or 'complex', got {case!r}")

    # --- symmetry / non-negative-definiteness ---
    tol = 1e-8
    if not np.allclose(CPSDmat, CPSDmat.conj().T, atol=tol):
        raise ValueError("'CPSDmat' must be Hermitian (symmetric)")

    eigvals = np.linalg.eigvalsh(CPSDmat)
    if np.any(eigvals < -tol):
        raise ValueError(
            f"'CPSDmat' must be non-negative-definite, "
            f"got minimum eigenvalue {eigvals.min()}"
        )

def _ExecuteNoiseProjectionReal(CPSDmat, navs, r, case):
    """
    Execute noise projection for real susceptibilities.

    Parameters:
    -----------
    CPSDmat: np.ndarray
        CPSD matrix.
    navs:
        Number of averaged windows for spectral estimation.
    r:
        Number of decorrelated timeseries.
    case:
        Must be `"real"`

    Returns:
    ---------
    Instance of noisefinder.NoiseProjSfResults
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
    A = CPSDmat * navs
    A1y = A[0, 1:]
    Ayy = A[1:, 1:]

    nu = 2 * navs - r
    PSDres_dist = st.invgamma(a=navs - r / 2, scale=schur * navs)
    muvector = A1y @ np.linalg.inv(Ayy)
    covmatrix = np.linalg.inv(Ayy) * schur * navs / nu
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
        navs=navs,
        r=r,
        nu=nu,
        case=case,
        PSDres_dist=PSDres_dist,
        alpre_dist=alpre_dist,
        alpim_dist=alpim_dist,
        alphasmultivar_dist=alphasmultivar_dist,
    )
    return sfnp


def _ExecuteNoiseProjectionComplex(CPSDmat, navs, r, case):
    """
    Execute noise projection for complex susceptibilities.

    Parameters:
    -----------
    CPSDmat: np.ndarray
        CPSD matrix.
    navs:
        Number of averaged windows for spectral estimation.
    r:
        Number of decorrelated timeseries.
    case:
        Must be `"complex"`

    Returns:
    ---------
    Instance of noisefinder.NoiseProjSfResults
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
    A = CPSDmat * navs
    A1y = A[0, 1:]
    Ayy = A[1:, 1:]

    nu = 2 * navs - 2 * r
    PSDres_dist = st.invgamma(a=navs - r, scale=schur * navs)
    mucomplex = A1y @ np.linalg.inv(Ayy)
    muvector = np.block([np.real(mucomplex), np.imag(mucomplex)])
    covmatrix = (
        np.linalg.inv(
            np.block([[np.real(Ayy), np.imag(Ayy)], [-np.imag(Ayy), np.real(Ayy)]])
        )
        * schur
        * navs
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
        navs=navs,
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
    power and susceptibilities, for a single frequency.
    """

    CPSDmat: np.ndarray
    """Single-frequency Cross Power Spectral Density matrix used to
        compute this result, square, with shape ``(p, p)``."""
    navs: float
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

