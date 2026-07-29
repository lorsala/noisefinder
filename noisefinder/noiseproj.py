"""
Multi-frequency noise projection.
"""

from dataclasses import dataclass
import numpy as np

from . import noiseprojsf


def run_noiseproj(CPSDmat: np.ndarray, navs: np.ndarray, case="complex"):
    """Perform noise projection (decorrelation) across all frequencies.

    For each frequency, computes the posterior distributions of
    residual noise power and susceptibilities by decorrelating the
    channels of the CPSD matrix. Internally builds one
    :class:`noiseprojsf.NoiseProjSfResults` instance per frequency and
    collects them into a :class:`noiseproj`.


    Parameters
    ----------
    CPSDmat : np.ndarray
        Multi-frequency Cross Power Spectral Density matrix, with
        shape ``(n_freqs, p, p)``.
    navs : np.ndarray
        Number of averaged periodograms at each frequency, with shape
        ``(n_freqs,)``. Each entry must satisfy ``navs[i] >= p``.
    case : str, optional
        Prior assumed on the susceptibilities: ``"complex"`` or
        ``"real"``. Defaults to ``"complex"``.

    Returns
    -------
    noiseproj
        Instance of :class:`noiseproj` containing the per-frequency
        noise projection results.

    Raises
    ------
    ValueError
        If `CPSDmat` is not 3D, not square in its last two axes, if
        `navs` does not have one entry per frequency, if `case` is
        not ``"real"`` or ``"complex"``, or if there is only one
        timeseries (nothing to decorrelate).
    TypeError
        If `navs` is not a NumPy array.
    """

    if not CPSDmat.ndim == 3:
        msg = "CPSD matrix must be 3D (multi-frequency CPSD matrix). First axis is frequency axis."
        raise ValueError(msg)
    if not CPSDmat.shape[1] == CPSDmat.shape[2]:
        msg = "CPSD matrix must be square."
        raise ValueError(msg)
    if not isinstance(navs, np.ndarray):
        msg = "nav must be an array."
        raise TypeError(msg)
    if not CPSDmat.shape[0] == navs.shape[0]:
        msg = (
            "navs dimension must coincide with CPSD dimension (number of frequencies)."
        )
        raise ValueError(msg)
    if case not in ("real", "complex"):
        msg = "Case needs to be either real or complex."
        raise ValueError(msg)

    r = CPSDmat.shape[1] - 1

    if not r > 0:
        msg = "Just one timeseries, nothing to decorrelate."
        raise ValueError(msg)

    sfnp_arr = [
        noiseprojsf.run_noiseproj_sf(CPSDmat=CPSDmat[ffi, :, :], nav=navs[ffi], case=case)
        for ffi in range(len(navs))
    ]

    sfnp_arr = NoiseProjResults(
        sfnp_arr=sfnp_arr,
    )

    return sfnp_arr


@dataclass(frozen=True, kw_only=True)
class NoiseProjResults:
    """Multi-frequency noise projection (decorrelation) results.

    Collects the single-frequency noise projection results produced
    by :func:`NoiseProj`, one per analyzed frequency.

    Parameters
    ----------
    sfnp_arr : list of noiseprojsf.noiseproj_sf
    """

    sfnp_arr: list[noiseprojsf.NoiseProjSfResults]
    """List of single-frequency noise projection results, ordered by
        frequency, as returned by :class:`noiseprojsf.NoiseProjSf`."""


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

    PSDres_qnt = [noiseprojsf.sfPSDresidual_qnt(sfnp, q) for sfnp in mfnp.sfnp_arr]
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


def alpProj_qnt(mfnp, q):
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

    alp_qnt = [noiseprojsf.sfalpProj_qnt(sfnp, q) for sfnp in mfnp.sfnp_arr]
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

    R2contrib_qnt = [noiseprojsf.sfR2contrib_qnt(sfnp, q) for sfnp in mfnp.sfnp_arr]
    return np.asarray(R2contrib_qnt).T
