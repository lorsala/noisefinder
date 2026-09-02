"""
Multi-frequency noise projection.
"""

from dataclasses import dataclass
import numpy as np

from . import noiseproj_onebin


def run_noiseproj(CPSDmat: np.ndarray, navs: np.ndarray, case="complex"):
    """Perform noise projection (decorrelation) across all frequencies.

    For each frequency, computes the posterior distributions of
    residual noise power and susceptibilities by decorrelating the
    channels of the CPSD matrix. Internally builds one
    :class:`noiseproj_onebin.NoiseProjSfResults` instance per frequency and
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

    # --- CPSDmat: 3-D ndarray, square in last two dims ---
    if not isinstance(CPSDmat, np.ndarray):
        raise TypeError(f"'CPSDmat' must be a numpy.ndarray, got {type(CPSDmat).__name__}")
    if CPSDmat.ndim != 3:
        raise ValueError(f"'CPSDmat' must be 3-D, got shape {CPSDmat.shape}")
    if CPSDmat.shape[1] != CPSDmat.shape[2]:
        raise ValueError(
            f"'CPSDmat' last two dimensions must be equal (square matrices), "
            f"got shape {CPSDmat.shape}"
        )
    if not CPSDmat.shape[1] > 1:
        msg = "Just one timeseries, nothing to decorrelate."
        raise ValueError(msg)

    # --- navs: ndarray of positive ints, same length as CPSDmat's first dim ---
    if not isinstance(navs, np.ndarray):
        raise TypeError(f"'navs' must be a numpy.ndarray, got {type(navs).__name__}")
    if not np.issubdtype(navs.dtype, np.integer):
        raise TypeError(f"'navs' must have an integer dtype, got {navs.dtype}")
    if np.any(navs < 0):
        raise ValueError(f"'navs' must not contain negative values, got min {navs.min()}")
    if navs.shape[0] != CPSDmat.shape[0]:
        raise ValueError(
            f"'navs' length ({navs.shape[0]}) must match CPSDmat's first dimension "
            f"({CPSDmat.shape[0]})"
        )

    # --- case: "real" or "complex" ---
    if case not in ("real", "complex"):
        raise ValueError(f"'case' must be 'real' or 'complex', got {case!r}")

    sfnp_arr = [
        noiseproj_onebin._run_noiseproj_onebin(CPSDmat=CPSDmat[ffi, :, :], navs=navs[ffi], case=case)
        if navs[ffi] != 0 else None
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

    sfnp_arr: list[noiseproj_onebin.NoiseProjSfResults]
    """List of single-frequency noise projection results, ordered by
        frequency, as returned by :class:`noiseproj_onebin.NoiseProjSf`."""


