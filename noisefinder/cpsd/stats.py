"""
Compute statistical properties of CPSD matrices:
Posterior PDFs, quantiles.
"""

import numpy as np

from . import stats_methods


def PSDposterior_dist(CPSDval, tsidx):
    """
    Returns frozen PSD posterior.

    Parameters
    ------------

    CPSDval: noisefinder.cpsd.CPSDresults
    tsidx: int
        Index of the timeseries of which evaluate the PSD.
    """
    if (tsidx >= CPSDval.matp or tsidx < 0):
        msg = "Invalid tsidx."
        raise ValueError(msg)

    PSDs = np.real(CPSDval.CPSD[:, tsidx, tsidx])
    navs = CPSDval.navs
    PSDpostfrozen = [
        stats_methods.PSDposterior_dist_onebin(PSDexp=PSD, navs=n)
        for PSD, n in zip(PSDs, navs)
    ]
    return PSDpostfrozen


def PSDposterior_pdf(CPSDval, tsidx, PSDaxis):
    """
    Returns frozen PSD posterior.

    Parameters
    ------------

    CPSDval: noisefinder.cpsd.CPSDresults
    tsidx: int
        Index of the timeseries of which evaluate the PSD.
    PSDaxis : ``numpy.ndarray``
        PSD axis on which the posterior is evaluated.
    """
    
    PSDpostfrozen = PSDposterior_dist(CPSDval, tsidx)
    PSDPDFs = [PSDdistrib.pdf(PSDaxis) for PSDdistrib in PSDpostfrozen]
    return PSDPDFs


def ASDposterior_pdf(CPSDval, tsidx, ASDaxis):
    """
    Returns frozen PSD posterior.

    Parameters
    ------------

    CPSDval: noisefinder.cpsd.CPSDresults
    tsidx: int
        Index of the timeseries of which evaluate the PSD.
    ASDaxis : ``numpy.ndarray``
        ASD axis on which the posterior is evaluated.
    """
    PSDPDFs = PSDposterior_pdf(CPSDval, tsidx, ASDaxis**2)
    ASDPDFs = [PSDPDF * 2 * ASDaxis for PSDPDF in PSDPDFs]
    return ASDPDFs


def MSCposterior_pdf(CPSDval, idx1, idx2, MSCthaxis):
    """
    Returns the MSC posterior.

    Parameters
    ----------
    CPSDval: noisefinder.cpsd.CPSDresults
    MSCthaxis : ``numpy.ndarray``
        MSCthaxis axis on which the posterior is evaluated.
    idx1,idx2 : int
        Indexes of the timeseries for MSC evaluation.
    """
    if not (idx1 != idx2 or idx1 < CPSDval.matp or idx2 < CPSDval.matp):
        msg = "Invalid indexes."
        raise ValueError(msg)
    MSCexp = CPSDval.MSC[:, idx1, idx2]
    PDFs = []
    for MSCexptmp, navstmp in zip(MSCexp, CPSDval.navs):
        PDFs.append(stats_methods.MSCposterior_onebin(MSCthaxis, MSCexptmp, navstmp))
    return PDFs


def R2posterior_pdf(CPSDval, R2thaxis):
    """
    Returns the multiple coherence R2 posterior.

    Parameters
    ----------
    CPSDval: noisefinder.cpsd.CPSDresults
    R2thaxis : ``numpy.ndarray``
        R2thaxis axis on which the posterior is evaluated.
    """
    PDFs = []
    for R2exptmp, navstmp in zip(CPSDval.R2, CPSDval.navs):
        PDFs.append(
            stats_methods.R2posterior_onebin(R2thaxis, R2exptmp, navstmp, CPSDval.matp)
        )
    return PDFs


def PSDposterior_qnt(CPSDval, q, tsidx=0):
    """
    Returns the posterior PSD quantiles, at the given Lower-tail probability

    Parameters
    ----------
    CPSDval: noisefinder.cpsd.CPSDresults
    q : float
        Lower-tail probability
    tsidx: int
        Index of the timeseries of which evaluate the PSD.
    """
    if (tsidx >= CPSDval.matp or tsidx < 0):
        msg = "Invalid tsidx."
        raise ValueError(msg)

    Sxx = CPSDval.PSD[tsidx]
    PSD_quantile = stats_methods.PSDposterior_qnt_onebin(Sxx, CPSDval.navs, q=q)
    PSD_quantile = np.asarray(PSD_quantile)
    return PSD_quantile

def ASDposterior_qnt(CPSDval, q, tsidx=0):
    """
    Returns the posterior ASD quantiles, at the given Lower-tail probability

    Parameters
    ----------
    CPSDval: noisefinder.cpsd.CPSDresults
    q : float
        Lower-tail probability
    tsidx: int
        Index of the timeseries of which evaluate the PSD.
    """

    PSD_quantile = PSDposterior_qnt(CPSDval=CPSDval, q=q, tsidx=tsidx)
    ASD_quantile = np.sqrt(PSD_quantile)
    return ASD_quantile


def MSCposterior_qnt(CPSDval, q, idx1, idx2):
    """
    Returns the posterior MSC quantiles, at the given Lower-tail probability

    Parameters
    ----------
    CPSDval: noisefinder.cpsd.CPSDresults
    q : float
        Lower-tail probability
    idx1,idx2 : int
        Indexes of the timeseries for MSC evaluation.
    """
    MSC = CPSDval.MSC[:, idx1, idx2]
    MSCquantile = np.asarray(
        [
            stats_methods.MSCposterior_qnt_onebin(MSCexp=r2e, navs=M, q=q)
            for r2e, M in zip(MSC, CPSDval.navs)
        ]
    )
    return MSCquantile


def R2posterior_qnt(CPSDval, q):
    """
    Returns the posterior R2 quantiles, at the given Lower-tail probability

    Parameters
    ----------
    CPSDval: noisefinder.cpsd.CPSDresults
    q : float
        Lower-tail probability
    """
    R2quantile = np.asarray(
        [
            stats_methods.R2posterior_qnt_onebin(R2e, navs=M, p=CPSDval.matp, q=q)
            for R2e, M in zip(CPSDval.R2, CPSDval.navs)
        ]
    )
    return R2quantile
