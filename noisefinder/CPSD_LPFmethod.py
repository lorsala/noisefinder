import numpy as np
from .CPSDstats import CPSDstats
from . import specwindows

class CPSD_LPFmethod(CPSDstats):
    """
    Compute the one-sided Cross Power Spectral Density (CPSD) matrix,
    with the LPF frequency scheme, the BH92 spectral window, and 50% overlap.
    Compute cross-coherences, R2.

    Parameters
    ----------
    ts: list
        List of synchronously sampled time series.
    fs: float    
        Sampling frequency.
    Tmax : float
        Maximum length of stretch (used at the lowest frequency).
    fmax : float
        Maximum frequency. If ``None``, uses Nyquist.
    detrend_c: bool
        If ``True``, subtract mean before performing fft. Defaults to ``False``.
    verbose : bool
    """
    def __init__(self,ts,Tmax,fmax,fs,detrend_c=False,verbose=False):
        super().__init__(ts=ts,fs=fs,win=specwindows.BH92,olap=0.50,detrend_c=detrend_c)
        freqs,Ls,kcoeffs = self._LPFfreqscheme(Nmax=Tmax*fs,fmax=fmax,fs=fs)
        self.CPSD_eval(freqs,Ls,kcoeffs)
        if(verbose):
            print(
                """**** CPSD_LPFmethod init verbose ****
This object contains the CPSD evaluated with the LPF scheme:
a few frequencies, logarithmically-spaced with r=5/3, starting from {0:.2g}Hz.
This scheme enforces the use of a BH92 window, 50% overlap, and frequency index k=4.
You can access CPSD with getCPSD(), or access the PSD with getPSD().
You can perform noise projection with the class noiseproj, giving this object as parameter.
            """.format(self.freqs[0]))
        
    def _LPFfreqscheme(self,Nmax,fmax,fs):
        fmax = fs/2 if fmax is None else fmax #set fmax
        fmax = fs/2 if fmax>fs/2 else fmax #set fmax, max Nyquist
        k0 = 4
        k = 8
        r = 3/5
        f = [k0/(Nmax/fs)]
        Ls = [Nmax]
        kcoeffs = [k0]
        aa=2
        while(1):
            tmpL = np.floor(r**(aa-2) * Nmax)
            tmpf = k/(tmpL/fs)
            if (tmpf>fmax):
                break
            Ls.append(tmpL)
            f.append(tmpf)
            kcoeffs.append(k)
            aa += 1
        return np.asarray(f),np.asarray(Ls,dtype=int),np.asarray(kcoeffs,dtype=int)
        
        
        
