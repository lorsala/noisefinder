import numpy as np
from .CPSDstats import CPSDstats,_getcohere,_getR2
from . import specwindows

class CPSD_WOSAmethod(CPSDstats):
    """
    Compute the one-sided Cross Power Spectral Density (CPSD) matrix,
    with the WOSA (Welch) frequency scheme.
    Compute cross-coherences, R2.

    Parameters
    ----------
    ts: list
        List of synchronously sampled time series.
    fs: float    
        Sampling frequency.
    nperseg : int
        Number of samples per stretch.
    win: Callable   
        Spectral window (function).
    olap: float
        Overlap among segments. Defaults to 0.50.
    detrend_c: bool
        If ``True``, subtract mean before performing fft. Defaults to ``False``.
    verbose : bool
    """
    def __init__(self,ts:list,nperseg:float,fs:float,win,olap=0.50,detrend_c=False,verbose:bool=False):
        super().__init__(ts=ts,fs=fs,detrend_c=detrend_c,win=win,olap=olap)
        self.nperseg=int(nperseg)
        self._CPSD_WOSA_eval()
        if(verbose):
            print(
                """**** CPSD_WOSAmethod init verbose ****
This object contains the CPSD evaluated with the Welch scheme:
linearly-spaced frequencies, with spacing {0:.2g}Hz.
You can access CPSD with getCPSD(), or access the PSD with getPSD().
You can perform noise projection with the class noiseproj, giving this object as parameter.
Be aware that, given the amount of frequencies, calculations may take a lot of time.
            """.format(self.freqs[1]-self.freqs[0]))

    def _CPSD_WOSA_eval(self,nocohere=False):
        self.CPSD, self.navs, self.freqs, _ = _evalWOSACPSD(datamat=self.datamat,nperseg=self.nperseg,fs=self.fs,win=self.win,
                                       olap=self.olap,detrend_c=self.detrend_c)
        self.Ls = self.freqs**0 * self.nperseg
        self.kcoeffs = np.arange(0,len(self.freqs),1)
        self.cohere = _getcohere(CPSD=self.CPSD)
        self.MSC = np.abs(self.cohere)**2
        self.R2 = _getR2(CPSD=self.CPSD,navs=self.navs)
        self.PSD = [np.real(self.CPSD[:,i,i]) for i in range(self.matp)]



def _evalWOSACPSD(datamat,nperseg,fs,win,olap,detrend_c):
    ndim = datamat.shape[0]
    npoints = datamat.shape[1]
    freqs = np.fft.rfftfreq(n=nperseg,d=1/fs)
    nfreqs = len(freqs)

    winpt = win(nperseg) #spectral window

    startpoint = 0
    tmpnavs = 0
    Amat = np.zeros((ndim,ndim,nfreqs),dtype=complex)
    periodograms = []
    while(1):
        endpoint = startpoint + nperseg
        if(endpoint > npoints):
            break
        xs0 = datamat[:,startpoint:endpoint] #multivariate stretch
        if detrend_c: xs0 = xs0 - np.mean(xs0,axis=1,keepdims=True)
        xs = winpt * xs0 #windowed multivariate stretch

        # use FFT, as it's more efficient with WOSA periodograms
        ax = np.asarray([np.fft.rfft(x) for x in xs])

        periodograms.append(ax)
        Amat += ax[:,None,:] * ax.conj()[None,:,:] #np.outer(ax,ax.conj()) #fill CPSD matrix
        tmpnavs += 1
        startpoint += int(nperseg*(1-olap))

    Amat = np.moveaxis(Amat,-1,0)
    # if tmpnavs==0: return Amat*0,0,[]
    wins2 = winpt@winpt
    tmpCPSD  = 2.0*Amat/tmpnavs/fs/wins2; #one-sided CPSD matrix
    periodograms  = np.asarray(periodograms)*np.sqrt(2.0/fs/wins2); #one-sided periodograms
    tmpnavs = freqs**0 * tmpnavs


    return tmpCPSD,tmpnavs,freqs,periodograms
        