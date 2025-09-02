import numpy as np
from . import CPSDstats_methods
from . import specwindows
import warnings


class CPSDstats():
    """
    Compute the one-sided Cross Power Spectral Density (CPSD) matrix.
    This is the base class, it does not include a frequency scheme. 
    The user should usually initialize with a child class implementing a frequency scheme, such as :class:`noisefinder.CPSD_LPFmethod` or :class:`noisefinder.CPSD_WOSAmethod`. 
    Then, use methods from this class to get statistics.
    This class evaluates the CPSD matrix at the given frequencies with the given stretch lengths, averaging over periodograms. It evaluates cross-coherences, and multiple coherence R2.

    Parameters
    -------------

    ts: list
        List of synchronously sampled time series.
    fs: float    
        Sampling frequency.
    win: Callable   
        Spectral window (function).
    olap: float
        Overlap among segments, 1 means complete overlap. Defaults to 0.50.
    detrend_c: bool
        If ``True``, subtract mean before performing fft. Defaults to ``False``.
    winscheme : Callable
        Allows the user to use different spectral windows at different frequencies. Defaults to ``None``.
    verbose : bool

    Attributes
    ------------

    self.datamat, self.fs, self.olap, self.win, self.detrend_c, self.winscheme:
        See Parameters
    self.matp:
        Number of measurements, matrix dimension.
    self.N:
        Number of datapoints.
    self.freqs:
        Only available after calling :func:`CPSDstats.CPSD_eval`, PSD frequencies.
    self.Ls:
        Only available after calling :func:`CPSDstats.CPSD_eval`, stretch lengths.
    self.kcoeffs:
        Only available after calling :func:`CPSDstats.CPSD_eval`, DFT frequency coefficients.
    self.CPSD:
        Only available after calling :func:`CPSDstats.CPSD_eval`, CPSD matrices.
    self.navs:
        Only available after calling :func:`CPSDstats.CPSD_eval`, number of averaged periodograms.
    self.cohere:
        Only available after calling :func:`CPSDstats.CPSD_eval`, coherence matrix.
    self.MSC:
        Only available after calling :func:`CPSDstats.CPSD_eval`, MSC matrix.
    self.R2:
        Only available after calling :func:`CPSDstats.CPSD_eval`, multiple coherence.
    self.PSD:
        Only available after calling :func:`CPSDstats.CPSD_eval`, PSDs of the input timeseries.
    """
    def __init__(self,ts,fs,win,olap=0.50,detrend_c=False,winscheme=None,verbose=False):
        self._sanitycheck(ts)
        self.datamat = np.asarray(ts)
        if(self.datamat.ndim==1):
            self.datamat = np.reshape(self.datamat,(1,self.datamat.size))
        self.fs = fs
        self.matp = self.datamat.shape[0]
        self.N = self.datamat.shape[1]
        assert olap<=1, "Overlap must be less than 1."
        self.olap = olap
        self.win = win
        self.detrend_c = detrend_c
        self.winscheme = winscheme
        if(verbose):
            print(
                """**** CPSDstats init verbose ****
This object is ready to calculate CPSDs, but it lacks a frequency scheme.
If this is not what you want, please use another class which inherits from this one.
            """)

    def CPSD_eval(self,freqs,Ls,kcoeffs):
        """
        Evaluates the CPSD at the input frequencies, with stretch length in input. For efficiency, this does not use fft but singularly computer the Fourier coefficients.
        This function re-calculates coherences, and R2.

        Parameters
        ----------
        freqs : ``numpy.ndarray``
            Frequencies. This parameter is just to be checked.
        Ls : array of int
            Stretch length, as samples per stretch.
        kcoeffs : array of int
            Frequency indexes, must obey ``k=f*L/f_s``.
        """
        assert(freqs[-1]<=self.fs/2), "Can't have a frequency greater than Nyquist."
        self.Ls = Ls
        self.kcoeffs = kcoeffs
        assert np.all(np.isclose(self.kcoeffs,np.floor(self.kcoeffs))), "kcoeffs must be an array of integers" #Just make sure this is an array of integers
        assert np.all(np.isclose(self.Ls,np.floor(self.Ls))), "Ls must be an array of integers"  #Just make sure this is an array of integers

        freqout = self.kcoeffs*self.fs/self.Ls
        if not np.all(np.isclose(freqs,freqout)):
            warnings.warn("The frequencies you passed are not consistent. I'm overriding with correct values. Please check.")

        self.freqs = freqout

        self.CPSD, self.navs = _evalCPSD(datamat=self.datamat,Ls=self.Ls,kcoeffs=self.kcoeffs,fs=self.fs,win=self.win,
                                       olap=self.olap,detrend_c=self.detrend_c,winscheme=self.winscheme)
        self.cohere = _getcohere(CPSD=self.CPSD)
        self.MSC = np.abs(self.cohere)**2
        self.R2 = _getR2(CPSD=self.CPSD,navs=self.navs)
        self.PSD = [np.real(self.CPSD[:,i,i]) for i in range(self.matp)]

        
    def _sanitycheck(self,ts:list):
        assert isinstance(ts,list), "input timeseries must be given as a list"
        assert np.all([ts_tmp.shape==ts[0].shape for ts_tmp in ts]), "lists have different amount of data"
        return

    def _checkCPSD(self):
        assert hasattr(self, 'CPSD'), "'CPSD' attribute is missing. First you need to evaluate it."
        assert hasattr(self, 'navs'), "'navs' attribute is missing. First you need to evaluate it."
        
    def merge(self,other):
        """
        Merges two CPSD classes, with a weighted average based on the number of periodograms available. Obviously, the CPSDs must be evaluated at the same frequencies.
        """
        self._checkCPSD()
        other._checkCPSD()
        assert np.all(self.fs==other.fs)
        assert np.all(self.Ls==other.Ls)
        assert np.all(self.CPSD.shape==other.CPSD.shape)
        #set CPSD. note that nan_to_num sets nans to zero so that we can sum even if PSD is nondefined (e.g. navs=0)
        self.CPSD = (np.nan_to_num(self.CPSD)*self.navs[:,None,None]+np.nan_to_num(other.CPSD)*other.navs[:,None,None])/(self.navs[:,None,None]+other.navs[:,None,None])
        self.navs = self.navs+other.navs
        self.cohere = _getcohere(CPSD=self.CPSD)
        self.MSC = np.abs(self.cohere)**2
        self.R2 = _getR2(CPSD=self.CPSD,navs=self.navs)
        self.PSD = [np.real(self.CPSD[:,i,i]) for i in range(self.matp)]
        
    def getCPSD(self): 
        """
        Returns: frequencies, CPSD matrix, number of averaging windows.
        """
        self._checkCPSD()
        return self.freqs,self.CPSD,self.navs
    def getPSD(self): 
        """
        Returns: frequencies, PSDs, number of averaging windows.
        """
        self._checkCPSD()
        return self.freqs,self.PSD,self.navs
    def getMSC(self):
        """
        Returns: frequencies, MSC, R2, number of averaging windows.
        """
        self._checkCPSD()
        return self.freqs,self.MSC,self.R2,self.navs
    
    #add stats methods to base class
    def PSDposterior_eval(self,PSDaxis,tsidx):
        """
        Returns the PSD posterior.

        Parameters
        ----------
        PSDaxis : ``numpy.ndarray``
            PSD axis on which the posterior is evaluated.
        tsidx : int
            Index of the timeseries for PSD evaluation.
        """
        self._checkCPSD()
        assert tsidx<self.matp
        PSDs = np.real(self.CPSD[:,tsidx,tsidx])
        navs = self.navs
        PSDPDFs = [CPSDstats_methods.PSDposterior(expPSD=PSD,n=n).pdf(PSDaxis) for PSD,n in zip(PSDs,navs)]
        return PSDPDFs
    def ASDposterior_eval(self,ASDaxis,tsidx):
        """
        Returns the ASD posterior.

        Parameters
        ----------
        ASDaxis : ``numpy.ndarray``
            ASD axis on which the posterior is evaluated.
        tsidx : int
            Index of the timeseries for PSD evaluation.
        """
        PSDPDFs = self.PSDposterior_eval(ASDaxis**2,tsidx)
        ASDPDFs = [PSDPDF*2*ASDaxis for PSDPDF in PSDPDFs]
        return ASDPDFs       
    
    def MSCposterior_eval(self,rho2thaxis,idx1,idx2):
        """
        Returns the MSC posterior.

        Parameters
        ----------
        rho2thaxis : ``numpy.ndarray``
            rho2thaxis axis on which the posterior is evaluated.
        idx1,idx2 : int
            Indexes of the timeseries for MSC evaluation.
        """
        assert idx1!=idx2
        assert idx1<self.matp
        assert idx2<self.matp
        rho2exp = self.MSC[:,idx1,idx2]
        PDFs = []
        for rho2exptmp,navstmp in zip(rho2exp,self.navs):
            PDFs.append(CPSDstats_methods.rho2posterior(rho2thaxis,rho2exptmp,navstmp))
        return PDFs
    def R2posterior_eval(self,R2thaxis):
        """
        Returns the multiple coherence R2 posterior.

        Parameters
        ----------
        R2thaxis : ``numpy.ndarray``
            R2thaxis axis on which the posterior is evaluated.
        """
        PDFs = []
        for R2exptmp,navstmp in zip(self.R2,self.navs):
            PDFs.append(CPSDstats_methods.R2posterior(R2thaxis,R2exptmp,navstmp,self.matp))
        return PDFs

    def PSDquantiles_eval(self,cval): #also returns median
        """
        Returns the posterior PSD quantiles, at the given confidence level.

        Parameters
        ----------
        cval : float
            Confidence level.
        """
        self._checkCPSD()
        PSDquantile = []
        for idx in range(self.CPSD.shape[-1]):
            Sxx  = np.real(self.CPSD[:,idx,idx])
            PSD_CI = CPSDstats_methods.PSDpostCI(Sxx,self.navs,c=cval)
            PSDquantile.append(PSD_CI)
        PSDquantile = np.asarray(PSDquantile)
        return PSDquantile

    def MSCquantiles_eval(self,idx1,idx2,cval):
        """
        Returns the posterior MSC quantiles, at the given confidence level.

        Parameters
        ----------
        cval : float
            Confidence level.
        idx1,idx2 : int
            Indexes of the timeseries for MSC evaluation.
        """
        self._checkCPSD()
        MSC = self.MSC[:,idx1,idx2]
        MSCquantile = np.asarray([CPSDstats_methods.rho2postCI(r2e=r2e,navs=M,p=self.matp,c=cval) for r2e,M in zip(MSC,self.navs)])
        return MSCquantile

    def R2quantiles_eval(self,cval):
        """
        Returns the posterior R2 quantiles, at the given confidence level.

        Parameters
        ----------
        cval : float
            Confidence level.
        """
        self._checkCPSD()
        R2quantile = np.asarray([CPSDstats_methods.R2postCI(R2e,M,p=self.matp,c=cval) for R2e,M in zip(self.R2,self.navs)])
        return R2quantile
        

def _evalCPSD(datamat,Ls,kcoeffs,fs,win,olap,detrend_c,winscheme):
    CPSD = []
    navs = []
    for (tmpL,tmpk) in zip(Ls,kcoeffs):
        if winscheme is None or winscheme is False: wintmp = win
        else: wintmp = winscheme(tmpk,fs)
        tmpCPSD,tmpnavs,_ = _evalCPSD_1freq(datamat=datamat,tmpL=tmpL,tmpk=tmpk,fs=fs,win=wintmp,olap=olap,detrend_c=detrend_c)
            
        CPSD.append(tmpCPSD)
        navs.append(tmpnavs)
    return np.asarray(CPSD),np.asarray(navs)


def _evalCPSD_1freq(datamat,tmpL,tmpk,fs,win,olap,detrend_c):
    """
    CPSD calculator (only one frequency bin). 
    Split data into chunks of length tmpL, window it, evaluate FT at frequency tmpf, evaluate CPSD.
    Note: data can be multidimensional (p x N), to evaluate the CPSD.
    """
    ndim = datamat.shape[0]
    npoints = datamat.shape[1]

    # DFT coefficients
    winpt = win(tmpL) #spectral window
    p = -2j*np.pi*np.arange(0,tmpL,1)/tmpL
    C = np.exp(tmpk*p); #DFT coefficients

    startpoint = 0
    tmpnavs = 0
    Amat = np.zeros((ndim,ndim),dtype=complex)
    periodograms = []
    while(1):
        endpoint = startpoint + tmpL
        if(endpoint > npoints):
            break
        xs0 = datamat[:,startpoint:endpoint] #multivariate stretch
        if detrend_c: xs0 = xs0 - np.mean(xs0,axis=1,keepdims=True)
        xs = winpt * xs0 #windowed multivariate stretch
        ax = xs @ C #complex periodogram
        periodograms.append(ax)
        Amat += np.outer(ax,ax.conj()) #fill CPSD matrix
        tmpnavs += 1
        startpoint += int(tmpL*(1-olap))
    
    if tmpnavs==0: return Amat*0,0,[]
    wins2 = winpt@winpt
    tmpCPSD  = 2.0*Amat/tmpnavs/fs/wins2; #one-sided CPSD matrix
    periodograms  = np.asarray(periodograms)*np.sqrt(2.0/fs/wins2); #one-sided periodograms
    return tmpCPSD,tmpnavs,periodograms

def _getcohere(CPSD):
    cohere = [ np.diag(np.diag(Sij)**(-1/2)) @ Sij @ np.diag(np.diag(Sij)**(-1/2))
        for Sij in CPSD]
    return np.asarray(cohere)

def _getR2(CPSD,navs):
    R2 = []
    for Sij,nn in zip(CPSD,navs):
        if nn<Sij.shape[0]: R2.append(np.nan)
        else: R2.append(np.real(1-1/(np.linalg.inv(Sij)[0,0]*Sij[0,0])))
    return np.asarray(R2)

