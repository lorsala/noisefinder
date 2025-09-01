import numpy as np
import scipy.stats as st
import scipy.special as sp
from .CPSDstats import CPSDstats
from .noiseproj_1f import noiseproj_1f
import warnings
    
class noiseproj:
    """
    Single-frequency noise projection (decorrelation) tool.
    Evaluates noise projection parameters at individual frequencies.
    
    Parameters
    -----------

    CPSDin:
        The nfreq x p x p CPSD matrix, or an :class:`noisefinder.CPSDstats` object (or any inheriting class).
    navs: float 
        Number of averaged periodograms, requirement: ``navs>=CPSDdim``. Must be left void if CPSDin is an CPSDstats object.
    freqs : float 
        Frequency array, optional. Must be left void if CPSDin is an CPSDstats object.
    case : string 
        Prior on susceptibilities: "complex" or "real". Defaults to "complex".
    
    Yields
    -----------

    Instances: 
        of noiseproj_1f for all frequencies
    Sres_CI: 
        residual noise PSD, confidence interval
    alpre_CI: 
        real part of susceptibilities, confidence interval
    alpim_CI: 
        imag part of susceptibilities, confidence interval
    """        

    def __init__(self,CPSDin,navs=None,freqs=None,
                 case="complex",verbose=False):        
        if isinstance(CPSDin,CPSDstats):
            if navs is not None: warnings.warn("In noiseproj.__init__, input CPSD is an object CPSDstats, but you gave navs as input, are you sure? I'm overriding navs.", RuntimeWarning)
            if freqs is not None: warnings.warn("In noiseproj.__init__, input CPSD is an object CPSDstats, but you gave freqs as input, are you sure? I'm overriding freqs.", RuntimeWarning)
            CPSDin._checkCPSD()
            CPSD = CPSDin.CPSD
            navs = CPSDin.navs
            freqs = CPSDin.freqs
        else: CPSD = np.copy(CPSDin)
        self._init_main_(CPSD=CPSD,navs=navs,freqs=freqs,case=case,verbose=verbose)


    def _init_main_(self,CPSD,navs,freqs,case="complex",verbose=False):
        assert isinstance(CPSD,np.ndarray)
        assert CPSD.ndim==3, "If you only have one frequency bin, use noiseproj_1f"
        assert CPSD.shape[1]==CPSD.shape[2]
        
        self.r = CPSD.shape[-1]-1
        self.navs = navs
        self.case = case
        self.CPSD = CPSD
        if hasattr(self, 'freqs'):
            self.freqs = freqs
            assert self.freqs.shape==self.navs.shape, "freqs and navs must have the same shape."
        
        self.sfnp_1f = []
        for ffi in range(len(navs)):
            sfnp_1f_tmp = noiseproj_1f(self.CPSD[ffi,:,:],self.navs[ffi],case=self.case)
            self.sfnp_1f.append(sfnp_1f_tmp)
        
        if(verbose):
            print(
                """**** noiseproj init verbose ****
This object contains the noise projection posteriors, with CPSDs you provided.
You can access the confidence intervals with get_noiseprojpost(cval),
for other information please look at the documentation.
            """)
            
    def _init_fromCPSDstats_(cls,other:CPSDstats,navs=None,freqs=None):
        """Alternative constructor with CPSDstats object as input"""
        return cls(other.CPSD, other.navs, other.freqs ,case="complex")
    
    def get_noiseprojCI(self,cval): 
        """
        Performs noise projection and returns confidence intervals.

        Parameters
        ----------
        cval:float
            Confidence level.

        Returns
        ----------
        Sres_CI: Confidence interval for the residual PSD.
        alpre_CI: Confidence interval for the susceptibilities, real part.
        alpim_CI: Confidence interval for the susceptibilities, imaginary part.
        """       
        Sres_CI = []
        alpre_CI = []
        alpim_CI = []
        for ffi in range(len(self.navs)):
            sfnp_1f_tmp = noiseproj_1f(self.CPSD[ffi,:,:],self.navs[ffi],case=self.case)
            Sres_CI_tmp,alpre_CI_tmp,alpim_CI_tmp = sfnp_1f_tmp.sfnp_quantiles(c=cval)
            Sres_CI.append(Sres_CI_tmp)
            alpre_CI.append(alpre_CI_tmp)
            alpim_CI.append(alpim_CI_tmp)
        Sres_CI = np.asarray(Sres_CI)
        alpre_CI = np.asarray(alpre_CI)
        alpim_CI = np.asarray(alpim_CI)
        return Sres_CI,alpre_CI,alpim_CI

    def PSDrespost_eval(self,PSDaxis=None):
        """
        Performs noise projection and returns the residual PSD posterior.

        Parameters
        ----------
        PSDaxis: 
            Axis on which to evaluate the PSD posterior. If ``None``, auto evaluates it.

        Returns
        ----------
        PSDaxis
        list of PSD posteriors
        """       
        if PSDaxis is not None:
            pass
        else: #generate PSDaxis
            c=0.99
            Sres_CI = np.asarray([[sfnp_1f.Sres_PDF.ppf((1-c)/2),0,sfnp_1f.Sres_PDF.ppf((1+c)/2)] for sfnp_1f in self.sfnp_1f])
            PSDlim = np.min(Sres_CI[:,0]), np.max(Sres_CI[:,2])
            PSDaxis = np.geomspace(PSDlim[0]/100,PSDlim[1]*100,1000)
        return PSDaxis, [sfnp_1f.PSDrespost_eval(PSDaxis=PSDaxis)[1] for sfnp_1f in self.sfnp_1f]
    def ASDrespost_eval(self,ASDaxis=None):
        """
        Performs noise projection and returns the residual ASD posterior.

        Parameters
        ----------
        ASDaxis: 
            Axis on which to evaluate the ASD posterior. If ``None``, auto evaluates it.

        Returns
        ----------
        ASDaxis
        list of ASD posteriors
        """  
        PSDaxis = None if ASDaxis is None else ASDaxis**2
        PSDaxis,PSDPDFs = self.PSDrespost_eval(PSDaxis=PSDaxis)
        ASDaxis=np.sqrt(PSDaxis)
        ASDPDFs = [PSDPDF * 2 * PSDaxis for PSDPDF in PSDPDFs]
        return ASDaxis,ASDPDFs
        
        