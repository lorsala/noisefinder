import numpy as np
import scipy.stats as st
import scipy.special as sp


class noiseproj_1f:
    """
    Single-frequency noise projection (decorrelation) tool.
    Calculates frozen pdfs of residual noise and susceptibilities, for a single frequency. For multiple frequencies, use :class:`noisefinder.noiseproj`.
        
    Parameters
    -----------

    CPSDmat
        The p x p CPSD matrix.
    navs: float 
        Number of averaged periodograms, requirement: ``navs>=CPSDdim``.
    case : string 
        Prior on susceptibilities: "complex" or "real". Defaults to "complex".
    
    Yields
    -----------

    Instances: 
        of :class:`noisefinder.noiseproj_1f` for all frequencies
    Sres_PDF: 
        frozen :class:`scipy.stats` pdf of noise residual
    alpre_PDF: 
        frozen :class:`scipy.stats` pdfs of susceptibilities (real part)
    alpim_PDF: 
        frozen :class:`scipy.stats` pdfs of susceptibilities (imag part)
    """   

    def __init__(self,CPSDmat:np.ndarray,navs:np.ndarray,case="complex"):
        self.r = CPSDmat.shape[0]-1
        self.navs = navs
        self.case = case
        
        assert self.r>0, "This is a 1D distribution, can't decorrelate."
        assert self.navs>self.r, "You need M>r for matrix not to be degenerate."
        assert case=="real" or case=="complex", "Case needs to be either real or complex."

        CPSDmat = CPSDmat if case=="complex" else np.real(CPSDmat)
        self.CPSDmat = CPSDmat
        self._executedecorr()
        
    def _executedecorr(self):
        Schur = np.real(1/np.linalg.inv(self.CPSDmat)[0,0])
        A = self.CPSDmat*self.navs
        A1y = A[0,1:]
        Ayy = A[1:,1:]
        
        if self.case=="complex":
            self.nu = 2*self.navs-2*self.r
            self.Sres_PDF = st.invgamma(a=self.navs-self.r,scale=Schur*self.navs)
            mucomplex = A1y@np.linalg.inv(Ayy)
            muvector  = np.block([np.real(mucomplex),np.imag(mucomplex)])
            covmatrix = np.linalg.inv(np.block([[np.real(Ayy),np.imag(Ayy)],[-np.imag(Ayy),np.real(Ayy)]]))*Schur*self.navs/self.nu
            alphas_PDF = [st.t(loc=mu,scale=std,df=self.nu)
                          for mu,std in zip(muvector,np.sqrt(np.diag(covmatrix)))]
            self.alpre_PDF = [alphas_PDF[i] for i in range(self.r)]
            self.alpim_PDF = [alphas_PDF[i+self.r] for i in range(self.r)]
            self.alphasmultivar_PDF = st.multivariate_t(loc=muvector,shape=covmatrix,df=self.nu,allow_singular=True)
            
        elif self.case=="real":
            self.nu = 2*self.navs-self.r
            self.Sres_PDF = st.invgamma(a=self.navs-self.r/2,scale=Schur*self.navs)
            muvector = A1y@np.linalg.inv(Ayy)
            covmatrix = np.linalg.inv(Ayy)*Schur*self.navs/self.nu
            self.alpre_PDF = [st.t(loc=mu,scale=std,df=self.nu)
                              for mu,std in zip(muvector,np.sqrt(np.diag(covmatrix)))]
            self.alpim_PDF = None
            self.alphasmultivar_PDF = st.multivariate_t(loc=muvector,shape=covmatrix,df=self.nu,allow_singular=True)
            
    
    def genDecorrRVS(self,size): #extract from multivariate t distribution
        """
        Returns samples from Sres and alpha distributions. 
        """
        Sres_rvs = self.Sres_PDF.rvs(size=size)
        realrvsStud  = self.alphasmultivar_PDF.rvs(size=size)
        alphas_rvs = realrvsStud[:,:self.r]+1.j*realrvsStud[:,self.r:] if self.case=="complex" else realrvsStud
        return Sres_rvs,alphas_rvs
        
    def genDecorrPDFs(self):
        """
        Returns frozen :class:`scipy.stats` objects, PDFs of Sres and alpha distributions. 
        """
        return self.Sres_PDF,self.alpre_PDF,self.alpim_PDF
        

    def sfnp_quantiles(self,c):
        """
        Returns quantiles after noise projection (residual and susceptibilities), at the given confidence level.

        Parameters
        ----------
        cval:float
            Confidence level.

        Returns
        ------------

        Sres_CI: residual noise PSD, confidence interval
        alpre_CI: real part of susceptibilities, confidence interval
        alpim_CI: imag part of susceptibilities, confidence interval
        R2_CI: R2, confidence interval
        """
        
        Sres_CI = [self.Sres_PDF.ppf((1-c)/2),self.Sres_PDF.ppf(0.5),self.Sres_PDF.ppf((1+c)/2)]
        
        alpre_CI = np.asarray([[self.alpre_PDF[i].ppf((1-c)/2),self.alpre_PDF[i].ppf(0.50),self.alpre_PDF[i].ppf((1+c)/2)] for i in range(self.r)])

        if self.case=="complex":
            alpim_CI = np.asarray([[self.alpim_PDF[i].ppf((1-c)/2),self.alpim_PDF[i].ppf(0.50),self.alpim_PDF[i].ppf((1+c)/2)] for i in range(self.r)])
        elif self.case=="real":
            alpim_CI = [None,None,None]
        return Sres_CI,alpre_CI,alpim_CI

    def PSDrespost_eval(self,PSDaxis=None):
        """
        Performs noise projection and returns the residual PSD posterior.
        """      
        Sres_CI,_,_ = self.sfnp_quantiles(c=0.99)
        if PSDaxis is None: PSDaxis=np.geomspace(Sres_CI[0]/10,Sres_CI[2]*10,1000)
        PDF = self.Sres_PDF.pdf(x=PSDaxis)
        return PSDaxis,PDF
    def ASDrespost_eval(self,ASDaxis=None):
        """
        Performs noise projection and returns the residual ASD posterior.
        """   
        PSDaxis = None if ASDaxis is None else ASDaxis**2
        PSDaxis,PSDPDF = self.PSDrespost_eval(PSDaxis=PSDaxis)
        ASDaxis=np.sqrt(PSDaxis)
        ASDPDF = PSDPDF * 2 * PSDaxis
        return ASDaxis,ASDPDF


    
