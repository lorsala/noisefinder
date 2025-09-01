import numpy as np
import scipy.special as sp
import scipy.stats as st
import scipy.integrate
import mpmath


def loghyp2f1_eff(a,b,c,z):
    res = np.log(sp.hyp2f1(a,b,c,z))
    if np.isinf(res) or np.isnan(res):
        res = mpmath.log(mpmath.hyp2f1(a,b,c,z,maxterms=1e6))
    return res

def PSDposterior(expPSD,navs):
    return st.invgamma(a=navs,scale=expPSD*navs)

def PSDpostCI(expPSD,navs,c):
    CI = [navs * expPSD / sp.gammainccinv(navs,(1-c)/2),
          navs * expPSD / sp.gammainccinv(navs,0.50),
          navs * expPSD / sp.gammainccinv(navs,(1+c)/2)]
    return np.asarray(CI)

def rho2posterior(rho2th,rho2exp,navs,p=None):
    return R2posterior(R2th=rho2th,R2exp=rho2exp,navs=navs,p=2)

def R2posterior(R2th,R2exp,navs,p):
    #note that hyp2f1 diverges for R2-->1 and navs-->inf, a future release should treat this better.
    logpdf = []
    for R2thtmp in R2th:
        logpdf.append((navs)*np.log(1-R2thtmp) + loghyp2f1_eff(navs,navs,p-1,R2exp*R2thtmp))
    maxlogpdf=max(logpdf)-np.log(1e50) #just to avoid overflow
    logpdf = [float(lp-maxlogpdf) for lp in logpdf]
    PDF = np.exp(logpdf)
    PDF = PDF/scipy.integrate.trapezoid(x=R2th,y=PDF)
    return PDF

def rho2postCI(r2e,navs,c,p=None):
    if navs>1000 and r2e>0.5: print("Warning, navs={:d}, evaluation can take a long time for MSC and R2, especially if there's high correlation.".format(navs))
    CI = _pdfCI(pdffunc=rho2posterior,valexp=r2e,navs=navs,p=None,c=c)
    return np.asarray(CI)

def R2postCI(R2e,navs,p,c):
    if navs>1000 and R2e>0.5: print("Warning, navs={:d}, evaluation can take a long time for MSC and R2, especially if there's high correlation.".format(navs))
    CI = _pdfCI(pdffunc=R2posterior,valexp=R2e,navs=navs,p=p,c=c)
    return np.asarray(CI)

def _pdfCI(pdffunc,valexp,navs,p,c):
    x = np.linspace(1e-6,1-1e-6,500)
    PDF = pdffunc(x,valexp,navs,p)
    CDF = scipy.integrate.cumulative_trapezoid(x=x,y=PDF)
    CDF = CDF/CDF[-1]
    CI = [x[np.argmax(CDF>(1-c)/2)],x[np.argmax(CDF>0.5)],x[np.argmax(CDF>(1+c)/2)]]
    return CI