import numpy as np
import scipy
 
from noisefinder.cpsd import CPSDevaluate,stats
from noisefinder.freqscheme_presets import lpfScheme
from noisefinder.specwindows import BH92,Nuttall4
from noisefinder import noiseproj

 
def test_cpsd_posteriors_testexec():
	"""
	Just test execution.
	"""

	dataA = scipy.stats.norm.rvs(size=1000)
	dataB = scipy.stats.norm.rvs(size=1000)

	LPFfreqscheme = lpfScheme(Lmax=200,fmax=1e-1,fs=10,optimalolap=True)

	data_CPSD = CPSDevaluate(
		ts=[dataA,dataB],
		freqscheme=LPFfreqscheme
	)

	mfnp = noiseproj.run_noiseproj(CPSDmat=data_CPSD.CPSD,navs=data_CPSD.navs,case="complex")

	noiseproj.stats.ASDresidual_qnt(mfnp,q=0.50)
	noiseproj.stats.PSDresidual_qnt(mfnp,q=0.50)
	noiseproj.stats.R2contrib_qnt(mfnp,q=0.50)
	noiseproj.stats.alpha_qnt(mfnp,q=0.50)