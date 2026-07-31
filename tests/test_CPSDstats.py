import numpy as np
import scipy
 
from noisefinder.cpsd import CPSDevaluate,stats
from noisefinder.freqscheme_presets import lpfScheme
from noisefinder.specwindows import BH92,Nuttall4

 
def test_cpsd_posteriors_testexec():
	"""
	Just test execution.
	"""

	dataA = scipy.stats.norm.rvs(size=1000)
	dataB = scipy.stats.norm.rvs(size=1000)

	LPFfreqscheme = lpfScheme(Lmax=800,fmax=1e-1,fs=10,optimalolap=True)

	data_CPSD = CPSDevaluate(
		ts=[dataA,dataB],
		freqscheme=LPFfreqscheme
	)

	stats.PSDposterior_dist(data_CPSD,tsidx=0)

	stats.ASDposterior_pdf(data_CPSD,tsidx=0,ASDaxis=np.linspace(0,10,10))
	stats.PSDposterior_pdf(data_CPSD,tsidx=0,PSDaxis=np.linspace(0,10,10))

	stats.ASDposterior_qnt(data_CPSD,tsidx=0,q=0.50)
	stats.PSDposterior_qnt(data_CPSD,tsidx=0,q=0.50)

	stats.MSCposterior_pdf(data_CPSD,idx1=0,idx2=1,MSCthaxis=np.linspace(0.01,0.99,10))
	stats.MSCposterior_qnt(data_CPSD,idx1=0,idx2=1,q=0.50)
	
	stats.R2posterior_pdf(data_CPSD,R2thaxis=np.linspace(0.01,0.99,10))
	stats.R2posterior_qnt(data_CPSD,q=0.50)