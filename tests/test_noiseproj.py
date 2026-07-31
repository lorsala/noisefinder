import numpy as np
from scipy import stats
 
from noisefinder.cpsd import CPSDevaluate
from noisefinder.freqscheme_presets import lpfScheme
from noisefinder.noiseproj import run_noiseproj
from noisefinder.specwindows import BH92

 
def test_noiseproj():
	datares = stats.norm.rvs(size=50000)
	dataB = stats.norm.rvs(size=50000)
	dataA = datares + 0.1*dataB

	Lmax = 10005
	fs=10

	lpf_freqscheme = lpfScheme(Lmax=Lmax,fs=fs,fmax=1e-1,optimalolap=False)
	lpf_res = CPSDevaluate(
		ts=[dataA,dataB],
		freqscheme=lpf_freqscheme
	)

	mfnp_c = run_noiseproj(CPSDmat=lpf_res.CPSD,navs=lpf_res.navs,case="complex")
	mfnp_r = run_noiseproj(CPSDmat=lpf_res.CPSD,navs=lpf_res.navs,case="real")
