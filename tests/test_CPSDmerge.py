import numpy as np
from scipy import stats
 
from noisefinder.cpsd import CPSDevaluate,CPSDmerge
from noisefinder.freqscheme_presets import lpfScheme
from noisefinder.specwindows import BH92

 
def test_cpsd_merge():

	data = stats.norm.rvs(size=50000)
	Lmax = 10005
	fs=10

	lpf_freqscheme = lpfScheme(Lmax=Lmax,fs=fs,fmax=1e-1,optimalolap=False)
	lpf_res1 = CPSDevaluate(
		ts=[data[:20000]],
		freqscheme=lpf_freqscheme
	)
	lpf_res2 = CPSDevaluate(
		ts=[data[40000:]],
		freqscheme=lpf_freqscheme
	)

	lpf_res = CPSDmerge(lpf_res1,lpf_res2)

	np.testing.assert_allclose(
		lpf_res1.navs + lpf_res2.navs, lpf_res.navs,
		rtol=1e-8, atol=1e-8,
		err_msg="number of averaged windows does not correspond",
	)
	np.testing.assert_allclose(
		np.nan_to_num(lpf_res1.PSD[0]*lpf_res1.navs) + np.nan_to_num(lpf_res2.PSD[0]*lpf_res2.navs), lpf_res.PSD[0]*lpf_res.navs, 
		rtol=1e-8, atol=1e-8,
		err_msg="PSD sum does not correspond",
	)