import numpy as np
import pytest
from scipy import signal
from scipy import stats
 
from noisefinder.cpsd import CPSDevaluate
from noisefinder.freqscheme_presets import lpfScheme, wosaScheme
from noisefinder.specwindows import BH92

 
 
@pytest.mark.parametrize("Lmax", [100000, 59847])
@pytest.mark.parametrize("fs", [4, 6.54])
@pytest.mark.parametrize("optimalolap", [True, False])
def test_psd_lpf_matches_wosa_firstfreq(Lmax, fs, optimalolap):

	dataA = stats.norm.rvs(size=200000)

	wosa_freqscheme = wosaScheme(nperseg=Lmax,fs=fs,win=BH92,optimalolap=optimalolap)
	wosa_ref = CPSDevaluate(
		ts=[dataA],
		freqscheme=wosa_freqscheme
	)

	lpf_freqscheme = lpfScheme(Lmax=Lmax,fs=fs,fmax=1e-2,optimalolap=optimalolap)
	lpf_res = CPSDevaluate(
		ts=[dataA],
		freqscheme=lpf_freqscheme
	)

	lpf_firstfreq = lpf_res.freqs[0]
	lpf_firstfreq_psd = lpf_res.PSD[0][0]

	wosa_checkfreq = wosa_ref.freqs[4]
	wosa_checkfreq_psd = wosa_ref.PSD[0][4]

	np.testing.assert_allclose(
		wosa_checkfreq, lpf_firstfreq, rtol=1e-8, atol=1e-8,
		err_msg="frequencies do not correspond",
	)
	np.testing.assert_allclose(
		wosa_checkfreq_psd, lpf_firstfreq_psd, rtol=1e-8, atol=1e-10,
		err_msg=(
			f"PSD mismatch: {wosa_checkfreq_psd} vs {lpf_firstfreq_psd}"
		),
	)