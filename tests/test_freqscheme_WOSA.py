import numpy as np
import pytest
from scipy import signal
from scipy import stats
 
from noisefinder.freqscheme_presets import lpfScheme, wosaScheme
from noisefinder.cpsd import CPSDevaluate
from noisefinder.specwindows import BH92

 
 
@pytest.mark.parametrize("nperseg", [100000, 59847])
@pytest.mark.parametrize("fs", [4, 6.54])
def test_psd_wosa_matches_scipy_welch(nperseg, fs):

	dataA = stats.norm.rvs(size=200000)

	f_ref, pxx_ref = signal.welch(
		dataA,
		fs=fs,
		window=BH92(nperseg),
		nperseg=nperseg,
		detrend=False,
		return_onesided=True,
	)

	wosa_freqscheme = wosaScheme(nperseg=nperseg,fs=fs,win=BH92,optimalolap=False)
	wosa_result = CPSDevaluate(
		ts=[dataA],
		freqscheme=wosa_freqscheme
	)
	f_test = wosa_result.freqs
	pxx_test = wosa_result.PSD[0]

	assert f_test.shape == f_ref.shape, (
		f"frequency array shape mismatch: {f_test.shape} vs {f_ref.shape}"
	)
	np.testing.assert_allclose(
		f_test, f_ref, rtol=1e-8, atol=1e-8,
		err_msg="frequency bins differ from scipy.signal.welch",
	)
	assert pxx_test.shape == pxx_ref.shape, (
		f"PSD array shape mismatch: {pxx_test.shape} vs {pxx_ref.shape}"
	)
	np.testing.assert_allclose(
		pxx_test, pxx_ref, rtol=1e-8, atol=1e-10,
		err_msg=(
			"PSD values differ from scipy.signal.welch beyond tolerance "
			f"(nperseg={nperseg})"
		),
	)

@pytest.mark.parametrize("nperseg", [100000, 59847])
@pytest.mark.parametrize("fs", [4, 6.54])
def test_csd_wosa_matches_scipy_welch(nperseg, fs):
	dataA = stats.norm.rvs(size=200000)
	dataB = stats.norm.rvs(size=200000)

	f_ref, cxx_ref = signal.csd(
		dataA,dataB,
		fs=fs,
		window=BH92(nperseg),
		nperseg=nperseg,
		detrend=False,
		return_onesided=True,
	)

	wosa_freqscheme = wosaScheme(nperseg=nperseg,fs=fs,win=BH92,optimalolap=False)
	wosa_result = CPSDevaluate(
		ts=[dataA,dataB],
		freqscheme=wosa_freqscheme
	)
	f_test = wosa_result.freqs
	cxx_test = wosa_result.CPSD[:,0,1]
 
	assert f_test.shape == f_ref.shape, (
		f"frequency array shape mismatch: {f_test.shape} vs {f_ref.shape}"
	)
	np.testing.assert_allclose(
		f_test, f_ref, rtol=1e-8, atol=1e-8,
		err_msg="frequency bins differ from scipy.signal.welch",
	)
	assert cxx_test.shape == cxx_ref.shape, (
		f"CSD array shape mismatch: {cxx_test.shape} vs {cxx_ref.shape}"
	)
	np.testing.assert_allclose(
		cxx_test, np.conj(cxx_ref), rtol=1e-8, atol=1e-10,
		err_msg=(
			"CSD values differ from scipy.signal.welch beyond tolerance "
			f"(nperseg={nperseg})"
		),
	)

def test_psd_wosa_optimalolap_test():

	dataA = stats.norm.rvs(size=50000)

	nperseg = 10000
	fs = 8.56

	ref = CPSDevaluate(
		ts=[dataA],
		freqscheme=wosaScheme(nperseg=nperseg,fs=fs,win=BH92,optimalolap=False)
	)
	pxx_ref = ref.PSD[0]

	result = CPSDevaluate(
		ts=[dataA],
		freqscheme=wosaScheme(nperseg=nperseg,fs=fs,win=BH92,optimalolap=True)
	)
	pxx_test = result.PSD[0]
 
	np.testing.assert_allclose(
		pxx_test, pxx_ref, rtol=1e-8, atol=1e-10,
		err_msg=(
			"optimalolap should have no effect if Ltot is multiple of nperseg"
		),
	)