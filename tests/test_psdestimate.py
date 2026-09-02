import numpy as np
import pytest
from scipy import signal
from scipy import stats
 
from noisefinder.cpsd import CPSDevaluate
from noisefinder.freqscheme import FreqScheme
from noisefinder.specwindows import BH92

 
 
@pytest.mark.parametrize("L", [100000, 59847])
@pytest.mark.parametrize("dft_idx", [0, 8, 59])
@pytest.mark.parametrize("fs", [4, 6.54])
def test_cpsd_custom_matches_scipy_wosa(L, dft_idx, fs):

	rng = np.random.default_rng(86957)
	dataA = stats.norm.rvs(size=200000)
	dataB = stats.norm.rvs(size=200000)

	def custom_freqscheme(fs,L,dft_idx): # more user-defined params
	    Ls = np.array([L]) #user-defined
	    dft_idxs = np.array([dft_idx]) #user-defined, with external method
	    return FreqScheme(
	        fs=fs,
	        olapmax=0.50,
	        dft_idxs=dft_idxs,
	        Ls=Ls,
	        win=BH92,
	        optimalolap=False,
	        name="one_freq"
	    )
	customscheme = custom_freqscheme(fs=fs,L=L,dft_idx=dft_idx)
	cpsd_customscheme = CPSDevaluate(
		ts=[dataA,dataB],
		freqscheme=customscheme
	)

	f_ref, CSD_AB_ref_arr = signal.csd(
		dataA,dataB,
		fs=fs,
		window=BH92(L),
		nperseg=L,
		detrend=False,
		return_onesided=True,
	)
	f_ref_AA, PSD_AA_ref_arr = signal.welch(
		dataA,
		fs=fs,
		window=BH92(L),
		nperseg=L,
		detrend=False,
		return_onesided=True,
	)
	f_ref_BB, PSD_BB_ref_arr = signal.welch(
		dataB,
		fs=fs,
		window=BH92(L),
		nperseg=L,
		detrend=False,
		return_onesided=True,
	)


	f_test = cpsd_customscheme.freqs[0]
	PSD_AA_test = cpsd_customscheme.CPSD[0,0,0]
	PSD_BB_test = cpsd_customscheme.CPSD[0,1,1]
	CSD_AB_test = cpsd_customscheme.CPSD[0,0,1]
	CSD_BA_test = cpsd_customscheme.CPSD[0,1,0]

	PSD_AA_ref = PSD_AA_ref_arr[dft_idx]
	PSD_BB_ref = PSD_BB_ref_arr[dft_idx]
	CSD_AB_ref = np.conj(CSD_AB_ref_arr[dft_idx])

	np.testing.assert_allclose(
		f_ref, f_ref_AA, rtol=1e-8, atol=1e-8,
		err_msg="frequencies do not correspond",
	)
	np.testing.assert_allclose(
		f_ref, f_ref_BB, rtol=1e-8, atol=1e-8,
		err_msg="frequencies do not correspond",
	)

	np.testing.assert_allclose(
		PSD_AA_test, PSD_AA_ref, rtol=1e-8, atol=1e-10,
		err_msg=(
			f"PSD mismatch: {PSD_AA_test} vs {PSD_AA_ref}"
		),
	)
	np.testing.assert_allclose(
		PSD_BB_test, PSD_BB_ref, rtol=1e-8, atol=1e-10,
		err_msg=(
			f"PSD mismatch: {PSD_BB_test} vs {PSD_BB_ref}"
		),
	)
	np.testing.assert_allclose(
		CSD_AB_test, CSD_AB_ref, rtol=1e-8, atol=1e-10,
		err_msg=(
			f"PSD mismatch: {CSD_AB_test} vs {CSD_AB_ref}"
		),
	)
	np.testing.assert_allclose(
		CSD_AB_test, np.conj(CSD_BA_test), rtol=1e-8, atol=1e-10,
		err_msg=(
			f"CSD mismatch: {CSD_AB_test} vs {np.conj(CSD_BA_test)}"
		),
	)