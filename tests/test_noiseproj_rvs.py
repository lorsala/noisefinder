import numpy as np
import pytest

from noisefinder import noiseproj
from noisefinder.noiseproj import noiseproj_onebin, stats_onebin


def _random_cpsd(p, navs, seed=0, case="complex"):
	"""Sample CPSD matrix (Wishart-like) from `navs` random periodograms."""
	rng = np.random.default_rng(seed)
	X = rng.normal(size=(navs, p))
	if case == "complex":
		X = X + 1j * rng.normal(size=(navs, p))
	X[:, 0] += 0.5 * X[:, 1:].sum(axis=1)  # correlate channel 0 with the others
	return X.conj().T @ X / navs


def _sfnp(p, case, navs=20, seed=0):
	return noiseproj_onebin._run_noiseproj_onebin(
		CPSDmat=_random_cpsd(p, navs, seed=seed, case=case), navs=navs, case=case
	)


def test_alpha_onebin_RVS_shape_and_dtype():
	for case in ("real", "complex"):
		for p in (2, 3):
			sfnp = _sfnp(p, case)
			for size in (1, 7):
				rvs = stats_onebin.alpha_onebin_RVS(sfnp, size)
				assert rvs.shape == (size, p - 1)
				assert np.iscomplexobj(rvs) == (case == "complex")


def test_alpha_onebin_RVS_reproducible():
	sfnp = _sfnp(3, "complex")
	a = stats_onebin.alpha_onebin_RVS(sfnp, 5, random_state=42)
	b = stats_onebin.alpha_onebin_RVS(sfnp, 5, random_state=42)
	c = stats_onebin.alpha_onebin_RVS(sfnp, 5, random_state=43)
	assert np.array_equal(a, b)
	assert not np.array_equal(a, c)


def test_alpha_onebin_RVS_moments():
	"""Sample mean/covariance must match the multivariate Student-t."""
	for case in ("real", "complex"):
		sfnp = _sfnp(3, case, navs=30)
		rvs = stats_onebin.alpha_onebin_RVS(sfnp, 200_000, random_state=1)
		if case == "complex":
			rvs = np.hstack([rvs.real, rvs.imag])
		dist = sfnp.alphasmultivar_dist
		cov = dist.shape * sfnp.nu / (sfnp.nu - 2)
		std = np.sqrt(np.diag(cov))
		assert np.allclose(rvs.mean(axis=0), dist.loc, atol=5 * std / np.sqrt(len(rvs)) + 1e-12)
		assert np.allclose(np.cov(rvs.T), cov, rtol=0.05, atol=0.05 * std.max() ** 2)


def test_alpha_onebin_RVS_bad_input():
	sfnp = _sfnp(2, "real")
	with pytest.raises(ValueError):
		stats_onebin.alpha_onebin_RVS(None, 5)
	with pytest.raises(ValueError):
		stats_onebin.alpha_onebin_RVS(sfnp, 0)
	with pytest.raises(TypeError):
		stats_onebin.alpha_onebin_RVS(sfnp, 2.5)
	with pytest.raises(TypeError):
		stats_onebin.alpha_onebin_RVS(sfnp, True)


def test_alpha_RVS_multifreq():
	p, nf, size = 3, 4, 6
	navs = np.array([20, 0, 25, 30])
	for case in ("real", "complex"):
		CPSD = np.stack([_random_cpsd(p, max(n, p), seed=i, case=case) for i, n in enumerate(navs)])
		mfnp = noiseproj.run_noiseproj(CPSDmat=CPSD, navs=navs, case=case)

		rvs = noiseproj.stats.alpha_RVS(mfnp, size, random_state=3)
		assert rvs.shape == (nf, size, p - 1)
		assert np.iscomplexobj(rvs) == (case == "complex")
		assert np.all(np.isnan(rvs[1]))
		assert np.all(np.isfinite(rvs[[0, 2, 3]]))

		again = noiseproj.stats.alpha_RVS(mfnp, size, random_state=3)
		assert np.array_equal(rvs[[0, 2, 3]], again[[0, 2, 3]])


def test_alpha_RVS_no_valid_bin():
	mfnp = noiseproj.noiseproj.NoiseProjResults(sfnp_arr=[None, None])
	with pytest.raises(ValueError):
		noiseproj.stats.alpha_RVS(mfnp, 5)
