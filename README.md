# About noisefinder

[![Doc badge](https://img.shields.io/badge/Docs-available-brightgreen)](docs/_build/html/index.html)
[![DOI](https://img.shields.io/badge/Reference-10.48550/arXiv.2507.20846-blue)](https://doi.org/10.48550/arXiv.2507.20846)
[![License badge](https://img.shields.io/badge/License-BSD-orange)](.)

`noisefinder` is a python package for precise spectral estimation, and noise decorrelation. It derives from our work, which you can [find on arXiv](https://doi.org/10.48550/arXiv.2507.20846).\
An initial version of this code has been used for the data analysis of the LISA Pathfinder mission [^1] [^2].

`noisefinder` measures the spectral properties of time series. The precise Bayesian estimate is particularly relevant whenever the number of available averaging windows (*periodograms*) is very low, for instance at very-low frequencies. Moreover, it performs noise projection (timeseries decorrelation), allowing for precise retrieval of background noise in the presence of disturbing sources.

In particular, `noisefinder` allows to:
* Calculate the **cross-PSD** of the given (synchronous) timeseries, their *coherence*, and *multiple coherence*.
    * For each frequency, calculate the confidence interval of the **PSD estimate** and the **coherences**. This result is inferred from the Bayesian posterior distributions, depending on the number of available averaging windows.
    * Choose a frequency scheme as needed. For instance, the user can select the LISA Pathfinder scheme.
* Apply noise projection estimate to decorrelate all timeseries from the first one:
    * Retrieve the **residual noise**, and its confidence level.
    * Retrieve the **coupling coefficients** of the time series to the first one (*susceptibilities*, complex-valued), and their confidence level.
    * Retrieve the total **contribution to noise** (*multiple coherence*), and its confidence level.


[^1]: The LISA Pathfinder Collaboration, *In-depth analysis of LISA Pathfinder performance results: Time evolution, noise projection, physical models, and implications for LISA.* [(Phys. Rev. D 110, 042004, 2024)](https://doi.org/10.1103/PhysRevD.110.042004)
[^2]: The LISA Pathfinder Collaboration, *Beyond the Required LISA Free-Fall Performance: New LISA Pathfinder Results down to 20 μ⁢Hz.* [(Phys. Rev. Lett. 120, 061101, 2018)](https://doi.org/10.1103/PhysRevLett.120.061101)


## Citation

If you use this package in your research or project, please cite the [following paper]((https://doi.org/10.48550/arXiv.2507.20846)).\
(this link will be substituted after peer review)

```
@misc{precisionspectral,
      title={Precision spectral estimation at sub-Hz frequencies: closed-form posteriors and Bayesian noise projection}, 
      author={Lorenzo Sala and Stefano Vitale},
      year={2025},
      eprint={2507.20846},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2507.20846}, 
}
```


