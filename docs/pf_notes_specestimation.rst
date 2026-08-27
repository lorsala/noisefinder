Additional notes (on spectral estimation)
=========================================

Here we collect a few notes on the estimation of spectral quantities, conventions, notations, etc.


CPSD calculation convention
---------------------------

Several conventions exist for defining the cross-spectral density, differing
mainly in **which signal is conjugated** and the **sign of the exponential**
in the Fourier transform. This document adopts the following conventions:

.. math::
   X(f) = \int_{-\infty}^{+\infty} x(t)\,e^{-2 \pi i f t}\,dt.

.. math::
   S_{xy}(f) = \lim_{T \to \infty} \frac{1}{T}\, \mathbb{E}\left[ X(f)\, Y^{*}(f) \right]

where :math:`X(f)` and :math:`Y(f)` are the Fourier transforms of
:math:`x(t)` and :math:`y(t)`, and :math:`Y^{*}(f)` denotes the complex
conjugate.

With this choice, the **second** index (:math:`x`) corresponds to the
conjugated signal. Note that some sources swap this convention (conjugating
:math:`X` instead of :math:`Y`), which flips the sign of the phase of
:math:`S_{xy}(f)`. Our notation is the same as `MATLAB cpsd <https://it.mathworks.com/help/signal/ref/cpsd.html>`_, but the opposite of `scipy.signal.csd <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html>`_.

This means that, if :math:`Y` has a **phase delay**, the coherence in our convention has a **positive phase**. 

Spectral Coherence and MSC
--------------------------

We define the **coherence** between two signals as the frequency-domain measure of their
linear relationship, normalized to lie between 0 and 1:

.. math::

   \rho_{xy}(f) = \frac{S_{xy}(f)}{S_{xx}^{1/2}(f)\, S_{yy}^{1/2}(f)}

It is calculated from the CPSD matrix by normalizing it w.r.t. the diagonal elements.
Find it at :attr:`noisefinder.cpsd.cpsdresults.CPSDresults.cohere`. 

We often use the **magnitude-squared coherence** (MSC), defined as :math:`|\rho_{xy}(f)|^2`, 
:attr:`noisefinder.cpsd.cpsdresults.CPSDresults.MSC`. The reason for this is that we have 
an analytical expression for the posterior of the latter, which is useful when a few
periodograms are available.

.. note::
    A value of :math:`\text{MSC}(f) = 1` at a given frequency indicates a
    perfect linear relationship between the two signals at that frequency, while
    :math:`\text{MSC}(f) = 0` indicates no linear relationship. In practice,
    the estimated MSC is always less than 1 even for perfectly coherent signals,
    due to finite averaging and noise — so its value must
    be interpreted relative to the number of averaged segments used in the
    estimate. Module :mod:`noisefinder.cpsd.stats` provides the posterior distribution.


Multiple coherence R2
---------------------

If many timeseries are available, they may cross-correlate with the main timeseries,
and cross-correlate with one another, possibly with phase shifts and different coupling coefficients.
The *multiple coherence* :math:`R^2(f)` is a generalization of the MSC, 
which accounts for the whole set of timeseries, and their possible contribution to the main one.
If correlation can be assumed to imply causation, :math:`R^2=1-S_\text{contrib}/S_\text{tot}`.

.. warning::
    The multiple coherence can only be calculated if the number of additional timeseries 
    (not counting the main one) is **smaller** than the number of available periodograms, i.e.,
    if the rank of the CPSD matrix is high enough for it to be inverted.

.. _opzimize_overlap:

Optimal overlap
---------------

`noisefinder` allows the user to optimize the overlap between adjacent
segments in order to maximize data usage. This is achieved by reducing the
overlap from its nominal value
:attr:`~noisefinder.freqscheme.FreqScheme.olapmax`, effectively
making the *overlap* parameter act as a *maximum overlap* rather than a fixed
value (hence the name :attr:`~noisefinder.freqscheme.FreqScheme.olapmax`). 
Crucially, the number of averaged segments remains unchanged
throughout this adjustment -- only the spacing between segments is optimized
to make better use of the available data. See the image below for a visual
explanation.

   .. figure:: _images/optimaloverlap.png
       :width: 700px
       :align: center
