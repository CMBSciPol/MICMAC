``Likelihood`` -- Likelihood sampling
==============================

The `MICMAC` package relies on the sampling of likelihood, either in pixel domain as provided by the :mod:`micmac.likelihood.pixel` module, or in spherical harmonic domain as provided by the :mod:`micmac.likelihood.harmonic` module.

The :mod:`micmac.likelihood.sampling` module provides the necessary tools to sample either likelihood, while the :mod:`micmac.likelihood.initobjects` module displays the necessary objects to initialize the likelihoods from `toml` files.


Module contents
---------------
.. toctree::
   :maxdepth: 2

   likelihood/initobjects.rst
   likelihood/pixel.rst
   likelihood/harmonic.rst
   likelihood/sampling.rst
