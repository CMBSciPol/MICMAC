``Foregrounds`` -- Foreground module
---------------

The foreground related routines are in the :mod:`micmac.foregrounds` module. This includes:
- :mod:`micmac.foregrounds.initmixingmatrix` for routines related to the construction of the mixing matrix if the initial foreground parameters are known
- :mod:`micmac.foregrounds.mixingmatrix` for routines related to the mixing matrix handling itself and for the rest of the package
- :mod:`micmac.foregrounds.models` to provide customized sky observations
- :mod:`micmac.foregrounds.templates` to provide multipatch templates for the foregrounds


.. autosummary::
   :toctree: _autosummary
   :maxdepth: 2

   micmac/foregrounds/*
