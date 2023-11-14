# Minimally Informed CMB MAp Constructor: MICMAC
(Temporary ?) Repository to centralize work on pixel implementation for the non-parametric approach in pixel domain of component separation

Note : maybe change the name of the repo to something better

# Installation

In your virtual environment and from the root directory of this package, just type :
`python -m pip install .`

Then, you may call this package using `import micmac`

# Required dependencies
* emcee
* cmbdb (https://github.com/dpole/cmbdb/tree/master)
* fgbuster (only fgbuster.observation_helpers to get the input frequency maps in the tutorials, the src code is completely independent from fgbuster)
* chex