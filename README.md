# Minimally Informed CMB MAp Constructor: MICMAC
Repository to centralize work on pixel implementation for the non-parametric approach in pixel domain of component separation

Note : maybe change the name of the repo to something better

# Installation

To create a virtual environment environment from scratch, you can use `conda` and the following command:

`conda env create -f micmac_env.yml`


Otherwise, you can go in your virtual environment and from the root directory of this package, run:

`python -m pip install .`

Then, you may call this package using `import micmac`

# Required dependencies
* emcee
* cmbdb (https://github.com/dpole/cmbdb/tree/master)
* fgbuster (only fgbuster.observation_helpers to get the input frequency maps in the tutorials, the src code is completely independent from fgbuster)
* chex

# Notes

* DO NOT USE THIS PACKAGE ON LOGIN NODES ON HPC !!! (do not perform any computation on login-node in general, but this package in particular might use all the login nodes resources available)