# Minimally Informed CMB MAp Constructor: MICMAC
Pixel implementation for the non-parametric component separation.
Extension to component separation method of Leloup et al. 2024 (https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.123547)

# Installation
Create a clean virtual environment with the reuired dependencies:
`conda env create -f micmac_env.yml`

Install the micmac package by running:
`cd micmac`
`python -m pip install .`

You can then use the package by importing it as:
`import micmac`

# Required dependencies
Install jax as:
```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
Optional (for tutorials):
* cmbdb (https://github.com/dpole/cmbdb/tree/master)
* fgbuster (only uses fgbuster.observation_helpers to get the input frequency maps in the tutorials, the src code is completely independent from fgbuster)

# Notes
* DO NOT USE THIS PACKAGE ON LOGIN NODES ON HPC !!! (do not perform any computation on login-node in general, but this package in particular might use all the login node resources available)
