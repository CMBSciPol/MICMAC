# Minimally Informed CMB MAp Constructor: MICMAC
# [Release coming soon]

Pixel implementation of the non-parametric component separation.
Extension to component separation method of [Leloup et al. (2023)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.108.123547)

**Note**: Make sure **not to use** this package on supercomputer **LOGIN NODES** or it will use all the available resources.


## Installation

The easiest way to install and use `micmac` is the following

- clone the repo
    ```
    git clone https://github.com/CMBSciPol/MICMAC && cd MICMAC
    ```
- create a virtual environment with the required dependencies
    ```bash
    conda env create -f micmac_env.yml
    ```
- install `micmac`
    ```bash
    conda activate micmac_env
    python -m pip install .
    ```

**Note:** this package uses the [JAX library](ttps://jax.readthedocs.io), for which you should follow the official [installation guide](https://jax.readthedocs.io/en/latest/installation.html) in order to make sure the installation can go through.

## How to use `micmac`

You will find in the `tutorial` directory a list of notebooks showcasing how to use `micmac`.

The tutorials make use of an additional Python library not installed by default
- `cmbdb`
    ```shell
    python -m pip install git+https://github.com/dpole/cmbdb
    ```

## License
This code is released under the GPLv3 license, which can be found in the [LICENSE](./LICENSE) file.
