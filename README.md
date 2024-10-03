# Minimally Informed CMB MAp Constructor: MICMAC

<img src="./MICMAC-2.png" alt="drawing" width="200">

MICMAC Logo, credits: Ema Tsang King Sang

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

**Note:** this package uses the [JAX library](https://jax.readthedocs.io), for which you should follow the official [installation guide](https://jax.readthedocs.io/en/latest/installation.html) in order to make sure the installation can go through.

## How to use `micmac`

You will find in the `tutorials` directory a list of notebooks showcasing how to use `micmac`.

The tutorials make use of an additional Python library not installed by default
- `cmbdb`
    ```shell
    python -m pip install git+https://github.com/dpole/cmbdb
    ```
Note: If you fork the repository and want to commit some changes, you may want to use pre-commit. When committing with pre-commit, your changes will probably be reformatted, you must then re-add them and re-commit.

## License
This code is released under the GPLv3 license, which can be found in the [LICENSE](./LICENSE) file.

## Contact

For any solicitation, please contact `morshed at apc.in2p3.fr` or `rizzieri at apc.in2p3.fr`


## Citation

If you use `micmac`, please consider citing:
```
@misc{morshed2024pixel,
      title={Pixel domain implementation of the Minimally Informed CMB MAp foreground Cleaning (MICMAC) method},
      author={Magdy Morshed and Arianna Rizzieri and Cl\'ement Leloup and Josquin Errard and Radek Stompor},
      year={2024},
      eprint={2405.18365},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO}
}
@article{Leloup:2023vkb,
    author = "Leloup, Cl\'ement and Errard, Josquin and Stompor, Radek",
    title = "{Nonparametric maximum likelihood component separation for CMB polarization data}",
    eprint = "2308.03412",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevD.108.123547",
    journal = "Phys. Rev. D",
    volume = "108",
    number = "12",
    pages = "123547",
    year = "2023"
}
```
