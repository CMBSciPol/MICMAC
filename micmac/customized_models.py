"""
Module to create fake fgs models starting from the PySM ones.
"""

import numpy as np
import healpy as hp
import sympy
import pysm3.units as u
import matplotlib.pyplot as plt
from pysm3 import Sky
from scipy import constants
from astropy.cosmology import Planck15
from collections.abc import Iterable

from fgbuster.observation_helpers import get_instrument, get_sky, get_observation, \
                                        standardize_instrument, get_noise_realization


h_over_k = constants.h * 1e9 / constants.k


def d1s1_sky_customized(nside_map, nside_spv):
    """
    Gives d1s1-like model with less spv of the spectral parameters.
    """
    ### Get the sky
    sky = Sky(nside=nside_map, preset_strings=['d1', 's1'])
    ### Modify the spectral parameter values
    # beta dust
    beta_mbb = sky.components[0].mbb_index.value
    beta_mbb_dowgraded = hp.ud_grade(beta_mbb, nside_spv)
    # hp.mollview(beta_mbb_dowgraded, title='Downgraded beta dust')
    beta_mbb_new = hp.ud_grade(beta_mbb_dowgraded, nside_map)
    # hp.mollview(beta_mbb_new, title='New beta dust')
    for i, item in enumerate(beta_mbb_new):
        sky.components[0].mbb_index.value[i] = item
    # temp dust
    temp_mbb = sky.components[0].mbb_temperature.value
    temp_mbb_dowgraded = hp.ud_grade(temp_mbb, nside_spv)
    # hp.mollview(temp_mbb_dowgraded, title='Downgraded temp dust')
    temp_mbb_new = hp.ud_grade(temp_mbb_dowgraded, nside_map)
    # hp.mollview(temp_mbb_new, title='New temp dust')
    for i, item in enumerate(temp_mbb_new):
        sky.components[0].mbb_temperature.value[i] = item
    # beta synch
    beta_pl = sky.components[1].pl_index.value
    beta_pl_dowgraded = hp.ud_grade(beta_pl, nside_spv)
    # hp.mollview(beta_pl_dowgraded, title='Downgraded beta synch')
    beta_pl_new = hp.ud_grade(beta_pl_dowgraded, nside_map)
    # hp.mollview(beta_pl_new, title='New beta synch')
    for i, item in enumerate(beta_pl_new):
        sky.components[1].pl_index[i] = item
    # plt.show()
    return sky



def get_observation_customized(instrument='', sky=None,
                    noise=False, nside=None, unit='uK_CMB'):
    """ 
    NOTE: This is a customized version of the FGBuster function
          it takes the PySm Sky directly instead of a string tag

    Get a pre-defined instrumental configuration

    Parameters
    ----------
    instrument:
        It can be either a `str` (see :func:`get_instrument`) or an
        object that provides the following as a key or an attribute.

        - **frequency** (required)
        - **depth_p** (required if ``noise=True``)
        - **depth_i** (required if ``noise=True``)

        They can be anything that is convertible to a float numpy array.
        If only one of ``depth_p`` or ``depth_i`` is provided, the other is
        inferred assuming that the former is sqrt(2) higher than the latter.
    sky: str of pysm3.Sky
        Sky to observe. It can be a `pysm3.Sky` or a tag to create one.
    noise: bool
        If true, add Gaussian, uncorrelated, isotropic noise.
    nside: int
        Desired output healpix nside. It is optional if `sky` is a `pysm3.Sky`,
        and required if it is a `str` or ``None``.
    unit: str
        Unit of the output. Only K_CMB and K_RJ (and multiples) are supported.

    Returns
    -------
    observation: array
        Shape is ``(n_freq, 3, n_pix)``
    """
    if isinstance(instrument, str):
        instrument = get_instrument(instrument)
    else:
        instrument = standardize_instrument(instrument)
    if nside is None:
        nside = sky.nside
    elif not isinstance(sky, str):
        try:
            assert nside == sky.nside, (
                "Mismatch between the value of the nside of the pysm3.Sky "
                "argument and the one passed in the nside argument.")
        except AttributeError:
            raise ValueError("Either provide a pysm3.Sky as sky argument "
                             " or specify the nside argument.")

    if noise:
        res = get_noise_realization(nside, instrument, unit)
    else:
        res = np.zeros((len(instrument.frequency), 3, hp.nside2npix(nside)))

    for res_freq, freq in zip(res, instrument.frequency):
        emission = sky.get_emission(freq * u.GHz).to(
            getattr(u, unit),
            equivalencies=u.cmb_equivalencies(freq * u.GHz))
        res_freq += emission.value

    return res



# # Testing
# nside_map = 64
# nside_spv = 1
# instr = 'LiteBIRD'

# my_sky = d1s1_sky_customized(nside_map, nside_spv)
# my_freq_maps = get_observation_customized(instr, my_sky, noise=False, nside=nside_map)

# freq_maps = get_observation(instr, 'd1s1', noise=False, nside=nside_map)

# hp.mollview( freq_maps[0, 1, :], title='True d1s1 freq map')
# hp.mollview(my_freq_maps[0, 1, :], title='My freq map')
# hp.mollview(my_freq_maps[0, 1, :]-freq_maps[0, 1, :], title='My freq map - d1s1 freq maps')
# plt.show()

