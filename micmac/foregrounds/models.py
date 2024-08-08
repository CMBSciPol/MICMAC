# This file is part of MICMAC.
# Copyright (C) 2024 CNRS / SciPol developers
#
# MICMAC is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MICMAC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MICMAC. If not, see <https://www.gnu.org/licenses/>.

"""
Module to create fake fgs models starting from the PySM ones.
"""
import healpy as hp
import numpy as np
from pysm3 import Sky
from scipy import constants

__all__ = [
    'get_spectral_params_true_values',
    'parametric_sky_customized',
    'get_observation_customized',
    'fgs_freq_maps_from_customized_model_nonparam',
    'd1s1_sky_customized',
]

h_over_k = constants.h * 1e9 / constants.k


def get_spectral_params_true_values(nside, model=['s0', 'd0']):
    """
    For a given PySM model returns the true values
    of the spectral parameters.
    The first fgs must have attribute pl_index
    and the second mbb_index and mbb_temperature.

    Parameters
    ----------
    nside: int
        Healpix nside
    model: list of str
        PySM model to consider

    Returns
    -------
    beta_pl: float
        Synchrotron spectral index
    beta_mbb: float
        Dust spectral index
    temp_mbb: float
        Dust temperature
    """
    sky = Sky(nside=nside, preset_strings=model)
    # Synchrotron
    synch = sky.components[0]
    beta_pl = synch.pl_index.value
    # Dust
    dust = sky.components[1]
    beta_mbb = dust.mbb_index.value
    temp_mbb = dust.mbb_temperature.value

    return beta_pl, beta_mbb, temp_mbb


def parametric_sky_customized(fgs_models, nside_map, nside_spv):
    """
    Gives d1s1-like model with less spv of the spectral parameters
    (still parametric)
    Returns a PySM-like Sky object

    Parameters
    ----------
    fgs_models: list of str
        PySM model to consider
    nside_map: int
        Healpix nside of the final maps
    nside_spv: int
        Healpix nside of the spectral parameters

    Returns
    -------
    sky: pysm3.Sky
        PySM-like Sky object with the modified spectral parameters
    new_spectral_params: list of np.array
        List of the new spectral parameters
    """
    new_spectral_params = []
    ### Get the sky
    sky = Sky(nside=nside_map, preset_strings=fgs_models)
    for f, model in enumerate(fgs_models):
        ### Modify the spectral parameter values
        if model == 'd1':
            # beta dust
            beta_mbb = sky.components[f].mbb_index.value
            beta_mbb_dowgraded = hp.ud_grade(beta_mbb, nside_spv)
            # hp.mollview(beta_mbb_dowgraded, title='Downgraded beta dust')
            beta_mbb_new = hp.ud_grade(beta_mbb_dowgraded, nside_map)
            # hp.mollview(beta_mbb_new, title='New beta dust')
            new_spectral_params.append(beta_mbb_new)
            for i, item in enumerate(beta_mbb_new):
                sky.components[f].mbb_index.value[i] = item
            # temp dust
            temp_mbb = sky.components[f].mbb_temperature.value
            temp_mbb_dowgraded = hp.ud_grade(temp_mbb, nside_spv)
            # hp.mollview(temp_mbb_dowgraded, title='Downgraded temp dust')
            temp_mbb_new = hp.ud_grade(temp_mbb_dowgraded, nside_map)
            # hp.mollview(temp_mbb_new, title='New temp dust')
            new_spectral_params.append(temp_mbb_new)
            for i, item in enumerate(temp_mbb_new):
                sky.components[f].mbb_temperature.value[i] = item
        elif model == 's1':
            # beta synch
            beta_pl = sky.components[f].pl_index.value
            beta_pl_dowgraded = hp.ud_grade(beta_pl, nside_spv)
            # hp.mollview(beta_pl_dowgraded, title='Downgraded beta synch')
            beta_pl_new = hp.ud_grade(beta_pl_dowgraded, nside_map)
            # hp.mollview(beta_pl_new, title='New beta synch')
            new_spectral_params.append(beta_pl_new)
            for i, item in enumerate(beta_pl_new):
                sky.components[f].pl_index[i] = item
            # plt.show()
        else:
            raise ValueError('Model not recognized (only d1 and s1 supported as of now)')
    return sky, new_spectral_params


def get_observation_customized(instrument='', sky=None, noise=False, nside=None, unit='uK_CMB'):
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
    import pysm3.units as u
    from fgbuster.observation_helpers import (
        get_instrument,
        get_noise_realization,
        standardize_instrument,
    )

    if isinstance(instrument, str):
        instrument = get_instrument(instrument)
    else:
        instrument = standardize_instrument(instrument)
    if nside is None:
        nside = sky.nside
    elif not isinstance(sky, str):
        try:
            assert nside == sky.nside, (
                'Mismatch between the value of the nside of the pysm3.Sky '
                'argument and the one passed in the nside argument.'
            )
        except AttributeError:
            raise ValueError('Either provide a pysm3.Sky as sky argument ' ' or specify the nside argument.')

    if noise:
        res = get_noise_realization(nside, instrument, unit)
    else:
        res = np.zeros((len(instrument.frequency), 3, hp.nside2npix(nside)))

    for res_freq, freq in zip(res, instrument.frequency):
        emission = sky.get_emission(freq * u.GHz).to(getattr(u, unit), equivalencies=u.cmb_equivalencies(freq * u.GHz))
        res_freq += emission.value

    return res


def fgs_freq_maps_from_customized_model_nonparam(
    nside_map, nside_spv, instrument, fgs_models, idx_ref_freq, return_mixing_mat=False
):
    """
    Gives non-parametric model built from input (parametric) fgs_model,
    with a different level of spv of the scaling laws given by nside_spv.
    The scaling laws (A) come from the freq maps of the input fgs_model
    downgraded to nside_spv.
    Returns the final freq maps elements built as d=As.

    Parameters
    ----------
    nside_map: int
        nside at which the final maps are built
    nside_spv: int
        nside at which the spv in the scaling laws is downgraded
    instrument: dictionary
        dictionary containing the instrument configuration,
        must have key "frequency" containing a list of the instr freqs
    fgs_models: list[str]
        list of strings refferring to the PySM fgs models to start from
    idx_ref_freq: int
        index of the reference frequency to use for normalization in A
    return_mixing_mat: bool
        if True, return also the mixing matrix

    Returns
    -------
    freq_maps_final: array
        final frequency maps
    mixing_mat: array
        mixing matrix, returned only if return_mixing_mat is True
    """
    from fgbuster.observation_helpers import get_observation
    from fgbuster.separation_recipes import _my_ud_grade

    n_fgs_comp = len(fgs_models)
    mixing_mat = np.zeros((len(instrument.frequency), n_fgs_comp, hp.nside2npix(nside_map)))
    ref_maps_QU = []
    # Loop over fgs components (typically synch and dust)
    for i, model_i in enumerate(fgs_models):
        # Take initial freq maps
        if nside_spv == 0:
            freq_maps_fgs_i = get_observation(instrument, model_i, nside=1, noise=False)[:, 1, :]
            freq_maps_fgs_i = np.average(freq_maps_fgs_i, axis=1)
        else:
            freq_maps_fgs_i = get_observation(instrument, model_i, nside=nside_spv, noise=False)[
                :, 1, :
            ]  # keep only Q (assumed same mixing mat elements for Q and U)
        # Build new mixing matrix elements
        mixing_mat[:, i, :] = np.array(
            [
                _my_ud_grade(freq_maps_fgs_i[f] / freq_maps_fgs_i[idx_ref_freq], nside_map)
                for f in range(len(instrument.frequency))
            ]
        )
        # Take reference freq map
        ref_maps_QU.append(get_observation(instrument, model_i, nside=nside_map, noise=False)[idx_ref_freq, 1:, :])
    # Build final frequency maps
    freq_maps_final = np.einsum('fcp,csp->fsp', mixing_mat, np.array(ref_maps_QU))

    if return_mixing_mat:
        return freq_maps_final, mixing_mat

    return freq_maps_final


#### Deprecated (but still used in old version v3_customized of multinode notebook):
# TODO: verify that not needed anymore in the multinode notebook and delete
def d1s1_sky_customized(nside_map, nside_spv):
    """
    Gives d1s1-like model with less spv of the spectral parameters.

    Parameters
    ----------
    nside_map: int
        Healpix nside of the final maps
    nside_spv: int
        Healpix nside of the spectral parameters

    Returns
    -------
    sky: pysm3.Sky
        PySM-like Sky object with the modified spectral parameters
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
