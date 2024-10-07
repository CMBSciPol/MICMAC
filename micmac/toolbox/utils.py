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

import numpy as np

__all__ = ['get_instr', 'generate_power_spectra_CAMB', 'loading_params']


def get_instr(freqs, depth_p):
    """
    Return the instrument dictionary

    Parameters
    ----------
    freqs: array[float] or float
        frequency of the instrument
    depth_p: array[float] or float, of the same dimension as freqs
        depth of the instrument

    Returns
    -------
    instrument: dict
        instrument dictionary with keys 'frequency', 'depth_p', 'depth_i'
    """
    assert len(freqs) == len(depth_p)

    instrument_dict = {}
    instrument_dict['frequency'] = np.array(freqs)
    instrument_dict['depth_p'] = np.array(depth_p, dtype=float)
    instrument_dict['depth_i'] = instrument_dict['depth_p'] / np.sqrt(2)

    return instrument_dict


def generate_power_spectra_CAMB(
    Nside,
    r=0,
    Alens=1,
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.06,
    omk=0,
    tau=0.06,
    ns=0.965,
    As=2e-9,
    lens_potential_accuracy=1,
    nt=0,
    ntrun=0,
    type_power='total',
    typeless_bool=False,
):
    """
    Generate power spectra from CAMB
    Return [Cl^TT, Cl^EE, Cl^BB, Cl^TE]

    Parameters
    ----------
    Nside: int
        Nside of the maps
    r: float
        tensor to scalar ratio
    Alens: float
        lensing amplitude
    H0: float
        Hubble constant
    ombh2: float
        baryon density
    omch2: float
        cold dark matter density
    mnu: float
        sum of neutrino masses
    omk: float
        curvature density
    tau: float
        optical depth
    ns: float
        scalar spectral index
    As: float
        amplitude of the primordial power spectrum
    lens_potential_accuracy: int
        lensing potential accuracy
    nt: float
        tensor spectral index
    ntrun: float
        tensor running index
    type_power: str
        type of power spectra to return
    typeless_bool: bool
        return the full power spectra if True, otherwise only the power spectrum of type type_power

    Returns
    -------
    powers: dictionary or array[float]
        dictionary of power spectra if typeless_bool is True, otherwise power spectra of type type_power
    """
    try:
        import camb
    except ImportError:
        raise ImportError('camb is not installed. Please install it with "pip install camb"')

    lmax = 2 * Nside
    # pars = camb.CAMBparams(max_l_tensor=lmax, parameterization='tensor_param_indeptilt')
    pars = camb.CAMBparams(max_l_tensor=lmax)
    pars.WantTensors = True

    pars.Accuracy.AccurateBB = True
    pars.Accuracy.AccuratePolarization = True
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, Alens=Alens)
    pars.InitPower.set_params(As=As, ns=ns, r=r, parameterization='tensor_param_indeptilt', nt=nt, ntrun=ntrun)
    pars.max_eta_k_tensor = lmax + 100  # 15000  # 100

    # pars.set_cosmology(H0=H0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)

    print('Calculating spectra from CAMB !')
    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lmax=lmax)
    if typeless_bool:
        return powers
    return powers[type_power]


def loading_params(directory_save_file, file_ver, MICMAC_sampler_obj):
    """
    Load all parameters from the saved files

    Parameters
    ----------
    directory_save_file: str
        directory where the files are saved
    file_ver: str
        run version of the file
    MICMAC_sampler_obj: object
        MICMAC sampler object, to check which parameters were saved

    Returns
    -------
    dict_all_params: dict
        dictionary of all parameters loaded from the files
    """
    dict_all_params = dict()
    # Loading all files
    initial_freq_maps_path = directory_save_file + file_ver + '_initial_data.npy'
    initial_freq_maps = np.load(initial_freq_maps_path)
    dict_all_params['initial_freq_maps'] = initial_freq_maps

    initial_cmb_maps_path = directory_save_file + file_ver + '_initial_cmb_data.npy'
    input_cmb_maps = np.load(initial_cmb_maps_path)
    dict_all_params['input_cmb_maps'] = input_cmb_maps

    initial_noise_map_path = directory_save_file + file_ver + '_initial_noise_data.npy'
    initial_noise_map = np.load(initial_noise_map_path)
    dict_all_params['input_noise_map'] = initial_noise_map

    if MICMAC_sampler_obj.save_eta_chain_maps:
        all_eta_maps_path = directory_save_file + file_ver + '_all_eta_maps.npy'
        all_eta_maps = np.load(all_eta_maps_path)
        dict_all_params['all_eta_maps'] = all_eta_maps

    if MICMAC_sampler_obj.save_CMB_chain_maps:
        all_s_c_WF_maps_path = directory_save_file + file_ver + '_all_s_c_WF_maps.npy'
        all_s_c_WF_maps = np.load(all_s_c_WF_maps_path)
        dict_all_params['all_s_c_WF_maps'] = all_s_c_WF_maps

        all_s_c_fluct_maps_path = directory_save_file + file_ver + '_all_s_c_fluct_maps.npy'
        all_s_c_fluct_maps = np.load(all_s_c_fluct_maps_path)
        dict_all_params['all_s_c_fluct_maps'] = all_s_c_fluct_maps

    if MICMAC_sampler_obj.save_s_c_spectra:
        all_s_c_spectra_path = directory_save_file + file_ver + '_all_s_c_spectra.npy'
        all_s_c_spectra = np.load(all_s_c_spectra_path)
        dict_all_params['all_samples_s_c_spectra'] = all_s_c_spectra

    if MICMAC_sampler_obj.sample_r_Metropolis:
        all_r_samples_path = directory_save_file + file_ver + '_all_r_samples.npy'
        all_r_samples = np.load(all_r_samples_path)
        dict_all_params['all_r_samples'] = all_r_samples
    elif MICMAC_sampler_obj.sample_C_inv_Wishart:
        all_cell_samples_path = directory_save_file + file_ver + '_all_cell_samples.npy'
        all_cell_samples = np.load(all_cell_samples_path)
        dict_all_params['all_cell_samples'] = all_cell_samples

    all_params_mixing_matrix_samples_path = directory_save_file + file_ver + '_all_params_mixing_matrix_samples.npy'
    all_params_mixing_matrix_samples = np.load(all_params_mixing_matrix_samples_path)
    dict_all_params['all_params_mixing_matrix_samples'] = all_params_mixing_matrix_samples

    dict_all_params['last_PRNGKey'] = np.load(directory_save_file + file_ver + '_last_PRNGkey.npy')

    return dict_all_params
