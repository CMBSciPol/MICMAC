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

import os

import numpy as np

__all__ = ['get_instr', 'generate_power_spectra_CAMB', 'generate_CMB', 'loading_params']


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


def generate_CMB(nside, lmax, nstokes=2):
    """
    Returns CMB spectra of scalar modes only and tensor modes only (with r=1)
    Both CMB spectra are either returned in the usual form [number_correlations,lmax+1]
    """

    # Selecting the relevant auto- and cross-correlations from CAMB spectra
    if nstokes == 2:
        # EE, BB
        partial_indices_polar = np.array([1, 2])
    elif nstokes == 1:
        # TT
        partial_indices_polar = np.array([0])
    else:
        # TT, EE, BB, EB
        partial_indices_polar = np.arange(4)

    n_correlations = int(np.ceil(nstokes**2 / 2) + np.floor(nstokes / 2))

    # Generating the CMB power spectra
    all_spectra_r0 = generate_power_spectra_CAMB(nside * 2, r=0, typeless_bool=True)
    all_spectra_r1 = generate_power_spectra_CAMB(nside * 2, r=1, typeless_bool=True)

    # Retrieve the scalar mode spectrum
    camb_cls_r0 = all_spectra_r0['total'][: lmax + 1, partial_indices_polar]

    # Retrieve the tensor mode spectrum
    tensor_spectra_r1 = all_spectra_r1['tensor'][: lmax + 1, partial_indices_polar]

    theoretical_r1_tensor = np.zeros((n_correlations, lmax + 1))
    theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

    theoretical_r1_tensor[:nstokes, ...] = tensor_spectra_r1.T
    theoretical_r0_total[:nstokes, ...] = camb_cls_r0.T

    # Return spectra in the form [number_correlations,lmax+1]
    return theoretical_r0_total, theoretical_r1_tensor


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

    dict_existing_params = MICMAC_sampler_obj.__dict__

    # Loading all files
    if os.path.exists(directory_save_file + file_ver + '_initial_data.npy'):
        initial_freq_maps_path = directory_save_file + file_ver + '_initial_data.npy'
        initial_freq_maps = np.load(initial_freq_maps_path)
        dict_all_params['initial_freq_maps'] = initial_freq_maps

        initial_cmb_maps_path = directory_save_file + file_ver + '_initial_cmb_data.npy'
        input_cmb_maps = np.load(initial_cmb_maps_path)
        dict_all_params['input_cmb_maps'] = input_cmb_maps

        initial_noise_map_path = directory_save_file + file_ver + '_initial_noise_data.npy'
        initial_noise_map = np.load(initial_noise_map_path)
        dict_all_params['input_noise_map'] = initial_noise_map
    elif os.path.exists(directory_save_file + file_ver + '_input_maps.npz'):
        input_maps_path = directory_save_file + file_ver + '_input_maps.npz'
        input_maps = np.load(input_maps_path)
        dict_all_params['input_freq_maps'] = input_maps['input_freq_maps']
        dict_all_params['input_cmb_maps'] = input_maps['input_cmb_maps']
        dict_all_params['input_noise_map'] = input_maps['input_noise_map']
        if 'input_fgs_map' in input_maps:
            dict_all_params['input_fgs_map'] = input_maps['input_fgs_map']

    if 'save_eta_chain_maps' in dict_existing_params and MICMAC_sampler_obj.save_eta_chain_maps:
        all_eta_maps_path = directory_save_file + file_ver + '_all_eta_maps.npy'
        all_eta_maps = np.load(all_eta_maps_path)
        dict_all_params['all_eta_maps'] = all_eta_maps

    if 'save_CMB_chain_maps' in dict_existing_params and MICMAC_sampler_obj.save_CMB_chain_maps:
        all_s_c_WF_maps_path = directory_save_file + file_ver + '_all_s_c_WF_maps.npy'
        all_s_c_WF_maps = np.load(all_s_c_WF_maps_path)
        dict_all_params['all_s_c_WF_maps'] = all_s_c_WF_maps

        all_s_c_fluct_maps_path = directory_save_file + file_ver + '_all_s_c_fluct_maps.npy'
        all_s_c_fluct_maps = np.load(all_s_c_fluct_maps_path)
        dict_all_params['all_s_c_fluct_maps'] = all_s_c_fluct_maps

    if 'save_s_c_spectra' in dict_existing_params and MICMAC_sampler_obj.save_s_c_spectra:
        all_s_c_spectra_path = directory_save_file + file_ver + '_all_s_c_spectra.npy'
        all_s_c_spectra = np.load(all_s_c_spectra_path)
        dict_all_params['all_samples_s_c_spectra'] = all_s_c_spectra

    if 'sample_r_Metropolis' in dict_existing_params and MICMAC_sampler_obj.sample_r_Metropolis:
        all_r_samples_path = directory_save_file + file_ver + '_all_r_samples.npy'
        all_r_samples = np.load(all_r_samples_path)
        dict_all_params['all_r_samples'] = all_r_samples
    elif 'sample_C_inv_Wishart' in dict_existing_params and MICMAC_sampler_obj.sample_C_inv_Wishart:
        all_cell_samples_path = directory_save_file + file_ver + '_all_cell_samples.npy'
        all_cell_samples = np.load(all_cell_samples_path)
        dict_all_params['all_cell_samples'] = all_cell_samples
    else:
        if os.path.exists(directory_save_file + file_ver + '_all_r_samples.npy'):
            all_r_samples_path = directory_save_file + file_ver + '_all_r_samples.npy'
            all_r_samples = np.load(all_r_samples_path)
            dict_all_params['all_r_samples'] = all_r_samples
        else:
            print('No r samples found', flush=True)

    all_params_mixing_matrix_samples_path = directory_save_file + file_ver + '_all_params_mixing_matrix_samples.npy'
    all_params_mixing_matrix_samples = np.load(all_params_mixing_matrix_samples_path)
    dict_all_params['all_params_mixing_matrix_samples'] = all_params_mixing_matrix_samples

    dict_all_params['last_PRNGKey'] = np.load(directory_save_file + file_ver + '_last_PRNGkey.npy')

    if os.path.exists(directory_save_file + file_ver + '_c_approx.npy'):
        dict_all_params['c_approx'] = np.load(directory_save_file + file_ver + '_c_approx.npy')
    else:
        print('No c_approx found', flush=True)

    if os.path.exists(directory_save_file + file_ver + 'input_freq_alms.npy'):
        dict_all_params['input_freq_alms'] = np.load(directory_save_file + file_ver + 'input_freq_alms.npy')
    else:
        print('No input_freq_alms found', flush=True)

    return dict_all_params
