import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp
import healpy as hp
import astropy.io.fits as fits
import camb
import fgbuster

def get_instr(freqs, depth_p):
    """
        Return the instrument dictionnary
    """
    instrument_dict = {}
    instrument_dict['frequency'] = np.array(freqs)
    instrument_dict['depth_p'] = np.array(depth_p, dtype=float)
    instrument_dict['depth_i'] = instrument_dict['depth_p']/np.sqrt(2)

    # instrument = fgbuster.standardize_instrument(instrument_dict)
    return instrument_dict

def generate_power_spectra_CAMB(Nside,  r=0, Alens=1, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, ns=0.965, As=2e-9, lens_potential_accuracy=1, nt=0, ntrun=0, type_power='total', typeless_bool=False):
    """
    Return [Cl^TT, Cl^EE, Cl^BB, Cl^TE]
    """
    lmax = 2*Nside
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

    print("Calculating spectra from CAMB !")
    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lmax=lmax)    
    if typeless_bool:
        return powers
    return powers[type_power]

def loading_params(directory_save_file, file_ver, MICMAC_sampler_obj):
    dict_all_params = dict()
    # Loading all files
    initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
    initial_freq_maps = np.load(initial_freq_maps_path)
    dict_all_params['initial_freq_maps'] = initial_freq_maps

    initial_cmb_maps_path = directory_save_file+file_ver+'_initial_cmb_data.npy'
    input_cmb_maps = np.load(initial_cmb_maps_path)
    dict_all_params['input_cmb_maps'] = input_cmb_maps

    if MICMAC_sampler_obj.save_eta_chain_maps:
        all_eta_maps_path = directory_save_file+file_ver+'_all_eta_maps.npy'
        all_eta_maps = np.load(all_eta_maps_path)
        dict_all_params['all_eta_maps'] = all_eta_maps

    if MICMAC_sampler_obj.save_CMB_chain_maps:
        all_s_c_WF_maps_path = directory_save_file+file_ver+'_all_s_c_WF_maps.npy'
        all_s_c_WF_maps = np.load(all_s_c_WF_maps_path)
        dict_all_params['all_s_c_WF_maps'] = all_s_c_WF_maps

        all_s_c_fluct_maps_path = directory_save_file+file_ver+'_all_s_c_fluct_maps.npy'
        all_s_c_fluct_maps = np.load(all_s_c_fluct_maps_path)
        dict_all_params['all_s_c_fluct_maps'] = all_s_c_fluct_maps

    if MICMAC_sampler_obj.sample_r_Metropolis:
        all_r_samples_path = directory_save_file+file_ver+'_all_r_samples.npy'
        all_r_samples = np.load(all_r_samples_path)
        dict_all_params['all_r_samples'] = all_r_samples
    elif MICMAC_sampler_obj.sample_C_inv_Wishart:
        all_cell_samples_path = directory_save_file+file_ver+'_all_cell_samples.npy'
        all_cell_samples = np.load(all_cell_samples_path)
        dict_all_params['all_cell_samples'] = all_cell_samples

    all_params_mixing_matrix_samples_path = directory_save_file+file_ver+'_all_params_mixing_matrix_samples.npy'
    all_params_mixing_matrix_samples = np.load(all_params_mixing_matrix_samples_path)
    dict_all_params['all_params_mixing_matrix_samples'] = all_params_mixing_matrix_samples

    dict_all_params['last_PRNGKey'] = np.load(directory_save_file+file_ver+'_last_PRNGkey.npy')

    return dict_all_params

