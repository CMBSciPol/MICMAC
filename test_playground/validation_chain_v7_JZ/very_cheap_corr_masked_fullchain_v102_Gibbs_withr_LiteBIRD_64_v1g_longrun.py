import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import healpy as hp
import astropy.io.fits as fits
import camb
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
import numpyro
from functools import partial
import micmac as micmac
# from mcmc_tools import *

from jax import config
config.update("jax_enable_x64", True)

former_file_ver = 'corr_masked_full_v102_Gchain_SO_64_v1g' # -> corr LiteBIRD + r=1e-2 + 2000 iterations + corr_v1d_LiteBIRD + w/o restrict_to_mask + unmasked ; C_approx only lensing

file_ver = 'corr_masked_full_v102_Gchain_SO_64_v1f' # -> corr LiteBIRD + r=0 + 2000 iterations + corr_v1e_LiteBIRD + w/o restrict_to_mask + unmasked ; C_approx only lensing
file_ver = 'corr_masked_full_v102_Gchain_SO_64_v1g' # -> corr LiteBIRD + r=1e-3 + 2000 iterations + corr_v1f_LiteBIRD + w/o restrict_to_mask + unmasked ; C_approx only lensing
file_ver = 'corr_masked_full_v102_Gchain_SO_64_v1gb' # -> corr LiteBIRD + r=1e-3 + 3200 iterations + corr_v1fc_LiteBIRD + w/o restrict_to_mask + unmasked ; C_approx only lensing
# -> TODO !!!
reduction_noise = 1
factor_Fisher = 1

perso_repo_path = "/gpfswork/rech/nih/ube74zo/MICMAC_save/validation_chain_v7_JZ/"
path_home_test_playground = '/gpfswork/rech/nih/ube74zo/MICMAC/MICMAC/test_playground/'
# current_path = os.path.dirname(os.path.abspath(''))
# current_path = perso_repo_path
current_path = path_home_test_playground + '/validation_chain_v7_JZ/'
sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')

from fgbuster.observation_helpers import *
from micmac import *

common_repo = "/gpfswork/rech/nih/commun/"
path_mask = common_repo + "masks/mask_SO_SAT_apodized.fits"

# perso_repo_path = "/gpfswork/rech/nih/ube74zo/MICMAC_save/validation_chain_v6_JZ/"
directory_save_file = perso_repo_path + 'save_directory/'

# working_directory_path = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v5/'
working_directory_path = current_path + '/'#'/validation_chain_v6_JZ/'
directory_toml_file = working_directory_path + 'toml_params/'


path_toml_file = directory_toml_file + 'biased_v1a.toml'
path_toml_file = directory_toml_file + 'biased_v1b.toml'
path_toml_file = directory_toml_file + 'biased_v1c.toml'
path_toml_file = directory_toml_file + 'corr_v1d_LiteBIRD.toml'
path_toml_file = directory_toml_file + 'corr_v1e_LiteBIRD.toml'
path_toml_file = directory_toml_file + 'corr_v1f_LiteBIRD.toml'
path_toml_file = directory_toml_file + 'corr_v1fb_LiteBIRD.toml'


MICMAC_obj = micmac.create_MICMAC_sampler_from_toml_file(path_toml_file)

# path_home_test_playground = '/gpfswork/rech/nih/ube74zo/MICMAC/MICMAC/test_playground/'
# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/Fisher_matrix_SO_SAT_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'
# path_Fisher = '../Fisher_matrix_SO_SAT_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'
# path_Fisher = path_home_test_playground + 'Fisher_matrix_SO_SAT_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'
# path_Fisher = path_home_test_playground + 'Fisher_matrix_{}_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'.format(MICMAC_obj.instrument_name)


# General parameters
cmb_model = 'c1'
fgs_model = 'd0s0'
model = cmb_model+fgs_model
noise = True
# noise = False
noise_seed = 42
instr_name = MICMAC_obj.instrument_name #'SO_SAT'

# path_home_test_playground = '/gpfswork/rech/nih/ube74zo/MICMAC/MICMAC/test_playground/'
# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/Fisher_matrix_SO_SAT_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'
# path_Fisher = '../Fisher_matrix_SO_SAT_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'
# path_Fisher = path_home_test_playground + 'Fisher_matrix_SO_SAT_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'
# path_Fisher = path_home_test_playground + f'Fisher_matrix_{MICMAC_obj.instrument_name}_EB_model_{fgs_model}_noise_True_seed_42_lmin2_lmax128.txt'
path_Fisher = path_home_test_playground + f'Fisher_matrix_{MICMAC_obj.instrument_name}_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt'

# get instrument from public database
instrument = get_instrument(instr_name)

instrument['depth_p'] /= reduction_noise
# get input freq maps
np.random.seed(noise_seed)
# freq_maps = get_observation(instrument, model, nside=NSIDE, noise=noise)[:, 1:, :]   # keep only Q and U
freq_maps_fgs = get_observation(instrument, fgs_model, nside=MICMAC_obj.nside, noise=noise)[:, 1:, :]   # keep only Q and U
print("Shape for input frequency maps :", freq_maps_fgs.shape)


# Mask initialization
apod_mask = hp.ud_grade(hp.read_map(path_mask),nside_out=MICMAC_obj.nside)
mask = np.copy(apod_mask)
mask[apod_mask>0] = 1

mask = np.ones_like(apod_mask)
MICMAC_obj.mask = mask

# delta_ell = 10
# nb_bin = (MICMAC_obj.lmax-MICMAC_obj.lmin+1)//delta_ell

# MICMAC_obj.bin_ell_distribution = MICMAC_obj.lmin + jnp.arange(nb_bin+1)*delta_ell

# MICMAC_obj.freq_inverse_noise = micmac.get_noise_covar(instrument['depth_p'], MICMAC_obj.nside)
freq_inverse_noise = micmac.get_noise_covar(instrument['depth_p'], MICMAC_obj.nside) #MICMAC_obj.freq_inverse_noise

freq_inverse_noise_masked = np.zeros((MICMAC_obj.number_frequencies,MICMAC_obj.number_frequencies,MICMAC_obj.npix))

nb_pixels_mask = int(mask.sum())
freq_inverse_noise_masked[:,:,mask!=0] = np.repeat(freq_inverse_noise.ravel(order='F'), nb_pixels_mask).reshape((MICMAC_obj.number_frequencies,MICMAC_obj.number_frequencies,nb_pixels_mask), order='C')

MICMAC_obj.freq_inverse_noise = freq_inverse_noise_masked

initial_guess_r = MICMAC_obj.r_true
initial_guess_r=10**(-2)
initial_guess_r=10**(-3)
# initial_guess_r=10**(-4)
# initial_guess_r=10**(-8)



# Generation step-size
Fisher_matrix = np.loadtxt(path_Fisher)
minimum_std_Fisher = scipy.linalg.sqrtm(np.linalg.inv(Fisher_matrix))
minimum_std_Fisher_diag = np.diag(minimum_std_Fisher)

col_dim_B_f = MICMAC_obj.number_frequencies-len(MICMAC_obj.pos_special_freqs)

len_pos_special_freqs = len(MICMAC_obj.pos_special_freqs)
step_size_B_f = np.zeros((col_dim_B_f,2))
step_size_B_f[:,0] = minimum_std_Fisher_diag[:MICMAC_obj.number_frequencies-len_pos_special_freqs]
step_size_B_f[:,1] = minimum_std_Fisher_diag[MICMAC_obj.number_frequencies-len_pos_special_freqs:2*(MICMAC_obj.number_frequencies-len_pos_special_freqs)]

# MICMAC_obj.covariance_step_size_B_f = jnp.diag(step_size_B_f.ravel(order='F')**2)
# MICMAC_obj.covariance_step_size_B_f = np.copy(step_size_B_f)[:-1,:-1]
MICMAC_obj.covariance_step_size_B_f = np.copy(np.linalg.inv(Fisher_matrix))[:-1,:-1]/factor_Fisher

MICMAC_obj.step_size_r = minimum_std_Fisher_diag[-1]

# Generation input maps
input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = MICMAC_obj.generate_input_freq_maps_from_fgs(freq_maps_fgs, return_only_freq_maps=False)

input_freq_maps_masked = input_freq_maps*MICMAC_obj.mask

# Re-Defining the data if needed
indices_polar = np.array([1,2,4])
partial_indices_polar = indices_polar[:MICMAC_obj.nstokes]


theoretical_r0_total = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total)#[partial_indices_polar,:]
theoretical_r1_tensor = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r1_tensor)#[partial_indices_polar,:]

# Params mixing matrix
init_mixing_matrix_obj = micmac.InitMixingMatrix(MICMAC_obj.frequency_array, MICMAC_obj.number_components, pos_special_freqs=MICMAC_obj.pos_special_freqs)
exact_params_mixing_matrix = init_mixing_matrix_obj.init_params()


c_ell_approx = np.zeros((3,MICMAC_obj.lmax+1))
c_ell_approx[0,MICMAC_obj.lmin:] = theoretical_r0_total[0,:]
c_ell_approx[1,MICMAC_obj.lmin:] = theoretical_r0_total[1,:]


# First guesses
initial_wiener_filter_term = np.zeros((MICMAC_obj.nstokes, MICMAC_obj.npix))
initial_fluctuation_maps = np.zeros((MICMAC_obj.nstokes, MICMAC_obj.npix))

len_pos_special_freqs = len(MICMAC_obj.pos_special_freqs)
dimension_free_param_B_f = jnp.size(MICMAC_obj.indexes_free_Bf)

first_guess = jnp.copy(jnp.ravel(exact_params_mixing_matrix,order='F'))

# first_guess = first_guess.at[MICMAC_obj.indexes_free_Bf].set(
#     first_guess[MICMAC_obj.indexes_free_Bf]*np.random.uniform(low=.9,high=1.1, size=(dimension_free_param_B_f)))
# first_guess = first_guess.at[MICMAC_obj.indexes_free_Bf].set(
#     first_guess[MICMAC_obj.indexes_free_Bf]*np.random.uniform(low=.99,high=1.01, size=(dimension_free_param_B_f)))
# init_params_mixing_matrix = first_guess.reshape((MICMAC_obj.number_frequencies-len_pos_special_freqs),2,order='F')
print("First guess from 5 $\sigma$ Fisher !", flush=True)
first_guess = first_guess.at[MICMAC_obj.indexes_free_Bf].set(
    first_guess[MICMAC_obj.indexes_free_Bf] + minimum_std_Fisher_diag[:-1]*np.random.uniform(low=-5,high=5, size=(dimension_free_param_B_f)))
init_params_mixing_matrix = first_guess.reshape((MICMAC_obj.number_frequencies-len_pos_special_freqs),2,order='F')

CMB_c_ell = np.zeros_like(c_ell_approx)
# CMB_c_ell[:,MICMAC_obj.lmin:] = (theoretical_r0_total + MICMAC_obj.r_true*theoretical_r1_tensor)
CMB_c_ell[:,MICMAC_obj.lmin:] = (theoretical_r0_total + initial_guess_r*theoretical_r1_tensor)

if former_file_ver != '':
    print("### Continuing from previous run !", former_file_ver, flush=True)
    dict_all_params = loading_params(directory_save_file, former_file_ver, MICMAC_obj)

    init_params_mixing_matrix = dict_all_params['all_params_mixing_matrix_samples'][-1,:,:]

    if not(MICMAC_obj.cheap_save):
        initial_wiener_filter_term = dict_all_params['all_s_c_WF_maps'][-1,:,:]
        initial_fluctuation_maps = dict_all_params['all_s_c_fluct_maps'][-1,:,:]
    
    if MICMAC_obj.sample_r_Metropolis:
        initial_guess_r = dict_all_params['all_r_samples'][-1]
    elif MICMAC_obj.sample_C_inv_Wishart:
        CMB_c_ell = dict_all_params['all_cell_samples'][-1,:,:]
    

    input_cmb_maps = dict_all_params['input_cmb_maps']
    input_freq_maps_masked = dict_all_params['initial_freq_maps']*MICMAC_obj.mask

    # MICMAC_obj.number_iterations_done = MICMAC_obj.number_iterations_sampling

    MICMAC_obj.seed = MICMAC_obj.seed + MICMAC_obj.number_iterations_sampling



print(f'Exact param matrix : {exact_params_mixing_matrix}')
print(f'Initial param matrix : {init_params_mixing_matrix}')


time_start_sampling = time.time()
MICMAC_obj.perform_Gibbs_sampling(input_freq_maps_masked, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=initial_guess_r, initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
                         theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)

time_full_chain = (time.time()-time_start_sampling)/60
print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

if not(MICMAC_obj.cheap_save):
    all_eta  = MICMAC_obj.all_samples_eta
    all_s_c_WF_maps = MICMAC_obj.all_samples_wiener_filter_maps
    all_s_c_fluct_maps = MICMAC_obj.all_samples_fluctuation_maps
elif not(MICMAC_obj.very_cheap_save):
    all_s_c = MICMAC_obj.all_samples_s_c

if MICMAC_obj.sample_C_inv_Wishart:
    all_cell_samples = MICMAC_obj.all_samples_CMB_c_ell
if MICMAC_obj.sample_r_Metropolis:
    all_r_samples = MICMAC_obj.all_samples_r

all_params_mixing_matrix_samples = MICMAC_obj.all_params_mixing_matrix_samples

if former_file_ver != '':
    if not(MICMAC_obj.cheap_save):
        all_eta = np.hstack([dict_all_params['all_eta_maps'], all_eta[1:]])

        all_s_c_WF_maps = np.hstack([dict_all_params['all_s_c_WF_maps'], all_s_c_WF_maps[1:]])
        
        all_s_c_fluct_maps = np.hstack([dict_all_params['all_s_c_fluct_maps'], all_s_c_fluct_maps[1:]])


    elif not(MICMAC_obj.very_cheap_save):
        all_s_c = np.hstack([dict_all_params['all_s_c_samples'], all_s_c[1:]])
    if MICMAC_obj.sample_r_Metropolis:
        all_r_samples = np.hstack([dict_all_params['all_r_samples'], all_r_samples[1:]])
    elif MICMAC_obj.sample_C_inv_Wishart:
        all_cell_samples = np.hstack([dict_all_params['all_cell_samples'], all_cell_samples[1:]])

    # all_params_mixing_matrix_samples_path = directory_save_file+file_ver+'_all_params_mixing_matrix_samples.npy'
    # all_params_mixing_matrix_samples = np.load(all_params_mixing_matrix_samples_path)
    # dict_all_params['all_params_mixing_matrix_samples'] = all_params_mixing_matrix_samples

    all_params_mixing_matrix_samples = np.vstack([dict_all_params['all_params_mixing_matrix_samples'], all_params_mixing_matrix_samples[1:]])

    

# Saving all files
initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_freq_maps_path, flush=True)
np.save(initial_freq_maps_path, input_freq_maps)

initial_cmb_maps_path = directory_save_file+file_ver+'_initial_cmb_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_cmb_maps_path, flush=True)
np.save(initial_cmb_maps_path, input_cmb_maps)

if not(MICMAC_obj.cheap_save):
    all_eta_maps_path = directory_save_file+file_ver+'_all_eta_maps.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_eta_maps_path, flush=True)
    np.save(all_eta_maps_path, all_eta)

    all_s_c_WF_maps_path = directory_save_file+file_ver+'_all_s_c_WF_maps.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_WF_maps_path, flush=True)
    np.save(all_s_c_WF_maps_path, all_s_c_WF_maps)

    all_s_c_fluct_maps_path = directory_save_file+file_ver+'_all_s_c_fluct_maps.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_fluct_maps_path, flush=True)
    np.save(all_s_c_fluct_maps_path, all_s_c_fluct_maps)

elif not(MICMAC_obj.very_cheap_save):
    all_s_c_path = directory_save_file+file_ver+'_all_s_c.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_path, flush=True)
    np.save(all_s_c_path, all_s_c)

if MICMAC_obj.sample_C_inv_Wishart:
    all_cell_samples_path = directory_save_file+file_ver+'_all_cell_samples.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_cell_samples_path, flush=True)
    np.save(all_cell_samples_path, all_cell_samples)
if MICMAC_obj.sample_r_Metropolis:
    all_r_samples_path = directory_save_file+file_ver+'_all_r_samples.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_r_samples_path, flush=True)
    np.save(all_r_samples_path, all_r_samples)

all_params_mixing_matrix_samples_path = directory_save_file+file_ver+'_all_params_mixing_matrix_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_params_mixing_matrix_samples_path, flush=True)
np.save(all_params_mixing_matrix_samples_path, all_params_mixing_matrix_samples)
