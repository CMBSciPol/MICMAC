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

file_ver = 'full_v9_Gchain_SO_noise_v7a' # -> 10 iterations +  test_full_chain_v1a ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
file_ver = 'full_v9_Gchain_SO_noise_v7b' # -> 1000 iterations +  test_full_chain_v1b ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
file_ver = 'full_v9_Gchain_SO_noise_v7c_alt' # -> 1000 iterations +  test_full_chain_v1b ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
file_ver = 'full_v9_Gchain_SO_noise_v7d_alt' # -> 1000 iterations + gap=5 +  test_full_chain_v1b ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
file_ver = 'full_v91_Gchain_SO_64_v9a' # -> 3000 iterations + gap=10 + test_full_chain_v1c + initial_guess_r = 1e-3 ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
file_ver = 'full_v91_Gchain_SO_64_v9b' # -> 4000 iterations + gap=4 + test_full_chain_v1c + initial_guess_r = 1e-3 ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
file_ver = 'full_v91_Gchain_SO_64_v10a' # -> 1000 iterations + gap=4 + test_full_chain_v1d + initial_guess_r = 1e-3 ; test_full_chain_v1a + C_approx only lensing ; start_r=10**(-2) + exact values B_f
reduction_noise = 1

sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')
# from func_tools_for_tests import *
# from get_freq_maps_SO_64 import *
# from get_freq_maps_SO_64_lower_noise import *
# from get_freq_maps_LiteBIRD_64 import *

from fgbuster.observation_helpers import *
from micmac import *

# get input cmb
# input_cmb_maps = get_observation(instrument, cmb_model, nside=NSIDE, noise=False)[:, 1:, :]   # keep only Q and U
# print("Shape for input cmb maps :", input_cmb_maps.shape)

# freq_maps = freq_maps_fgs + input_cmb_maps

# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_sampling_step_4/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens_synchdust.txt'
# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens.txt'
path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/fisher_so-sat_d0s0_lmin2_lmax128_nomask_Alens1.0_r0.0_B_noiselens.txt'

working_directory_path = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v3/'
directory_save_file = working_directory_path + 'save_directory/'
directory_toml_file = working_directory_path + 'toml_params/'

path_toml_file = directory_toml_file + 'test_full_chain_v1a.toml'
path_toml_file = directory_toml_file + 'test_full_chain_v1b.toml'
path_toml_file = directory_toml_file + 'test_full_chain_v1c.toml'
path_toml_file = directory_toml_file + 'test_full_chain_v1d.toml'


MICMAC_obj = micmac.create_MICMAC_sampler_from_toml_file(path_toml_file)

with open(path_toml_file) as f:
    dictionary_parameters = toml.load(f)
f.close()

# General parameters
cmb_model = 'c1'
fgs_model = 'd0s0'
model = cmb_model+fgs_model
noise = True
# noise = False
noise_seed = 42
instr_name = dictionary_parameters['instrument_name'] #'SO_SAT'

# get instrument from public database
instrument = get_instrument(instr_name)

instrument['depth_p'] /= reduction_noise
# get input freq maps
np.random.seed(noise_seed)
# freq_maps = get_observation(instrument, model, nside=NSIDE, noise=noise)[:, 1:, :]   # keep only Q and U
freq_maps_fgs = get_observation(instrument, fgs_model, nside=MICMAC_obj.nside, noise=noise)[:, 1:, :]   # keep only Q and U
print("Shape for input frequency maps :", freq_maps_fgs.shape)

MICMAC_obj.freq_inverse_noise = micmac.get_noise_covar(instrument['depth_p'], MICMAC_obj.nside)



# Fisher_matrix = np.loadtxt(path_Fisher)
# minimum_std_Fisher = scipy.linalg.sqrtm(np.linalg.inv(Fisher_matrix))
# minimum_std_Fisher_diag = np.diag(minimum_std_Fisher)

# len_pos_special_freqs = len(MICMAC_obj.pos_special_freqs)
# step_size_B_f = np.zeros(((MICMAC_obj.number_frequencies-len_pos_special_freqs)*2))
# step_size_B_f[:MICMAC_obj.number_frequencies-len_pos_special_freqs] = minimum_std_Fisher_diag[:MICMAC_obj.number_frequencies-len_pos_special_freqs]
# step_size_B_f[MICMAC_obj.number_frequencies-len_pos_special_freqs:] = minimum_std_Fisher_diag[MICMAC_obj.number_frequencies-len_pos_special_freqs:2*(MICMAC_obj.number_frequencies-len_pos_special_freqs)]

# MICMAC_obj.step_size_B_f = step_size_B_f
if jnp.size(MICMAC_obj.step_size_B_f) == 1:
    MICMAC_obj.step_size_B_f = MICMAC_obj.step_size_B_f*jnp.ones((MICMAC_obj.number_frequencies-len(MICMAC_obj.pos_special_freqs))*2)

input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = MICMAC_obj.generate_input_freq_maps_from_fgs(freq_maps_fgs, return_only_freq_maps=False)

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

CMB_c_ell = np.zeros_like(c_ell_approx)
CMB_c_ell[:,MICMAC_obj.lmin:] = (theoretical_r0_total + MICMAC_obj.r_true*theoretical_r1_tensor)

len_pos_special_freqs = len(MICMAC_obj.pos_special_freqs)
step_size_B_f = MICMAC_obj.step_size_B_f*jnp.ones((MICMAC_obj.number_frequencies-len_pos_special_freqs)*2)
# gap = 10
# gap = 5
# gap = 4
gap = 2
# gap = 0
init_params_mixing_matrix = exact_params_mixing_matrix.ravel(order='F') + gap*np.random.uniform(low=-step_size_B_f,high=step_size_B_f, size=((MICMAC_obj.number_frequencies-len_pos_special_freqs)*2))

print(f'Exact param matrix : {exact_params_mixing_matrix}')
print(f'Initial param matrix : {init_params_mixing_matrix}')

# initial_guess_r=10**(-2)
initial_guess_r=10**(-3)

time_start_sampling = time.time()
MICMAC_obj.perform_sampling(input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=initial_guess_r, initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
                         theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)
# MICMAC_obj.perform_sampling_v2(input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
#                          initial_guess_r=initial_guess_r, initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
#                          theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)
time_full_chain = (time.time()-time_start_sampling)/60
print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

all_eta  = MICMAC_obj.all_samples_eta
all_s_c_WF_maps = MICMAC_obj.all_samples_wiener_filter_maps
all_s_c_fluct_maps = MICMAC_obj.all_samples_fluctuation_maps
all_cell_samples = MICMAC_obj.all_samples_CMB_c_ell
all_r_samples = MICMAC_obj.all_samples_r
all_params_mixing_matrix_samples = MICMAC_obj.all_params_mixing_matrix_samples

# directory_save_file = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v2/save_directory/'

# all_attr_dict = Full_Gibbs_Sampler.__dict__
# dict_last_samples = all_attr_dict['dict_last_samples']
# np.savez(directory_save_file + file_ver + '_last_samples', **dict_last_samples)
# del all_attr_dict['dict_last_samples']
# np.savez(directory_save_file + file_ver + '_parameters', **all_attr_dict)

initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_freq_maps_path, flush=True)
np.save(initial_freq_maps_path, input_freq_maps)

initial_cmb_maps_path = directory_save_file+file_ver+'_initial_cmb_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_cmb_maps_path, flush=True)
np.save(initial_cmb_maps_path, input_cmb_maps)

all_eta_maps_path = directory_save_file+file_ver+'_all_eta_maps.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_eta_maps_path, flush=True)
np.save(all_eta_maps_path, all_eta)

all_s_c_WF_maps_path = directory_save_file+file_ver+'_all_s_c_WF_maps.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_WF_maps_path, flush=True)
np.save(all_s_c_WF_maps_path, all_s_c_WF_maps)

all_s_c_fluct_maps_path = directory_save_file+file_ver+'_all_s_c_fluct_maps.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_fluct_maps_path, flush=True)
np.save(all_s_c_fluct_maps_path, all_s_c_fluct_maps)

all_cell_samples_path = directory_save_file+file_ver+'_all_cell_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_cell_samples_path, flush=True)
np.save(all_cell_samples_path, all_cell_samples)

all_r_samples_path = directory_save_file+file_ver+'_all_r_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_r_samples_path, flush=True)
np.save(all_r_samples_path, all_r_samples)

all_params_mixing_matrix_samples_path = directory_save_file+file_ver+'_all_params_mixing_matrix_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_params_mixing_matrix_samples_path, flush=True)
np.save(all_params_mixing_matrix_samples_path, all_params_mixing_matrix_samples)
