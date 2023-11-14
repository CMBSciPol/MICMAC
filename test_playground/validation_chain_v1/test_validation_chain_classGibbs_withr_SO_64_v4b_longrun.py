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


# file_ver = 'full_Gchain_SO_v1c'
file_ver = 'full_Gchain_SO_noise_v1d'
file_ver = 'full_Gchain_onlyB_SO_noise_v2a' # 1 Fisher start from params
file_ver = 'full_Gchain_onlyB_SO_noise_v2b' # 5 Fisher start from params + forcing c_ell_Wishart to be 0
file_ver = 'full_Gchain_onlyB_SO_noise_v2c' # 10 Fisher start from params + sampling directly inverse Gamma
file_ver = 'full_Gchain_onlyB_SO_noise_v2d' # 10 Fisher start from params + sampling from inverse Wishart
file_ver = 'full_Gchain_onlyB_SO_noise_v3a' # 10 Fisher start from params + sampling for r instead of C ; r=0.001, start r=0.1
file_ver = 'full_Gchain_onlyB_SO_noise_v3b' # 10 Fisher start from params + sampling for r instead of C ; r=0.001, start=10**(-8)
only_select_Bmodes = True
file_ver = 'full_Gchain_SO_noise_withr_v4a' # 10 Fisher start from params + with E + sampling for r instead of C ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_SO_noise_withr_v4ab' # 10 Fisher start from params + with E + sampling for r instead of C + only tensor modes in C_approx; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_SO_noise_withr_v4ac' # 10 Fisher start from params + with E + sampling for r instead of C + B and E lensing in C_approx; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_SO_noise_withr_v4ad' # 10 Fisher start from params + with E + sampling for r instead of C + B lensing in C_approx and almost no E ; r=0.001, start=10**(-8)
only_select_Bmodes = False
file_ver = 'full_Gchain_onlyB_SO_noise_v3c' # 10 Fisher start from params + sampling for r instead of C + C_approx only lensing ; r=0.001, start=10**(-8)
only_select_Bmodes = True
file_ver = 'full_Gchain_onlyB_SO_noise_v3d' # 10 Fisher start from params + sampling for r + C_approx only lensing ; minimal number of E-modes put to 0 artificially ; r=0.001, start=10**(-8)
only_select_Bmodes = True
no_Emodes_CMB = False
file_ver = 'full_Gchain_SO_noise_withr_v4b' # 10 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E ; r=0.001, start=10**(-8)
only_select_Bmodes = False
no_Emodes_CMB = True
file_ver = 'full_Gchain_onlyB_SO_noise_v3e' # Long run 500 iterations -- 10 Fisher start from params + sampling for r + C_approx only lensing ; minimal number of E-modes put to 0 artificially ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_onlyB_SO_noise_v3f' # Mini-test long run 5 iterations with chain-in-code -- 10 Fisher start from params + sampling for r + C_approx only lensing ; minimal number of E-modes put to 0 artificially + step-size r 10**(-5) ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_onlyB_SO_noise_v3g' # Mini-test long run 5 iterations with chain-in-code -- 10 Fisher start from params + sampling for r + C_approx only lensing ; minimal number of E-modes put to 0 artificially + step-size r 10**(-4) ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_onlyB_SO_noise_v3h' # Mini-test long run 5 iterations with chain-in-code -- 2 Fisher start from params + sampling for r + C_approx only lensing ; minimal number of E-modes put to 0 artificially + step-size r 10**(-4) ; r=0.001, start=10**(-8)
only_select_Bmodes = True
no_Emodes_CMB = False
sample_eta_B_f = True
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v1a' #  Mini-test classGibbs long run 5 iterations with chain-in-code -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v1b' #  Mini-test classGibbs long run 5 iterations with chain-in-code -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v1c' #  Mini-test classGibbs long run 5 iterations with chain-in-code noiseless -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v1d' #  Mini-test classGibbs long run 5 iterations with chain-in-code noiseless -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-5) ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v1e' #  Mini-test classGibbs long run 5 iterations with chain-in-code with noise -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v1f' #  Mini-test classGibbs long run 5 iterations with chain-in-code with noise -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 1000 ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v2a' #  Long run 30 iterations with chain-in-code with noise -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 1500 ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v2b' #  Long run 5 iterations with chain-in-code with noise and overrelaxation  -0.995 -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 1500 ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v2c' #  Long run 5 iterations with chain-in-code with noise and overrelaxation  -0.89 -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 1500 ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_SO_noise_withr_v2d' #  Long run 5 iterations with chain-in-code with noise and overrelaxation  -0.1 -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 1500 ; r=0.001, start=10**(-8)
only_select_Bmodes = False
no_Emodes_CMB = False
sample_eta_B_f = False
file_ver = 'full_Gchain_classGibbs_Emodes_SO_noise_withr_v4a' #  Full long run 5 iterations with chain-in-code with noise and E-modes -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 1500 ; r=0.001, start=10**(-8)
file_ver = 'full_Gchain_classGibbs_Emodes_SO_noise_withr_v4b' #  Full long run 200 iterations with chain-in-code with noise and E-modes -- 0 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + step-size r 10**(-4) + limit_iter_CG 2000 ; r=0.001, start=10**(-8)
only_select_Bmodes = False
no_Emodes_CMB = False
sample_eta_B_f = True
# overrelaxation_param = -0.995
# overrelaxation_param = -0.89
# overrelaxation_param = -0.1
r_true = 0.001
# r_true = 10**(-4)

sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')
from func_tools_for_tests import *
from get_freq_maps_SO_64 import *
# from get_freq_maps_LiteBIRD_64 import *
# from get_freq_maps_LiteBIRD_64_noc1 import *


# Setting the parameter of the chain
# number_iterations_sampling = 5 #500
number_iterations_sampling = 200 #30 #150
# number_iterations_sampling = 10000



# Getting the parameters of the problem
nstokes = 2
nside = 64
lmax = nside*2
lmin = 2

pos_special_freqs = [0,-1]

number_frequencies = freq_maps.shape[0]
number_components = 3

limit_iter_cg=2000
# limit_iter_cg=1500
tolerance_CG=10**(-12)
# tolerance_CG=10**(-8)

n_iter = 8
number_correlations = int(np.ceil(nstokes**2/2) + np.floor(nstokes/2))

npix = 12*nside**2

# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_sampling_step_4/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens_synchdust.txt'
# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens.txt'
path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/fisher_so-sat_d0s0_lmin2_lmax128_nomask_Alens1.0_r0.0_B_noiselens.txt'
Fisher_matrix = np.loadtxt(path_Fisher)

minimum_std_Fisher = scipy.linalg.sqrtm(np.linalg.inv(Fisher_matrix))
minimum_std_Fisher_diag = np.diag(minimum_std_Fisher)

number_steps_sampler_B_f = 100
number_steps_sampler_r = 100

len_pos_special_freqs = len(pos_special_freqs)
step_size_B_f = np.zeros(((number_frequencies-len_pos_special_freqs)*2))
step_size_B_f[:number_frequencies-len_pos_special_freqs] = minimum_std_Fisher_diag[:number_frequencies-len_pos_special_freqs]
step_size_B_f[number_frequencies-len_pos_special_freqs:] = minimum_std_Fisher_diag[number_frequencies-len_pos_special_freqs:2*(number_frequencies-len_pos_special_freqs)]

# step_size_r = 10**(-5)
step_size_r = 10**(-4)

freq_inverse_noise = micmac.get_noise_covar(instrument['depth_p'], nside)

sample_r_Metropolis=True
sample_C_inv_Wishart=False

n_walkers = 1
num_warmup = 0
fullsky_ver = True
slow_ver = False

Full_Gibbs_Sampler = micmac.MICMAC_Sampler(nside, lmax, nstokes, np.array(instrument['frequency']), freq_inverse_noise, 
                 number_components=number_components, lmin=lmin,
                 r_true=r_true, pos_special_freqs=pos_special_freqs, only_select_Bmodes=only_select_Bmodes, no_Emodes_CMB=no_Emodes_CMB,
                 sample_eta_B_f=sample_eta_B_f,
                 sample_r_Metropolis=sample_r_Metropolis, sample_C_inv_Wishart=sample_C_inv_Wishart,
                 n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance_CG=tolerance_CG,
                 n_walkers=n_walkers, step_size_B_f=step_size_B_f, step_size_r=step_size_r,
                 fullsky_ver=fullsky_ver, slow_ver=slow_ver,
                 number_steps_sampler_B_f=number_steps_sampler_B_f, number_steps_sampler_r=number_steps_sampler_r,
                 number_iterations_sampling=number_iterations_sampling, number_iterations_done=0)

# Re-Defining the data if needed
indices_polar = np.array([1,2,4])
partial_indices_polar = indices_polar[:nstokes]


input_freq_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = Full_Gibbs_Sampler.generate_input_freq_maps_from_fgs(freq_maps_fgs, return_only_freq_maps=False)



theoretical_r0_total = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total)#[partial_indices_polar,:]
theoretical_r1_tensor = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r1_tensor)#[partial_indices_polar,:]


c_ell_approx = np.zeros((3,lmax+1))
c_ell_approx[0,lmin:] = theoretical_r0_total[0,:]
c_ell_approx[1,lmin:] = theoretical_r0_total[1,:]

# Params mixing matrix
init_mixing_matrix_obj = micmac.InitMixingMatrix(np.array(instrument['frequency']), number_components, pos_special_freqs=pos_special_freqs)
exact_params_mixing_matrix = init_mixing_matrix_obj.init_params()

# First guesses
initial_wiener_filter_term = np.zeros((nstokes, npix))
initial_fluctuation_maps = np.zeros((nstokes, npix))

CMB_covariance = np.zeros_like(c_ell_approx)
CMB_covariance[:,lmin:] = (theoretical_r0_total + r_true*theoretical_r1_tensor)

gap = 10
gap = 2
gap = 0
init_params_mixing_matrix = exact_params_mixing_matrix.ravel(order='F') + gap*np.random.uniform(low=-step_size_B_f,high=step_size_B_f, size=((number_frequencies-len_pos_special_freqs)*2))

print(f'Exact param matrix : {exact_params_mixing_matrix}')
print(f'Initial param matrix : {init_params_mixing_matrix}')

initial_guess_r=10**(-8)

time_start_sampling = time.time()
all_eta, all_s_c_WF_maps, all_s_c_fluct_maps, all_cell_samples, all_r_samples, all_params_mixing_matrix_samples = Full_Gibbs_Sampler.perform_sampling(
                        input_freq_maps, c_ell_approx, np.copy(CMB_covariance), init_params_mixing_matrix.reshape((number_frequencies-len_pos_special_freqs,2),order='F'), 
                         initial_guess_r=initial_guess_r, 
                         initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
                         theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)

time_full_chain = (time.time()-time_start_sampling)/60
print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

directory_save_file = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v1/save_directory/'

all_attr_dict = Full_Gibbs_Sampler.__dict__
dict_last_samples = all_attr_dict['dict_last_samples']
np.savez(directory_save_file + file_ver + '_last_samples', **dict_last_samples)
del all_attr_dict['dict_last_samples']
np.savez(directory_save_file + file_ver + '_parameters', **all_attr_dict)

initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_freq_maps_path, flush=True)
np.save(initial_freq_maps_path, input_freq_maps)

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
