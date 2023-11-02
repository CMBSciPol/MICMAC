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


file_ver = 'full_Gchain_v1a'


sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')
from func_tools_for_tests import *
# from get_freq_maps_SO_64 import *
from get_freq_maps_LiteBIRD_64 import *
# from get_freq_maps_LiteBIRD_64_noc1 import *

path_Fisher =  '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_sampling_step_4/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens_synchdust.txt'
Fisher_matrix = np.loadtxt(path_Fisher)

# Setting the parameter of the chain
number_iterations_sampling = 10
# number_iterations_sampling = 10000


# Getting the parameters of the problem
nstokes = 2
nside = 64
lmax = nside*2
lmin = 2

number_frequencies = freq_maps.shape[0]
number_components = 3

limit_iter_cg=2000
tolerance_CG=10**(-12)

n_iter = 8
number_correlations = int(np.ceil(nstokes**2/2) + np.floor(nstokes/2))


param_dict = {'nside':nside, 'lmax':lmax, 'nstokes':nstokes, 'number_correlations':number_correlations,'number_frequencies':number_frequencies, 'number_components':number_components}
npix = 12*nside**2

# Getting C_approx and a first guess for C
input_cmb_maps_extended = np.vstack([np.zeros_like(input_cmb_maps[0,0,...]),input_cmb_maps[0]])
initial_spectra = hp.anafast(input_cmb_maps_extended, lmax=lmax, iter=n_iter)
initial_spectra[0,:] = 0
initial_spectra[nstokes+2:,:] = 0

c_ells_input = np.zeros((6,lmax+1))
c_ell_approx = np.zeros((6,lmax+1))

c_ells_input[:4,...] = initial_spectra[:4,:]
c_ell_approx[1] = initial_spectra[1,:]
c_ell_approx[2] = initial_spectra[2,:]

indices_polar = np.array([1,2,4])

if nstokes == 2 and (c_ells_input.shape[0] != len(indices_polar)):    
    c_ells_input = c_ells_input[indices_polar,:]
    c_ell_approx = c_ell_approx[indices_polar,:]
    # CMB_map_input = CMB_map_input[1:,:]
    # freq_maps = freq_maps[:,1:,:]

# CMB covariance preparation
red_cov_approx_matrix = micmac.get_reduced_matrix_from_c_ell(c_ell_approx)[lmin:,...]
red_cov_matrix_sample = micmac.get_reduced_matrix_from_c_ell(c_ells_input)[lmin:,...]

# Mixing matrix initialization
init_mixing_matrix_obj = micmac.InitMixingMatrix(np.array(instrument['frequency']), number_components, pos_special_freqs=[0,-1])

init_params = init_mixing_matrix_obj.init_params()

mixing_matrix_obj = micmac.MixingMatrix(instrument['frequency'], number_components, init_params, pos_special_freqs=[0,-1])
mixing_matrix_sampled = mixing_matrix_obj.get_B()


minimum_std_Fisher = scipy.linalg.sqrtm(np.linalg.inv(Fisher_matrix))
minimum_std_Fisher_diag = np.diag(minimum_std_Fisher)

len_pos_special_freqs = len(mixing_matrix_obj.pos_special_freqs)
step_size_array = np.zeros((2,number_frequencies-len_pos_special_freqs))
step_size_array[0,:] = minimum_std_Fisher_diag[:number_frequencies-len_pos_special_freqs]
step_size_array[1,:] = minimum_std_Fisher_diag[number_frequencies-len_pos_special_freqs:2*(number_frequencies-len_pos_special_freqs)]
number_steps_sampler = 100

first_guess_params_mixing_matrix =  np.random.uniform(low=init_params-5*step_size_array,high=init_params+5*step_size_array, size=init_params.shape)
n_walkers = 1
num_warmup = 0
fullsky_ver = True
slow_ver = False

# Noise initialization
freq_inverse_noise = micmac.get_noise_covar(instrument['depth_p'], nside)
BtinvNB = micmac.get_inv_BtinvNB(freq_inverse_noise, mixing_matrix_sampled)
BtinvN_sqrt = micmac.get_BtinvN(scipy.linalg.sqrtm(freq_inverse_noise), mixing_matrix_sampled)
BtinvN = micmac.get_BtinvN(freq_inverse_noise, mixing_matrix_sampled)


assert freq_inverse_noise.shape[0] == number_frequencies
assert freq_inverse_noise.shape[1] == number_frequencies


all_eta = np.zeros((number_iterations_sampling+1, nstokes, npix))
all_s_c_WF_maps = np.zeros((number_iterations_sampling+1, nstokes, npix))
all_s_c_fluct_maps = np.zeros((number_iterations_sampling+1, nstokes, npix))
all_cell_samples = np.zeros((number_iterations_sampling+1, number_correlations, lmax+1))
all_params_mixing_matrix_samples = np.zeros((number_iterations_sampling+1, number_correlations-1, number_frequencies-len(mixing_matrix_obj.position)))

# all_CMB_retrieval = np.zeros((number_iterations_sampling+1, nstokes, npix))

# Initial values
c_ell_sampled = np.copy(c_ells_input)
all_cell_samples[0,...] = c_ell_sampled
all_params_mixing_matrix_samples[0,...] = mixing_matrix_obj.params


wiener_filter_term = np.zeros((nstokes,npix))
fluctuation_maps = np.zeros((nstokes,npix))

initial_freq_maps = np.copy(freq_maps)

number_steps_sampler_random = np.zeros(number_iterations_sampling)

time_start_sampling = time.time()
# Start sampling !!!
for iteration in range(number_iterations_sampling):
    print("### Start Iteration n°", iteration, flush=True)

    # Application of new mixing matrix
    BtinvNB = micmac.get_inv_BtinvNB(freq_inverse_noise, mixing_matrix_sampled)
    BtinvN_sqrt = micmac.get_BtinvN(np.sqrt(freq_inverse_noise), mixing_matrix_sampled)
    # s_cML = np.einsum('cd,df,fsp->s', BtinvNB, micmac.get_BtinvN(freq_inverse_noise, mixing_matrix_sampled), initial_freq_maps)[0] # Computation of E^t (B^t N^{-1} B)^{-1} B^t N^{-1} d
    s_cML = micmac.get_Wd(freq_inverse_noise, mixing_matrix_sampled, initial_freq_maps, jax_use=False)[0]
        
    
    # Sampling step 1 : Sample eta term with formulation :
    map_random_x = []
    map_random_y = []
    time_start_sampling_eta_maps = time.time()
    eta_maps_sample = micmac.get_sampling_eta_v2(param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, lmin=lmin, n_iter=n_iter)
    time_sampling_eta_maps = (time.time()-time_start_sampling_eta_maps)/60
    print("##### Sampling eta_maps at iteration {} in {} minutes".format(iteration+1, time_sampling_eta_maps), flush=True)

    # Recording of the samples
    all_eta[iteration+1,...] = eta_maps_sample

    assert eta_maps_sample.shape[0] == nstokes
    assert eta_maps_sample.shape[1] == npix
    

    # Sampling step 2 : sampling of Gaussian variable s_c 
    initial_guess_WF = np.copy(wiener_filter_term)
    time_start_sampling_s_c_WF = time.time()
    wiener_filter_term = micmac.solve_generalized_wiener_filter_term(param_dict, s_cML, red_cov_matrix_sample, BtinvNB, initial_guess=initial_guess_WF, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG)
    time_sampling_s_c_WF = (time.time()-time_start_sampling_s_c_WF)/60
    print("##### Sampling s_c_WF at iteration {} in {} minutes".format(iteration+1, time_sampling_s_c_WF), flush=True)

    initial_guess_fluct = np.copy(fluctuation_maps)
    map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))
    time_start_sampling_s_c_fluct = time.time()
    fluctuation_maps = micmac.get_fluctuating_term_maps(param_dict, red_cov_matrix_sample, BtinvNB, BtinvN_sqrt, map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, initial_guess=initial_guess_fluct, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG)
    time_sampling_s_c_fluct = (time.time()-time_start_sampling_s_c_fluct)/60
    print("##### Sampling s_c_fluct at iteration {} in {} minutes".format(iteration+1, time_sampling_s_c_fluct), flush=True)
    
    # Recording of the samples
    all_s_c_WF_maps[iteration+1,...] = wiener_filter_term
    all_s_c_fluct_maps[iteration+1,...] = fluctuation_maps

    s_c_sample = wiener_filter_term + fluctuation_maps
    
    assert wiener_filter_term.shape[0] == nstokes
    assert wiener_filter_term.shape[1] == npix
    assert fluctuation_maps.shape[0] == nstokes
    assert fluctuation_maps.shape[1] == npix
    assert s_c_sample.shape[0] == nstokes
    assert s_c_sample.shape[1] == npix

    # Sampling step 3 : sampling of CMB covariance C

    # Preparation of sampling step 4
    extended_CMB_maps = np.zeros((number_components, nstokes, npix))
    extended_CMB_maps[0] = s_c_sample
    full_data_without_CMB = initial_freq_maps - np.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
    assert full_data_without_CMB.shape[0] == number_frequencies
    assert full_data_without_CMB.shape[1] == nstokes
    assert full_data_without_CMB.shape[2] == npix
    
    # Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
    c_ells_Wishart = micmac.get_cell_from_map(s_c_sample, lmax=lmax, n_iter=n_iter)
    red_c_ells_inv_Wishart_sample = np.zeros((lmax+1, nstokes, nstokes))
    c_ells_inv_Wishart_sample = np.zeros((number_correlations, lmax+1))

    time_start_sampling_C = time.time()
    red_cov_matrix_sample = micmac.get_inverse_wishart_sampling_from_c_ells(np.copy(c_ells_Wishart), l_min=lmin)#[lmin:]
    c_ells_inv_Wishart_sample = micmac.get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)
    time_sampling_C = (time.time()-time_start_sampling_C)/60
    print("##### Sampling C at iteration {} in {} minutes".format(iteration+1, time_sampling_C), flush=True)
    # Recording of the samples
    all_cell_samples[iteration+1,...] = micmac.get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)

    assert red_cov_matrix_sample.shape[0] == lmax + 1 - lmin
    
    
    # Sampling step 4
    time_start_sampling_Bf = time.time()
    number_steps_sampler_random[iteration] = number_steps_sampler + np.random.randint(0,number_steps_sampler)
    few_params_mixing_matrix_samples = micmac.get_sample_B_f(micmac.new_get_conditional_proba_full_likelihood_JAX_from_params, step_size_array, number_steps_sampler_random[iteration], first_guess_params_mixing_matrix.ravel(), random_PRNGKey=jax.random.PRNGKey(100+iteration), n_walkers=n_walkers, num_warmup=num_warmup, pos_special_freqs=mixing_matrix_obj.pos_special_freqs, fullsky_ver=fullsky_ver, slow_ver=slow_ver, param_dict=param_dict, full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, freq_inverse_noise=freq_inverse_noise, red_cov_approx_matrix=red_cov_approx_matrix, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG, with_prints=False)
    params_mixing_matrix_sample = few_params_mixing_matrix_samples[0,-1,:].reshape((2,number_frequencies-len_pos_special_freqs))
    time_sampling_Bf = (time.time()-time_start_sampling_Bf)/60
    print("##### Sampling B_f at iteration {} in {} minutes".format(iteration+1, time_sampling_Bf), flush=True)


    # Recording of the samples
    all_params_mixing_matrix_samples[iteration+1,...] = params_mixing_matrix_sample
    
    mixing_matrix_obj.update_params(params_mixing_matrix_sample)
    mixing_matrix_sampled = np.copy(mixing_matrix_obj.get_B())

    assert mixing_matrix_sampled.shape[0] == number_frequencies
    assert mixing_matrix_sampled.shape[1] == number_components
    

    # Few tests to verify everything's fine
    all_eigenvalues = np.linalg.eigh(red_cov_matrix_sample[lmin:])[0]
    assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_cov_matrix_sample[lmin:])[0])>10**(-15)]>0)


    # if iteration%50 == 0:
    #     print("### Iteration n°", iteration, flush=True)
    print("### Iteration n°", iteration, flush=True)

time_full_chain = (time.time()-time_start_sampling)/60
print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

directory_save_file = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v1/save_directory/'

file = open(directory_save_file + file_ver + '_parameters.par', 'w')
param_dict['number_iterations_sampling'] = number_iterations_sampling
param_dict['lmin'] = lmin
param_dict['limit_iter_cg'] = limit_iter_cg
param_dict['tolerance_CG'] = tolerance_CG
param_dict['n_iter'] = n_iter
for key,item in param_dict:
    file.write(key + ' = ' + item + '\n')
file.close()

initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_freq_maps_path, flush=True)
np.save(initial_freq_maps_path, initial_freq_maps)

all_eta_maps_path = directory_save_file+file_ver+'_all_eta_maps.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_eta_maps_path, flush=True)
np.save(initial_freq_maps_path, initial_freq_maps)

all_s_c_WF_maps_path = directory_save_file+file_ver+'_all_s_c_WF_maps.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_WF_maps_path, flush=True)
np.save(all_s_c_WF_maps_path, all_s_c_WF_maps)

all_s_c_fluct_maps_path = directory_save_file+file_ver+'_all_s_c_fluct_maps.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_fluct_maps_path, flush=True)
np.save(all_s_c_fluct_maps_path, all_s_c_fluct_maps)

all_cell_samples_path = directory_save_file+file_ver+'_all_cell_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_cell_samples_path, flush=True)
np.save(all_cell_samples_path, all_cell_samples)

all_params_mixing_matrix_samples_path = directory_save_file+file_ver+'_all_params_mixing_matrix_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_params_mixing_matrix_samples_path, flush=True)
np.save(all_params_mixing_matrix_samples_path, all_params_mixing_matrix_samples)
