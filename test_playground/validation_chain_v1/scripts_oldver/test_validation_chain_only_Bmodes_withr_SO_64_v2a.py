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
file_ver = 'full_Gchain_SO_noise_withr_v4b' # 10 Fisher start from params + with E (but no E CMB) + sampling for r instead of C + B lensing in C_approx and only tensor E + eta going thourgh C_approx ; r=0.001, start=10**(-8)
only_select_Bmodes = False
no_Emodes_CMB = True

r_true = 0.001
# r_true = 10**(-4)

sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')
from func_tools_for_tests import *
from get_freq_maps_SO_64 import *
# from get_freq_maps_LiteBIRD_64 import *
# from get_freq_maps_LiteBIRD_64_noc1 import *

# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_sampling_step_4/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens_synchdust.txt'
# path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/fisher_litebird_d0s0_lmin2_lmax128_masked_Alens1.0_r0.0_B_noiselens.txt'
path_Fisher = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/fisher_so-sat_d0s0_lmin2_lmax128_nomask_Alens1.0_r0.0_B_noiselens.txt'
Fisher_matrix = np.loadtxt(path_Fisher)

# Setting the parameter of the chain
number_iterations_sampling = 5
# number_iterations_sampling = 30 #150
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
tolerance_CG=10**(-12)
# tolerance_CG=10**(-8)

n_iter = 8
number_correlations = int(np.ceil(nstokes**2/2) + np.floor(nstokes/2))


param_dict = {'nside':nside, 'lmax':lmax, 'nstokes':nstokes, 'number_correlations':number_correlations,'number_frequencies':number_frequencies, 'number_components':number_components}
npix = 12*nside**2

# Re-Defining the data if needed
indices_polar = np.array([1,2,4])
partial_indices_polar = indices_polar[:nstokes]


all_spectra_r0 = generate_power_spectra_CAMB(nside*2, r=0, typeless_bool=True)
all_spectra_r1 = generate_power_spectra_CAMB(nside*2, r=1, typeless_bool=True)

camb_cls_r0 = all_spectra_r0['total'][:lmax+1,partial_indices_polar]
lensing_spectra_r0 = all_spectra_r0['lensed_scalar'][:lmax+1,partial_indices_polar]
tensor_spectra_r0 = all_spectra_r0['tensor'][:lmax+1,partial_indices_polar]
lens_potential_spectra_r0 = all_spectra_r0['lens_potential'][:lmax+1,partial_indices_polar]
unlensed_scalar_spectra_r0 = all_spectra_r0['unlensed_scalar'][:lmax+1,partial_indices_polar]
unlensed_total_spectra_r0 = all_spectra_r0['unlensed_total'][:lmax+1,partial_indices_polar]

camb_cls_r1 = all_spectra_r1['total'][:lmax+1,partial_indices_polar]
lensing_spectra_r1 = all_spectra_r1['lensed_scalar'][:lmax+1,partial_indices_polar]
tensor_spectra_r1 = all_spectra_r1['tensor'][:lmax+1,partial_indices_polar]
lens_potential_spectra_r1 = all_spectra_r1['lens_potential'][:lmax+1,partial_indices_polar]
unlensed_scalar_spectra_r1 = all_spectra_r1['unlensed_scalar'][:lmax+1,partial_indices_polar]
unlensed_total_spectra_r1 = all_spectra_r1['unlensed_total'][:lmax+1,partial_indices_polar]

theoretical_r1_tensor = np.zeros((number_correlations,lmax+1))
theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

theoretical_r1_tensor[:nstokes,...] = tensor_spectra_r1.T
theoretical_r0_total[:nstokes,...] = unlensed_scalar_spectra_r0.T

theoretical_red_cov_r1_tensor = micmac.get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[lmin:]
theoretical_red_cov_r0_total = micmac.get_reduced_matrix_from_c_ell(theoretical_r0_total)[lmin:]

if only_select_Bmodes:
    print("~~~ With case no E-modes in CMB or FGs (and neither in eta or fluctuation term of s_c) !!!", flush=True)
    c_ell_select_only_Bmodes = np.zeros((6,lmax+1))
    c_ell_select_only_Bmodes[2,lmin:] = 1
    red_cov_select_Bmodes = micmac.get_reduced_matrix_from_c_ell(c_ell_select_only_Bmodes[indices_polar,...])[lmin:,...]

    theoretical_red_cov_r1_tensor = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, theoretical_red_cov_r1_tensor)
    theoretical_red_cov_r0_total = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, theoretical_red_cov_r0_total)
    # theoretical_red_cov_r1_tensor[:,:,0] = 0
    # theoretical_red_cov_r0_total[:,:,0] = 0
    theoretical_red_cov_r1_tensor[:,0,0] = 10**(-30)
    theoretical_red_cov_r0_total[:,0,0] = 10**(-30)

    theoretical_r0_total[0,:] = 10**(-30)
    theoretical_r1_tensor[0,:] = 10**(-30)
    # theoretical_r0_total[2,:] = 0
    # theoretical_r1_tensor[2,:] = 0
    for freq in range(number_frequencies):
            freq_maps_fgs[freq] = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(freq_maps_fgs[freq]), red_cov_select_Bmodes, lmin=2, n_iter=n_iter)#[1:,...]


if no_Emodes_CMB:
    print("~~~ With case only tensor E-modes in CMB !!!", flush=True)
    c_ell_select_only_Bmodes = np.zeros((6,lmax+1))
    c_ell_select_only_Bmodes[2,lmin:] = 1
    red_cov_select_Bmodes = micmac.get_reduced_matrix_from_c_ell(c_ell_select_only_Bmodes[indices_polar,...])[lmin:,...]

    theoretical_red_cov_r0_total = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, theoretical_red_cov_r0_total)
    # theoretical_red_cov_r0_total[:,:,0] = 0
    theoretical_red_cov_r0_total[:,0,0] = 10**(-30)

    theoretical_r0_total[0,:] = 10**(-30)
    theoretical_r0_total[2,:] = 0
    # theoretical_r1_tensor[2,:] = 0

true_cmb_specra = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total + r_true*theoretical_red_cov_r1_tensor)
true_cmb_specra_extended = np.zeros((6,lmax+1))
true_cmb_specra_extended[indices_polar,lmin:] = true_cmb_specra

input_cmb_maps_alt = hp.synfast(true_cmb_specra_extended, nside=nside, new=True, lmax=lmax)[1:,...]

# if only_select_Bmodes:
    # input_cmb_maps_alt = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(input_cmb_maps_alt), red_cov_select_Bmodes, lmin=2, n_iter=n_iter)


input_cmb_maps = np.repeat(input_cmb_maps_alt.ravel(order='F'), number_frequencies).reshape((number_frequencies,nstokes,npix),order='F')
freq_maps = input_cmb_maps + freq_maps_fgs

true_red_cov_cmb_specra = micmac.get_reduced_matrix_from_c_ell(true_cmb_specra)

# Getting C_approx and a first guess for C

# input_cmb_maps_extended = np.vstack([np.zeros_like(input_cmb_maps[0,0,...]),input_cmb_maps[0]])
# initial_spectra = hp.anafast(input_cmb_maps_extended, lmax=lmax, iter=n_iter)
# initial_spectra[0,:] = 0
# initial_spectra[nstokes+2:,:] = 0

initial_spectra = np.zeros((6,lmax+1))
initial_spectra[indices_polar,lmin:] = true_cmb_specra

c_ells_input = np.zeros((6,lmax+1))
c_ell_approx = np.zeros((6,lmax+1))

c_ells_input[:4,:] = initial_spectra[:4,:]
# c_ell_approx[1] = initial_spectra[1,:]
# c_ell_approx[2] = initial_spectra[2,:]
c_ell_approx[1,:] = theoretical_r0_total[0,:]
# c_ell_approx[1] = 10**(-30)
c_ell_approx[2,:] = theoretical_r0_total[1,:]


if nstokes == 2 and (c_ells_input.shape[0] != len(indices_polar)):    
    c_ells_input = c_ells_input[indices_polar,:]
    c_ell_approx = c_ell_approx[indices_polar,:]
    # CMB_map_input = CMB_map_input[1:,:]
    # freq_maps = freq_maps[:,1:,:]


# CMB covariance preparation
red_cov_approx_matrix = micmac.get_reduced_matrix_from_c_ell(c_ell_approx)[lmin:,...]
red_cov_matrix_sample = micmac.get_reduced_matrix_from_c_ell(c_ells_input)[lmin:,...]

# if only_select_Bmodes:
    # red_cov_approx_matrix = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, red_cov_approx_matrix)
    # red_cov_matrix_sample = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, red_cov_matrix_sample)

    # for freq in range(number_frequencies):
    #     freq_maps[freq] = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(freq_maps[freq]), red_cov_select_Bmodes, lmin=2, n_iter=n_iter)[1:,...]

    # red_cov_approx_matrix = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, red_cov_approx_matrix)
    # red_cov_matrix_sample = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, red_cov_matrix_sample)
    # red_cov_approx_matrix[:,:,0] = 0
    # red_cov_matrix_sample[:,:,0] = 0
    # c_ells_input[0,...] = 0
    # c_ells_input[2,...] = 0

# Mixing matrix initialization
init_mixing_matrix_obj = micmac.InitMixingMatrix(np.array(instrument['frequency']), number_components, pos_special_freqs=pos_special_freqs)

init_params = init_mixing_matrix_obj.init_params()



minimum_std_Fisher = scipy.linalg.sqrtm(np.linalg.inv(Fisher_matrix))
minimum_std_Fisher_diag = np.diag(minimum_std_Fisher)


len_pos_special_freqs = len(pos_special_freqs)
step_size_array = np.zeros(((number_frequencies-len_pos_special_freqs)*2))
# step_size_array[:,0] = minimum_std_Fisher_diag[:number_frequencies-len_pos_special_freqs]
# step_size_array[:,1] = minimum_std_Fisher_diag[number_frequencies-len_pos_special_freqs:2*(number_frequencies-len_pos_special_freqs)]
step_size_array[:number_frequencies-len_pos_special_freqs] = minimum_std_Fisher_diag[:number_frequencies-len_pos_special_freqs]
step_size_array[number_frequencies-len_pos_special_freqs:] = minimum_std_Fisher_diag[number_frequencies-len_pos_special_freqs:2*(number_frequencies-len_pos_special_freqs)]
# number_steps_sampler = 20

# params_mixing_matrix_sample =  np.random.uniform(low=init_params-5*step_size_array,high=init_params+5*step_size_array, size=init_params.shape)
params_mixing_matrix_sample = init_params.ravel(order='F') + 10*np.random.uniform(low=-step_size_array,high=step_size_array, size=((number_frequencies-len_pos_special_freqs)*2))

# mixing_matrix_obj = micmac.MixingMatrix(instrument['frequency'], number_components, init_params, pos_special_freqs=pos_special_freqs)
mixing_matrix_obj = micmac.MixingMatrix(instrument['frequency'], number_components, params_mixing_matrix_sample.reshape(((number_frequencies-len_pos_special_freqs),2),order='F'), pos_special_freqs=pos_special_freqs)
mixing_matrix_sampled = mixing_matrix_obj.get_B()
print("Mixing matrix init : ", mixing_matrix_sampled, flush=True)


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
all_r_samples = np.zeros(number_iterations_sampling+1)
all_params_mixing_matrix_samples = np.zeros((number_iterations_sampling+1, number_frequencies-len_pos_special_freqs, number_correlations-1))

# all_CMB_retrieval = np.zeros((number_iterations_sampling+1, nstokes, npix))

# Initial values
c_ell_sampled = np.copy(c_ells_input)
all_cell_samples[0,...] = c_ell_sampled
all_params_mixing_matrix_samples[0,...] = mixing_matrix_obj.params
all_r_samples[0,...] = 10**(-8)


wiener_filter_term = np.zeros((nstokes,npix))
fluctuation_maps = np.zeros((nstokes,npix))

initial_freq_maps = np.copy(freq_maps)

number_steps_sampler_random = np.zeros(number_iterations_sampling, dtype=int)

time_start_sampling = time.time()

number_steps_sampler_Bf = 100
number_steps_sampler_r = 100

kernel_log_proba_Bf = micmac.MetropolisHastings_log(micmac.new_get_conditional_proba_full_likelihood_JAX_from_params, step_size=step_size_array.ravel(order='F'))
mcmc_kernel_log_proba_Bf = numpyro.infer.MCMC(kernel_log_proba_Bf, num_chains=n_walkers, num_warmup=num_warmup, num_samples=number_steps_sampler_Bf)

step_size_r = 10**(-4)
kernel_log_proba_r = micmac.MetropolisHastings_log(micmac.get_conditional_proba_C_from_r, step_size=step_size_r)
mcmc_kernel_log_proba_r = numpyro.infer.MCMC(kernel_log_proba_r, num_chains=n_walkers, num_warmup=num_warmup, num_samples=number_steps_sampler_r)

print(f"Starting iterations for version {file_ver}", flush=True)

# Start sampling !!!
for iteration in range(number_iterations_sampling):
    print("### Start Iteration n°", iteration, flush=True)

    # Application of new mixing matrix
    BtinvNB = micmac.get_inv_BtinvNB(freq_inverse_noise, mixing_matrix_sampled)
    BtinvN_sqrt = micmac.get_BtinvN(np.sqrt(freq_inverse_noise), mixing_matrix_sampled)
    # s_cML = np.einsum('cd,df,fsp->s', BtinvNB, micmac.get_BtinvN(freq_inverse_noise, mixing_matrix_sampled), initial_freq_maps)[0] # Computation of E^t (B^t N^{-1} B)^{-1} B^t N^{-1} d
    s_cML = micmac.get_Wd(freq_inverse_noise, mixing_matrix_sampled, initial_freq_maps, jax_use=False)[0]
    # if only_select_Bmodes:
    #     s_cML = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(s_cML), red_cov_select_Bmodes, lmin=2, n_iter=n_iter)#[1:,...]
    
    # Sampling step 1 : Sample eta term with formulation :
    map_random_x = []
    map_random_y = []
    time_start_sampling_eta_maps = time.time()
    eta_maps_sample = micmac.get_sampling_eta_v2(param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, lmin=lmin, n_iter=n_iter)

    if only_select_Bmodes:
        eta_maps_sample = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(eta_maps_sample), red_cov_select_Bmodes, lmin=lmin, n_iter=n_iter)            

    if no_Emodes_CMB:
        eta_maps_sample_b = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(eta_maps_sample), red_cov_approx_matrix, lmin=lmin, n_iter=n_iter)
        eta_maps_sample = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(eta_maps_sample_b), np.linalg.pinv(red_cov_approx_matrix), lmin=lmin, n_iter=n_iter)

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

    if only_select_Bmodes:
        fluctuation_maps = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(fluctuation_maps), red_cov_select_Bmodes, lmin=2, n_iter=n_iter)#[1:,...]

    # Recording of the samples
    all_s_c_WF_maps[iteration+1,...] = wiener_filter_term
    all_s_c_fluct_maps[iteration+1,...] = fluctuation_maps

    s_c_sample = wiener_filter_term + fluctuation_maps
    
    assert len(wiener_filter_term.shape) == 2
    assert wiener_filter_term.shape[0] == nstokes
    assert wiener_filter_term.shape[1] == npix
    assert len(fluctuation_maps.shape) == 2
    assert fluctuation_maps.shape[0] == nstokes
    assert fluctuation_maps.shape[1] == npix
    assert s_c_sample.shape[0] == nstokes
    assert s_c_sample.shape[1] == npix

    # Sampling step 3 : sampling of CMB covariance C

    
    
    # Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
    c_ells_Wishart = micmac.get_cell_from_map(s_c_sample, lmax=lmax, n_iter=n_iter)
    # c_ells_inv_Wishart_sample = np.zeros((number_correlations, lmax+1)
    # if only_select_Bmodes:
    #     c_ells_Wishart[0,:] = 0
    #     c_ells_Wishart[2,:] = 0

    c_ells_Wishart_modified = np.copy(c_ells_Wishart)
    for i in range(nstokes):
            c_ells_Wishart_modified[i] *= 2*np.arange(lmax+1) + 1
    red_c_ells_Wishart_modified = micmac.get_reduced_matrix_from_c_ell(c_ells_Wishart_modified)[lmin:]


    time_start_sampling_C = time.time()
    # Sampling with Wishart
    # red_cov_matrix_sample = micmac.get_inverse_wishart_sampling_from_c_ells(np.copy(c_ells_Wishart), l_min=lmin)#[lmin:]

    r_all_samples = micmac.get_sample_parameter(mcmc_kernel_log_proba_r, all_r_samples[iteration], random_PRNGKey=jax.random.PRNGKey(99+iteration), lmin=lmin, lmax=lmax, red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
    
    assert len(r_all_samples.shape) == 2
    assert r_all_samples.shape[0] == n_walkers
    assert r_all_samples.shape[1] == number_steps_sampler_r

    r_sample = r_all_samples[0,-1]
    red_cov_matrix_sample = theoretical_red_cov_r0_total + r_sample*theoretical_red_cov_r1_tensor
    print(f"## r sample : {r_sample}",flush=True)

    # red_cov_matrix_sample = micmac.get_inverse_gamma_sampling_only_BB_from_c_ells(np.copy(c_ells_Wishart), l_min=lmin)
    # c_ells_inv_Wishart_sample = micmac.get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)

    time_sampling_C = (time.time()-time_start_sampling_C)/60
    print("##### Sampling C from r at iteration {} in {} minutes".format(iteration+1, time_sampling_C), flush=True)
    # Recording of the samples
    # all_cell_samples[iteration+1,...] = micmac.get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)
    all_cell_samples[iteration+1,:,lmin:] = micmac.get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)
    all_r_samples[iteration+1,...] = r_sample

    if red_cov_matrix_sample.shape[0] == lmax + 1:
        red_cov_matrix_sample = red_cov_matrix_sample[lmin:]
    assert red_cov_matrix_sample.shape[0] == lmax + 1 - lmin
    assert red_cov_matrix_sample.shape[1] == nstokes
    assert red_cov_matrix_sample.shape[2] == nstokes

    # Preparation of sampling step 4
    extended_CMB_maps = np.zeros((number_components, nstokes, npix))
    extended_CMB_maps[0] = s_c_sample
    full_data_without_CMB = initial_freq_maps - np.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
    assert full_data_without_CMB.shape[0] == number_frequencies
    assert full_data_without_CMB.shape[1] == nstokes
    assert full_data_without_CMB.shape[2] == npix

    # Sampling step 4
    time_start_sampling_Bf = time.time()
    # number_steps_sampler_random[iteration] = number_steps_sampler + np.random.randint(0,number_steps_sampler)
    # few_params_mixing_matrix_samples = micmac.get_sample_B_f(micmac.new_get_conditional_proba_full_likelihood_JAX_from_params, step_size_array.ravel(order='F'), number_steps_sampler_random[iteration], first_guess_params_mixing_matrix.ravel(), random_PRNGKey=jax.random.PRNGKey(100+iteration), n_walkers=n_walkers, num_warmup=num_warmup, pos_special_freqs=mixing_matrix_obj.pos_special_freqs, fullsky_ver=fullsky_ver, slow_ver=slow_ver, param_dict=param_dict, full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, freq_inverse_noise=freq_inverse_noise, red_cov_approx_matrix=red_cov_approx_matrix, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG, with_prints=False)
    print("B_f sample :", params_mixing_matrix_sample, flush=True)
    all_numpyro_mixing_matrix_samples = micmac.get_sample_parameter(mcmc_kernel_log_proba_Bf, params_mixing_matrix_sample.ravel(order='F'), random_PRNGKey=jax.random.PRNGKey(100+iteration), pos_special_freqs=mixing_matrix_obj.pos_special_freqs, fullsky_ver=fullsky_ver, slow_ver=slow_ver, param_dict=param_dict, full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, freq_inverse_noise=freq_inverse_noise, red_cov_approx_matrix=red_cov_approx_matrix, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG, with_prints=False)
    params_mixing_matrix_sample = all_numpyro_mixing_matrix_samples[0,-1,:].reshape((number_frequencies-len_pos_special_freqs,2), order='F')
    time_sampling_Bf = (time.time()-time_start_sampling_Bf)/60
    print("##### Sampling B_f at iteration {} in {} minutes".format(iteration+1, time_sampling_Bf), flush=True)

    assert params_mixing_matrix_sample.shape[0] == number_frequencies-len_pos_special_freqs
    assert params_mixing_matrix_sample.shape[1] == 2
    assert len(params_mixing_matrix_sample.shape) == 2
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
param_dict['number_steps_sampler_Bf'] = number_steps_sampler_Bf
param_dict['number_steps_sampler_r'] = number_steps_sampler_r
param_dict['r_true'] = r_true

for key,item in param_dict.items():
    file.write(key + ' = ' + str(item) + '\n')
file.close()

initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_freq_maps_path, flush=True)
np.save(initial_freq_maps_path, initial_freq_maps)

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
