import os, sys, time
import argparse
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
import toml
import numpyro
from functools import partial
import micmac
from fgbuster.observation_helpers import *
# from mcmc_tools import *


from jax import config
sys.path.append(os.path.dirname(os.path.abspath('')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')

config.update("jax_enable_x64", True)

# Parsing additional arguments to prepare run
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('additional_toml', metavar='N', type=str, nargs='+',
                    help='a toml file path to be added to the config')
args = parser.parse_args()
path_additional_params = args.additional_toml[0]

print("Parsing from :", path_additional_params, flush=True)

if path_additional_params[0] != '/':
    path_additional_params = os.path.dirname(os.path.abspath(__file__)) + '/additional_params/' + path_additional_params

print("Effectively parsing from :", path_additional_params, flush=True)

with open(path_additional_params) as f:
    dictionary_additional_parameters = toml.load(f)
f.close()

# Setting up the directory for saving and loading files as mask, etc.
current_repo = dictionary_additional_parameters['current_directory']
MICMAC_repo = dictionary_additional_parameters['MICMAC_directory']
repo_mask = dictionary_additional_parameters['directory_mask']
repo_save = dictionary_additional_parameters['save_directory']

# Getting the rest of the additional arguments
delta_ell = dictionary_additional_parameters['delta_ell']
reduction_noise = dictionary_additional_parameters['reduction_noise']
factor_Fisher = dictionary_additional_parameters['factor_Fisher']
relative_treshold = dictionary_additional_parameters['relative_treshold']
sigma_gap = dictionary_additional_parameters['sigma_gap']
fgs_model = dictionary_additional_parameters['fgs_model']
initial_guess_r_ = dictionary_additional_parameters['initial_guess_r']
use_nhits = dictionary_additional_parameters['use_nhits']
name_mask = dictionary_additional_parameters['name_mask']
use_mask = dictionary_additional_parameters['use_mask']
name_toml = dictionary_additional_parameters['name_toml']
name_file_spv = dictionary_additional_parameters['name_file_spv']
seed_realization_input = dictionary_additional_parameters['seed_realization_input']
use_preconditioner = dictionary_additional_parameters['use_preconditioner']
# use_uncorrelated_patches = dictionary_additional_parameters['use_uncorrelated_patches']

# Starting MPI
from mpi4py import MPI
print("Starting MPI !!!", flush=True)
MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

print("r{} of {} -- Launch".format(MPI_rank, MPI_size), flush=True)

# Checking if continuiation of previous run
former_file_ver = dictionary_additional_parameters['former_file_ver'] 
if former_file_ver != '':
    former_file_ver += f"_{MPI_rank}_{MPI_size}"

# Loading the version of the file
file_ver = dictionary_additional_parameters['file_ver'] + f"_{MPI_rank}_{MPI_size}"


# Defining directories for saving and loading files
path_home_test_playground = MICMAC_repo + '/test_playground/'
if repo_save == '':
    repo_save = path_home_test_playground
directory_save_file = repo_save + current_repo + 'save_directory/'
current_path = path_home_test_playground + current_repo + '/'

path_mask = repo_mask + name_mask + ".fits"

directory_main_params_file = current_path + 'main_params/'
path_toml_file = directory_main_params_file + name_toml
path_file_spv = directory_main_params_file + name_file_spv

# Few tests for directories
if not os.path.exists(directory_save_file):
    raise Exception(f"Directory {directory_save_file} does not exist to save outputs !")
else:
    print(f"Directory {directory_save_file} exists to save outputs !", flush=True)

if not os.path.exists(directory_main_params_file):
    raise Exception(f"Directory {directory_main_params_file} does not exist to load input param files !")

# Creating MICMAC Sampler object
MICMAC_obj = micmac.create_MICMAC_sampler_from_toml_file(path_toml_file, path_file_spv=path_file_spv)

# Preparing the seed
MICMAC_obj.seed = MICMAC_obj.seed + MPI_rank

# General parameters for the foregrounds
# cmb_model = 'c1'
fgs_model_ = fgs_model
# model = cmb_model+fgs_model
# noise_seed = seed_realization_input
instr_name = MICMAC_obj.instrument_name #'SO_SAT'

path_Fisher = path_home_test_playground + f'Fisher_matrix_{MICMAC_obj.instrument_name}_EB_model_{fgs_model_}_noise_True_seed_42_lmin2_lmax128.txt'
try :
    Fisher_matrix = np.loadtxt(path_Fisher)
except:
    print("Fisher matrix not found !", flush=True)
    Fisher_matrix = np.loadtxt(path_home_test_playground + f'Fisher_matrix_{MICMAC_obj.instrument_name}_EB_model_d0s0_noise_True_seed_42_lmin2_lmax128.txt')

# get instrument from public database
instrument = get_instrument(instr_name)

# Apply potential noise reduction
instrument['depth_p'] /= reduction_noise



# Mask initialization

if use_mask:
    apod_mask = hp.ud_grade(hp.read_map(path_mask),nside_out=MICMAC_obj.nside)

    template_mask = np.copy(apod_mask)
    if use_nhits:
        template_mask[template_mask<relative_treshold] = 0
        inverse_nhits_mask = np.copy(template_mask)
        inverse_nhits_mask[template_mask>0] = 1/template_mask[template_mask>0]

        mask = np.copy(template_mask)
        mask[template_mask>0] = 1
        mask[template_mask==0] = 0
    else:
        mask = np.copy(apod_mask)
        mask[apod_mask>0] = 1
        mask[apod_mask==0] = 0
        template_mask = mask
    MICMAC_obj.mask = mask

else:
    # Then the mask have been initialized to 1 in the MICMAC_sampler object
    # mask = np.ones(MICMAC_obj.n_pix)
    template_mask = np.copy(MICMAC_obj.mask)


# Generating foregrounds and noise maps
# np.random.seed(seed_realization_input)
# freq_maps_fgs_noised = get_observation(instrument, fgs_model_, nside=MICMAC_obj.nside, noise=True)[:, 1:, :]   # keep only Q and U
np.random.seed(seed_realization_input)
freq_maps_fgs_denoised = get_observation(instrument, fgs_model_, nside=MICMAC_obj.nside, noise=False)[:, 1:, :]   # keep only Q and U
np.random.seed(seed_realization_input)
noise_map = get_noise_realization(MICMAC_obj.nside, instrument)[:, 1:, :]

# noise_map = freq_maps_fgs_noised - freq_maps_fgs_denoised

# Modifying the noise map with the hits map
if use_nhits:
    print("Using nhits for noise map !", flush=True)
    new_noise_map = noise_map * jnp.sqrt(inverse_nhits_mask)
else:
    new_noise_map = noise_map

freq_maps_fgs = freq_maps_fgs_denoised + new_noise_map

print("Shape for input frequency maps :", freq_maps_fgs.shape)

# Preparing the frequency noise covariance matrix, and masking it if needed
freq_inverse_noise = micmac.get_noise_covar_extended(instrument['depth_p'], MICMAC_obj.nside) #MICMAC_obj.freq_inverse_noise
assert freq_inverse_noise.shape == (MICMAC_obj.n_frequencies, MICMAC_obj.n_frequencies, MICMAC_obj.n_pix)
assert (freq_inverse_noise[:,:,0] == freq_inverse_noise[:,:,100]).all()
# freq_inverse_noise_masked = np.zeros((MICMAC_obj.n_frequencies,MICMAC_obj.n_frequencies,MICMAC_obj.n_pix))

# nb_pixels_mask = int(template_mask.sum())
# nb_pixels_mask = int(len(template_mask[template_mask!=0]))
# freq_inverse_noise_masked[:,:,template_mask!=0] = np.repeat(freq_inverse_noise.ravel(order='F'), nb_pixels_mask).reshape((MICMAC_obj.n_frequencies,MICMAC_obj.n_frequencies,nb_pixels_mask), order='C')


MICMAC_obj.freq_inverse_noise = freq_inverse_noise*template_mask

if use_preconditioner:
    print("Using preconditioner !", flush=True)
    MICMAC_obj.freq_noise_c_ell = micmac.get_true_Cl_noise(np.array(instrument['depth_p']), MICMAC_obj.lmax)

# Generation step-size from the Fisher matrix
try :
    minimum_std_Fisher = scipy.linalg.sqrtm(np.linalg.inv(Fisher_matrix))
except:
    print("Fisher matrix not invertible or scipy.linalg.sqrtm not working on GPU ! Taking only sqrt of diagonal elements instead !", flush=True)
    minimum_std_Fisher = np.sqrt(np.linalg.inv(Fisher_matrix))
minimum_std_Fisher_diag = np.diag(minimum_std_Fisher)

if MICMAC_obj.sample_eta_B_f:
    col_dim_B_f = MICMAC_obj.n_frequencies-len(MICMAC_obj.pos_special_freqs)

    len_pos_special_freqs = len(MICMAC_obj.pos_special_freqs)
    # step_size_B_f = np.zeros((col_dim_B_f,2))
    # step_size_B_f[:,0] = minimum_std_Fisher_diag[:MICMAC_obj.n_frequencies-len_pos_special_freqs]
    # step_size_B_f[:,1] = minimum_std_Fisher_diag[MICMAC_obj.n_frequencies-len_pos_special_freqs:2*(MICMAC_obj.n_frequencies-len_pos_special_freqs)]

    MICMAC_obj.covariance_B_f = np.copy(np.linalg.inv(Fisher_matrix))[:-1,:-1]/factor_Fisher  # TODO: extend to case of many spv params

    # if MICMAC_obj.len_params > (MICMAC_obj.n_frequencies-len_pos_special_freqs)*len(MICMAC_obj.pos_special_freqs):
    #     MICMAC_obj.covariance_B_f = np.repeat(MICMAC_obj.covariance_B_f, 
    #                                           MICMAC_obj.len_params//(MICMAC_obj.n_frequencies-len_pos_special_freqs), 
    #                                           axis=0)

    print("Covariance B_f :", MICMAC_obj.covariance_B_f)

MICMAC_obj.step_size_r = minimum_std_Fisher_diag[-1]

# Generation input maps
np.random.seed(seed_realization_input+1)
input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = MICMAC_obj.generate_input_freq_maps_from_fgs(freq_maps_fgs, return_only_freq_maps=False)

input_freq_maps_masked = input_freq_maps*MICMAC_obj.mask

# Re-Defining the data if needed
indices_polar = np.array([1,2,4])
partial_indices_polar = indices_polar[:MICMAC_obj.nstokes]

# Preparing initial spectra
theoretical_r0_total = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total)#[partial_indices_polar,:]
theoretical_r1_tensor = micmac.get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r1_tensor)#[partial_indices_polar,:]

# Preparing params mixing matrix
init_mixing_matrix_obj = micmac.InitMixingMatrix(freqs=MICMAC_obj.frequency_array, ncomp=MICMAC_obj.n_components, pos_special_freqs=MICMAC_obj.pos_special_freqs, spv_nodes_b=MICMAC_obj.spv_nodes_b)
exact_params_mixing_matrix = init_mixing_matrix_obj.init_params()

# Preparing c_approx
c_ell_approx = np.zeros((3,MICMAC_obj.lmax+1))
c_ell_approx[0,MICMAC_obj.lmin:] = theoretical_r0_total[0,:]
c_ell_approx[1,MICMAC_obj.lmin:] = theoretical_r0_total[1,:]


# First guesses preparation
initial_wiener_filter_term = np.zeros((MICMAC_obj.nstokes, MICMAC_obj.n_pix))
initial_fluctuation_maps = np.zeros((MICMAC_obj.nstokes, MICMAC_obj.n_pix))

len_pos_special_freqs = len(MICMAC_obj.pos_special_freqs)
dimension_free_param_B_f = jnp.size(MICMAC_obj.indexes_free_Bf)
dimension_free_param_B_f = MICMAC_obj.len_params #- len_pos_special_freqs

first_guess = jnp.copy(exact_params_mixing_matrix)

print('MICMAC_obj.indexes_free_Bf', MICMAC_obj.indexes_free_Bf)
print('dimension_free_param_B_f', dimension_free_param_B_f)
print('len_params', MICMAC_obj.len_params)
print(f"First guess from {sigma_gap} $\sigma$ Fisher !", f"rank {MPI_rank} over {MPI_size}", flush=True)
step_size_B_f = minimum_std_Fisher_diag[:-1]

if dimension_free_param_B_f != step_size_B_f.shape[0]:
    expend_factor = dimension_free_param_B_f//step_size_B_f.shape[0]
    step_size_B_f = np.repeat(step_size_B_f, expend_factor)

np.random.seed(MICMAC_obj.seed)
first_guess = first_guess.at[MICMAC_obj.indexes_free_Bf].set(
                first_guess[MICMAC_obj.indexes_free_Bf] + 
                step_size_B_f*np.random.uniform(low=-sigma_gap,high=sigma_gap, size=(dimension_free_param_B_f,)))
init_params_mixing_matrix = first_guess
initial_guess_r = initial_guess_r_ + np.random.uniform(low=-sigma_gap,high=sigma_gap, size=1)*MICMAC_obj.step_size_r

# Redefine initial r if negative
if initial_guess_r < 0:
    initial_guess_r = 1e-8*np.random.uniform(low=-sigma_gap,high=sigma_gap, size=1)

CMB_c_ell = np.zeros_like(c_ell_approx)
# CMB_c_ell[:,MICMAC_obj.lmin:] = (theoretical_r0_total + MICMAC_obj.r_true*theoretical_r1_tensor)
CMB_c_ell[:,MICMAC_obj.lmin:] = (theoretical_r0_total + initial_guess_r*theoretical_r1_tensor)

if MICMAC_obj.sample_C_inv_Wishart and MICMAC_obj.use_binning:
    nb_bin = (MICMAC_obj.lmax-MICMAC_obj.lmin+1)//delta_ell
    MICMAC_obj.bin_ell_distribution = MICMAC_obj.lmin + jnp.arange(nb_bin+1)*delta_ell
    MICMAC_obj.maximum_number_dof = int(MICMAC_obj.bin_ell_distribution[-1]**2 - MICMAC_obj.bin_ell_distribution[-2]**2)


# If continuation of previous run, preparation of the initial guess from previous run
if former_file_ver != '':
    print("### Continuing from previous run !", former_file_ver, f"rank {MPI_rank} over {MPI_size}", flush=True)
    dict_all_params = micmac.loading_params(directory_save_file, former_file_ver, MICMAC_obj)

    init_params_mixing_matrix = dict_all_params['all_params_mixing_matrix_samples'][-1,:]

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


print(f'Exact r : {MICMAC_obj.r_true}')
print(f'Initial r : {initial_guess_r}')

print(f'Exact param matrix : {exact_params_mixing_matrix}')
print(f'Initial param matrix : {init_params_mixing_matrix}')

# Launching the sampling !!!
time_start_sampling = time.time()
MICMAC_obj.perform_Gibbs_sampling(input_freq_maps_masked, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=initial_guess_r, initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
                         theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)

time_full_chain = (time.time()-time_start_sampling)/60
print("End of iterations in {} minutes, saving all files !".format(time_full_chain), f"rank {MPI_rank} over {MPI_size}", flush=True)

# Getting all samples
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

    all_params_mixing_matrix_samples = np.vstack([dict_all_params['all_params_mixing_matrix_samples'], all_params_mixing_matrix_samples[1:]])


# Saving all files
time_saving = time.time()

initial_freq_maps_path = directory_save_file+file_ver+'_initial_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_freq_maps_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
np.save(initial_freq_maps_path, input_freq_maps)

initial_cmb_maps_path = directory_save_file+file_ver+'_initial_cmb_data.npy'
print("FINAL SAVE - #### params_mixing_matrix :", initial_cmb_maps_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
np.save(initial_cmb_maps_path, input_cmb_maps)

if not(MICMAC_obj.cheap_save):
    all_eta_maps_path = directory_save_file+file_ver+'_all_eta_maps.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_eta_maps_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
    np.save(all_eta_maps_path, all_eta)

    all_s_c_WF_maps_path = directory_save_file+file_ver+'_all_s_c_WF_maps.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_WF_maps_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
    np.save(all_s_c_WF_maps_path, all_s_c_WF_maps)

    all_s_c_fluct_maps_path = directory_save_file+file_ver+'_all_s_c_fluct_maps.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_fluct_maps_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
    np.save(all_s_c_fluct_maps_path, all_s_c_fluct_maps)

elif not(MICMAC_obj.very_cheap_save):
    all_s_c_path = directory_save_file+file_ver+'_all_s_c.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_s_c_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
    np.save(all_s_c_path, all_s_c)

if MICMAC_obj.sample_C_inv_Wishart:
    all_cell_samples_path = directory_save_file+file_ver+'_all_cell_samples.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_cell_samples_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
    np.save(all_cell_samples_path, all_cell_samples)
if MICMAC_obj.sample_r_Metropolis:
    all_r_samples_path = directory_save_file+file_ver+'_all_r_samples.npy'
    print("FINAL SAVE - #### params_mixing_matrix :", all_r_samples_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
    np.save(all_r_samples_path, all_r_samples)

all_params_mixing_matrix_samples_path = directory_save_file+file_ver+'_all_params_mixing_matrix_samples.npy'
print("FINAL SAVE - #### params_mixing_matrix :", all_params_mixing_matrix_samples_path, f"rank {MPI_rank} over {MPI_size}", flush=True)
np.save(all_params_mixing_matrix_samples_path, all_params_mixing_matrix_samples)

print("End of saving in {} minutes !".format((time.time()-time_saving)/60), f"rank {MPI_rank} over {MPI_size}", flush=True)