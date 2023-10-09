import os, sys, time
import numpy as np
import healpy as hp
import astropy.io.fits as fits
# import ctypes as ct
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func_tools_for_tests import *

# from mpi4py import MPI

import micmac as blindcp

# rank = MPI.COMM_WORLD.rank
# size = MPI.COMM_WORLD.size
# print("Rank {}, Size {} - MPI processes".format(rank, size), flush=True)

nside = 64
lmax = 2*nside
lmin = 2
nstokes = 2

number_frequencies = 5
number_components = 3

noise_level = np.sqrt(10**(-7) * hp.nside2resol(nside, arcmin=True))
depth_p = np.array([noise_level]*number_frequencies)

init_params_mixing_matrix = np.array([[2, 3], [2, 3], [2, 3]])   # (nfreq-ncomps+1)*(ncomp-1)
pos_special_freqs = np.array([0, 1])   #[-2, -1])

npix = 12*nside**2

number_iterations_sampling=50

limit_iter_cg=2000
tolerance_CG=10**(-12)

n_iter = 8

max_number_correlations = 6
number_correlations = 3 #6 # 4

limit_steps_sampler_mixing_matrix = 1000
number_walkers=1

mask_binary = np.ones(npix)
mask_name = 'fullsky'


file_ver = 'PolarNonParamLKLDFullChainv0a' # Full chain sampling only polar -> with former notes

# dir_path = '/pscratch/sd/m/mag/WF_work/cl_sampling_test/'
dir_path = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/data_files/run_validation_chain_Pixel_Non_Param_CP_v0/'


path_map_CMB = dir_path + "Map_ver{}_only_CMB_{}.fits".format(file_ver, lmax)
path_map_wn = dir_path + "Map_ver{}_only_wn_{}.fits".format(file_ver, lmax)
path_map_total = dir_path + "Map_total_{}_CMB_plus_white_noise_{}.fits".format(file_ver, lmax)



all_spectra = generate_power_spectra_CAMB(nside*2, typeless_bool=True)
lensing_spectra = all_spectra['lensed_scalar'][:lmax+1,:]
camb_cls = all_spectra['total'][:lmax+1,:]
outname_c_ell = 'CAMB_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
print('Recording c_ell CAMB in :', dir_path, outname_c_ell, flush=True)
np.save(dir_path+outname_c_ell, camb_cls.T)

CMB_map_input = hp.synfast(camb_cls.T, nside, new=True, lmax=lmax)

c_ells_noise_2 = np.zeros((max_number_correlations, lmax+1))
c_ells_noise_2[:3,2:] = noise_level

map_wn_2 = hp.synfast(c_ells_noise_2, nside, new=True, lmax=lmax)
map_total = CMB_map_input + map_wn_2

white_noise_maps_out_CMB = np.array([hp.synfast(c_ells_noise_2, nside, new=True, lmax=lmax) for i in range(number_frequencies-1)])

print("Map input recorded in", path_map_CMB, path_map_wn, path_map_total, flush=True)
hp.write_map(path_map_total, map_total, overwrite=True)
hp.write_map(path_map_CMB, CMB_map_input, overwrite=True)
hp.write_map(path_map_wn, map_wn_2, overwrite=True)


# param_dict = {'nside':nside, 'lmax':lmax, 'nstokes':nstokes}
# param_dict["number_correlations"] = number_correlations

# c_ells_input = hp.anafast(CMB_map_input, lmax=lmax, iter=n_iter)
# c_ells_input[:,:2] = 0
# c_ells_input = np.zeros((6, lmax+1))
# c_ells_input[:4,] = camb_cls.T
c_ells_input = np.zeros((6, lmax+1))
ell_arange = np.arange(lmin, lmax+1)
c_ells_input[:3,lmin:] = 10**(-4)/(ell_arange*(ell_arange+1))
c_ells_input[:3,:lmin] = 10**(-30)

c_ell_approx = np.zeros((6,lmax+1))
c_ell_approx[1] = camb_cls.T[1]
c_ell_approx[2] = lensing_spectra.T[2]


initial_freq_maps = np.zeros((number_frequencies, 3, npix))

initial_freq_maps[:int(number_frequencies/2)] = white_noise_maps_out_CMB[:int(number_frequencies/2)]
initial_freq_maps[int(number_frequencies/2)] = map_total
initial_freq_maps[int(number_frequencies/2)+1:] = white_noise_maps_out_CMB[int(number_frequencies/2)+1:]

if nstokes == 2:
    indices_polar = np.array([1,2,4])
    c_ells_noise_2 = c_ells_noise_2[indices_polar,:]
    c_ells_input = c_ells_input[indices_polar,:]
    c_ell_approx = c_ell_approx[indices_polar,:]
    CMB_map_input = CMB_map_input[1:,:]
    map_total = map_total[1:,:]
    initial_freq_maps = initial_freq_maps[:,1:,:]

tot_cov_first_guess = c_ells_input


print("c_ells_input", c_ells_input[:,:10])

SemiBlindLKLD_sampler = blindcp.Non_parametric_Likelihood_Sampling(nside, lmax, nstokes, number_frequencies, number_components, lmin=lmin, number_iterations_sampling=number_iterations_sampling, limit_iter_cg=limit_iter_cg, tolerance_CG=tolerance_CG, n_iter=n_iter, number_walkers=number_walkers, limit_steps_sampler_mixing_matrix=limit_steps_sampler_mixing_matrix)

# red_inverse_noise = blindcp.get_inverse_reduced_matrix_from_c_ell(c_ells_noise_2, lmin=lmin)

print("Start sampling !", flush=True)
t0 = time.time()

# Parameters initialization
freq_inverse_noise = blindcp.get_noise_covar(depth_p, nside)
red_cov_approx_matrix = blindcp.get_reduced_matrix_from_c_ell(c_ell_approx)[lmin:,...]


# Initialization of CG maps to 0 for first iteration
eta_maps = np.zeros((nstokes, npix)) # copy initial map better ?

# Preparation of the initial guess
c_ell_sampled = np.copy(tot_cov_first_guess)

# Preparation of the input map data
pixel_maps_sampled = np.copy(map_total)

# Preparation of covariance matrix
red_covariance_matrix_sampled = blindcp.get_reduced_matrix_from_c_ell(c_ell_sampled)[lmin:,...]

# Preparation of the mixing matrix object
sample_mixing_matrix_full = blindcp.MixingMatrix(number_frequencies, number_components, init_params_mixing_matrix, pos_special_freqs)
fg_params_mixing_matrix_sampled = sample_mixing_matrix_full.params



param_dict = {'nside':nside, 'lmax':lmax, 'nstokes':nstokes, 'number_correlations':number_correlations,'number_frequencies':number_frequencies, 'number_components':number_components}


all_eta = np.zeros((number_iterations_sampling+1, nstokes, npix))
all_maps = np.zeros((number_iterations_sampling+1, nstokes, npix))
all_cell_samples = np.zeros((number_iterations_sampling+1, number_correlations, lmax+1))
all_mixing_matrix_samples = np.zeros((number_iterations_sampling+1, fg_params_mixing_matrix_sampled.shape[0], fg_params_mixing_matrix_sampled.shape[1]))

# Initial values
all_maps[0,...] = pixel_maps_sampled
all_cell_samples[0,...] = c_ell_sampled
all_mixing_matrix_samples[0,...] = fg_params_mixing_matrix_sampled

if nstokes != 1:
        assert initial_freq_maps.shape[0] == number_frequencies
        assert initial_freq_maps.shape[1] == nstokes
        assert initial_freq_maps.shape[2] == npix

assert freq_inverse_noise.shape[0] == number_frequencies
assert freq_inverse_noise.shape[1] == number_frequencies



for iteration in range(number_iterations_sampling):
    print("### Start Iteration n°", iteration, flush=True)

    mixing_matrix_sampled = sample_mixing_matrix_full.get_B()
    # Application of new mixing matrix
    cp_cp_noise = blindcp.get_inv_BtinvNB(freq_inverse_noise, mixing_matrix_sampled)
    cp_freq_inv_noise_sqrt = blindcp.get_BtinvN(np.sqrt(freq_inverse_noise), mixing_matrix_sampled)
    ML_initial_data_maps = np.einsum('cd,df,fsp->s', cp_cp_noise, blindcp.get_BtinvN(freq_inverse_noise, mixing_matrix_sampled), initial_freq_maps)[0] # Computation of E^t (B^t N^{-1} B)^{-1} B^t N^{-1} d


    assert eta_maps.shape[0] == nstokes
    assert eta_maps.shape[1] == npix
    
    assert red_covariance_matrix_sampled.shape[0] == lmax + 1 - lmin
    assert mixing_matrix_sampled.shape[0] == number_frequencies
    assert mixing_matrix_sampled.shape[1] == number_components

    
    # Sampling step 1 : Solve CG for eta term with formulation : (S_approx + mixed_noise) eta = S_approx^(-1/2) x + mixed_noise noise^(1/2) y
    map_random_x = []
    map_random_y = []
    eta_maps = blindcp.get_sampling_eta(param_dict, red_cov_approx_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, initial_guess=np.copy(eta_maps), lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG)

    # Sampling step 2 : sampling of Gaussian variable s_c with mean ML_initial_data_maps and variance (S_c + E^t (B^t N^{-1} B)^{-1} E)
    map_random_xi = []
    map_random_chi = []
    pixel_maps_sampled = blindcp.get_gaussian_sample_maps(param_dict, ML_initial_data_maps, red_covariance_matrix_sampled, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_xi=map_random_xi, map_random_realization_chi=map_random_chi, lmin=lmin, n_iter=n_iter)
    
    assert pixel_maps_sampled.shape[0] == nstokes
    assert pixel_maps_sampled.shape[1] == npix
    # Application of new gaussian maps

    extended_CMB_maps = np.zeros((number_components, nstokes, npix))
    extended_CMB_maps[0] = pixel_maps_sampled
    full_data_without_CMB = initial_freq_maps - np.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
    assert full_data_without_CMB.shape[0] == number_frequencies
    assert full_data_without_CMB.shape[1] == nstokes
    assert full_data_without_CMB.shape[2] == npix
    
    # Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
    red_covariance_matrix_sampled = SemiBlindLKLD_sampler.sample_covariance(pixel_maps_sampled)

    # Sampling step 4
    sample_params_mixing_matrix_FG = blindcp.sample_mixing_matrix_term(param_dict, sample_mixing_matrix_full, full_data_without_CMB, eta_maps, red_cov_approx_matrix, freq_inverse_noise, lmin=lmin, n_iter=n_iter, n_walkers=number_walkers, number_steps_sampler=limit_steps_sampler_mixing_matrix)

    sample_mixing_matrix_full.update_params(sample_params_mixing_matrix_FG)
    # Few tests to verify everything's fine
    all_eigenvalues = np.linalg.eigh(red_covariance_matrix_sampled[lmin:])[0]
    assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_covariance_matrix_sampled[lmin:])[0])>10**(-15)]>0)

    
    # Recording of the samples
    all_eta[iteration+1,...] = eta_maps
    all_maps[iteration+1,...] = pixel_maps_sampled
    all_cell_samples[iteration+1,...] = blindcp.get_c_ells_from_red_covariance_matrix(red_covariance_matrix_sampled)
    all_mixing_matrix_samples[iteration+1,...] = sample_params_mixing_matrix_FG

    if iteration%50 == 0:
        print("### Iteration n°", iteration, flush=True)


all_eta, all_maps, all_cell_samples, all_mixing_matrix_samples


print("Finish Gibbs sampling in :", (time.time()-t0)/60., "minutes", flush=True)

outname_map = 'All_Maps_Eta_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
print('Recording map eta in :', dir_path, outname_map, flush=True)
np.save(dir_path+outname_map, all_eta)

outname_map = 'All_Maps_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
print('Recording map in :', dir_path, outname_map, flush=True)
np.save(dir_path+outname_map, all_maps)

outname_c_ell = 'All_Cells_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
print('Recording c_ell in :', dir_path, outname_map, flush=True)
np.save(dir_path+outname_c_ell, all_cell_samples)

outname_mixing_matrix = 'All_MixingMat_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
print('Recording mixing matrices in :', dir_path, outname_map, flush=True)
np.save(dir_path+outname_c_ell, all_mixing_matrix_samples)