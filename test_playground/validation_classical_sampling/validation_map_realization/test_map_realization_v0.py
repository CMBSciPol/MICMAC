import os, sys, time
import numpy as np
import healpy as hp
import astropy.io.fits as fits
# import ctypes as ct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from func_tools_for_tests import *

from mpi4py import MPI

import non_parametric_ML_compsep as blindcp

rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size
print("Rank {}, Size {} - MPI processes".format(rank, size), flush=True)

nside = 64
lmax = 2*nside
lmin = 2
nstokes = 2

npix = 12*nside**2

# number_iterations_sampling=50
number_iterations_sampling=10

limit_iter_cg=2000
tolerance_PCG=10**(-12)
tolerance_fluctuation = 10**(-12)

n_iter = 8

max_number_correlations = 6
number_correlations = 3 #6 # 4

noise_level = 10**(-7)

mask_binary = np.ones(npix)
mask_name = 'fullsky'

unit_variance_maps = 1/hp.nside2resol(nside)

# file_ver = 'PolarClassicalGibbsMapv1a' # Map realization ; only polar
# file_ver = 'PolarClassicalGibbsMapv1b' # Map realization ; only polar ; short chain (10 samples)
# file_ver = 'PolarClassicalGibbsMapv1c' # Map realization ; only polar ; short chain (10 samples) ; initial guess = 0
# file_ver = 'PolarClassicalGibbsMapv1d' # Map realization ; only polar ; short chain (10 samples) ; initial guess = 0 ; unit_variance_maps = 1
# file_ver = 'PolarClassicalGibbsMapv1e' # Map realization ; only polar ; short chain (10 samples) 
#### ABOVE EVERYTHING IS WRONG !!!!
file_ver = 'PolarClassicalGibbsMapv2a' # Map realization ; only polar ; short chain (10 samples) 
file_ver = 'PolarClassicalGibbsMapv2b' # Map realization ; only polar ; short chain (10 samples) ; test CG fluct

# dir_path = '/pscratch/sd/m/mag/WF_work/cl_sampling_test/'
dir_path = '/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/data_files/run_validation_classical_sampling/run_validation_full_chain/'


path_map_CMB = dir_path + "Map_ver{}_only_CMB_{}.fits".format(file_ver, lmax)
path_map_wn = dir_path + "Map_ver{}_only_wn_{}.fits".format(file_ver, lmax)
path_map_total = dir_path + "Map_total_{}_CMB_plus_white_noise_{}.fits".format(file_ver, lmax)



camb_cls = generate_power_spectra_CAMB(max(lmax,nside*2))[:lmax+1,:]
outname_c_ell = 'CAMB_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
print('Recording c_ell CAMB in :', dir_path, outname_c_ell, flush=True)
np.save(dir_path+outname_c_ell, camb_cls.T)

CMB_map_input = hp.synfast(camb_cls.T, nside, new=True, lmax=lmax)

c_ells_noise_2 = np.zeros((max_number_correlations, lmax+1))
c_ells_noise_2[:3,2:] = noise_level

map_wn_2 = hp.synfast(c_ells_noise_2, nside, new=True, lmax=lmax)
map_total = CMB_map_input + map_wn_2

print("Map input recorded in", path_map_CMB, path_map_wn, path_map_total, flush=True)
hp.write_map(path_map_total, map_total, overwrite=True)
hp.write_map(path_map_CMB, CMB_map_input, overwrite=True)
hp.write_map(path_map_wn, map_wn_2, overwrite=True)


param_dict = {'nside':nside, 'lmax':lmax, 'nstokes':nstokes}
param_dict["number_correlations"] = number_correlations

# c_ells_input = hp.anafast(CMB_map_input, lmax=lmax, iter=n_iter)
# c_ells_input[:,:2] = 0
# c_ells_input = np.zeros((6, lmax+1))
# c_ells_input[:4,] = camb_cls.T
c_ells_input = np.zeros((6, lmax+1))
ell_arange = np.arange(lmin, lmax+1)
c_ells_input[:3,lmin:] = 10**(-4)/(ell_arange*(ell_arange+1))
c_ells_input[:3,:lmin] = 10**(-30)

if nstokes == 2:
    indices_polar = np.array([1,2,4])
    c_ells_noise_2 = c_ells_noise_2[indices_polar,:]
    c_ells_input = c_ells_input[indices_polar,:]
    CMB_map_input = CMB_map_input[1:,:]
    map_total = map_total[1:,:]

print("c_ells_input", c_ells_input[:,:10])

Gibbs_sampler = blindcp.Gibbs_Sampling(nside, lmax, nstokes, lmin=lmin, number_iterations_sampling=number_iterations_sampling, limit_iter_cg=limit_iter_cg, tolerance_PCG=tolerance_PCG, tolerance_fluctuation=tolerance_fluctuation, n_iter=n_iter)

red_inverse_noise = blindcp.get_inverse_reduced_matrix_from_c_ell(c_ells_noise_2, lmin=lmin)

print("Start sampling !", flush=True)
t0 = time.time()

# Preparation of the initial guess
c_ell_sampled = np.copy(c_ells_input)
# Preparation of the input map data
pixel_maps_sampled = np.copy(map_total)

if nstokes != 1:
    assert map_total.shape[0] == nstokes

all_maps = np.zeros((number_iterations_sampling+1, nstokes, npix))
all_samples = np.zeros((number_iterations_sampling+1, number_correlations, lmax+1))

all_fluctuations = np.zeros_like(all_maps)
all_WF = np.zeros_like(all_maps)

all_analytical_fluct = np.zeros((2,number_iterations_sampling, nstokes, npix))
all_analytical_WF = np.zeros((2,number_iterations_sampling, nstokes, npix))

real_fluctuation_variance = np.zeros((number_iterations_sampling+1, number_correlations, lmax+1-lmin))
variance_fluct = np.linalg.pinv(np.linalg.pinv(blindcp.get_reduced_matrix_from_c_ell(c_ell_sampled)[lmin:,...]) + red_inverse_noise)
real_fluctuation_variance[0, ...] = blindcp.get_c_ells_from_red_covariance_matrix(variance_fluct)

all_maps[0,...] = pixel_maps_sampled
all_samples[0,...] = c_ell_sampled

if red_inverse_noise.shape[0] == lmax+1:
    red_inverse_noise = red_inverse_noise[lmin:,...]
if nstokes != 1:
    assert map_total.shape[0] == nstokes

for iteration in range(number_iterations_sampling):
    print("### Start Iteration n°", iteration, flush=True)

    red_covariance_matrix = blindcp.get_reduced_matrix_from_c_ell(c_ell_sampled)[lmin:,...]

    assert len(red_covariance_matrix[np.isnan(red_covariance_matrix)]) == 0

    # Constrained map realization step
    # pixel_maps_sampled = Gibbs_sampler.constrained_map_realization(map_total, red_covariance_matrix, red_inverse_noise, initial_guess=np.copy(pixel_maps_sampled))
    assert red_covariance_matrix.shape[0] == lmax + 1 - lmin
    assert red_inverse_noise.shape[0] == lmax + 1 - lmin
    # initial_guess = np.copy(pixel_maps_sampled)
    # initial_guess = np.zeros_like(pixel_maps_sampled)
    if iteration > 0 :
        initial_guess_fluct = np.copy(all_fluctuations[iteration-1])
        initial_guess_WF = np.copy(all_fluctuations[iteration-1])
    else:
        initial_guess_fluct = np.copy(pixel_maps_sampled)
        initial_guess_WF = np.copy(pixel_maps_sampled)
    
    map_white_noise_xi = np.random.normal(loc=0, scale=unit_variance_maps, size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    map_white_noise_chi = np.random.normal(loc=0, scale=unit_variance_maps, size=(param_dict["nstokes"],12*param_dict["nside"]**2))

    # analytical_WF_0, analytical_WF_1 = get_all_analytical_WF_term_maps_v2(param_dict, map_total, red_covariance_matrix, red_inverse_noise, lmin=lmin, n_iter=n_iter)
    # analytical_Fluct_0, analytical_Fluct_1 = get_all_analytical_fluctuating_term_maps_v2(param_dict, np.linalg.pinv(red_covariance_matrix), red_inverse_noise, map_white_noise_xi, map_white_noise_chi, lmin=lmin, n_iter=n_iter)
    analytical_WF_0, analytical_WF_1 = get_all_analytical_WF_term_maps_v2(param_dict, map_total, red_covariance_matrix, red_inverse_noise, lmin=lmin, n_iter=n_iter)
    analytical_Fluct_0, analytical_Fluct_1 = get_all_analytical_fluctuating_term_maps_v3sqrt(param_dict, np.linalg.pinv(red_covariance_matrix), red_inverse_noise, map_white_noise_xi, map_white_noise_chi, lmin=lmin, n_iter=n_iter)

    all_analytical_fluct[0,iteration,...] = analytical_Fluct_0
    all_analytical_fluct[1,iteration,...] = analytical_Fluct_1
    all_analytical_WF[0,iteration,...] = analytical_WF_0
    all_analytical_WF[1,iteration,...] = analytical_WF_1

    fluctuating_map = blindcp.get_fluctuating_term_maps(param_dict, red_covariance_matrix, red_inverse_noise, map_white_noise_xi, map_white_noise_chi, initial_guess=initial_guess_fluct, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_fluctuation)
    wiener_filter_term = blindcp.solve_generalized_wiener_filter_term(param_dict, map_total, red_covariance_matrix, red_inverse_noise, initial_guess=initial_guess_WF, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_PCG)
    pixel_maps_sampled = fluctuating_map + wiener_filter_term

    all_fluctuations[iteration,...] = fluctuating_map
    all_WF[iteration,...] = wiener_filter_term

    # C_ell sampling step
    red_cov_mat_sampled = Gibbs_sampler.sample_covariance(pixel_maps_sampled)

    # Few tests to verify everything's fine
    all_eigenvalues = np.linalg.eigh(red_cov_mat_sampled[lmin:])[0]
    assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_cov_mat_sampled[lmin:])[0])>10**(-15)]>0)

    # Preparation of next step
    c_ell_sampled = blindcp.get_c_ells_from_red_covariance_matrix(red_cov_mat_sampled)
    
    variance_fluct = np.linalg.pinv(np.linalg.pinv(red_covariance_matrix) + red_inverse_noise)
    real_fluctuation_variance[iteration+1, ...] = blindcp.get_c_ells_from_red_covariance_matrix(variance_fluct)
    # print("### Number nan after inv Wishart :", len(c_ell_sampled[np.isnan(c_ell_sampled)]))
    # Recording of the samples
    
    all_maps[iteration+1,...] = pixel_maps_sampled
    all_samples[iteration+1,...] = c_ell_sampled

    if iteration%50 == 0:
        print("### Iteration n°", iteration, flush=True)

sampled_maps, sampled_cells =  all_maps, all_samples
print("Finish Gibbs sampling in :", (time.time()-t0)/60., "minutes", flush=True)


if rank == 0:
    outname_map = 'All_map_fluct_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording map in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_map, all_fluctuations)

    outname_map = 'All_map_WF_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording map in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_map, all_WF)


    outname_map = 'All_analytical_map_fluct_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording map in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_map, all_analytical_fluct)

    outname_map = 'All_analytical_map_WF_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording map in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_map, all_analytical_WF)

    outname_map = 'Map_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording map in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_map, sampled_maps)
    
    
    outname_map = 'Map_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording map in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_map, sampled_maps)

    outname_c_ell = 'Cell_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording c_ell in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_c_ell, sampled_cells)


    outname_c_ell = 'Variance_fluctuation_{}_ver{}_nside{:3d}_{}.npy'.format(mask_name, file_ver, nside, lmax)
    print('Recording c_ell in :', dir_path, outname_map, flush=True)
    np.save(dir_path+outname_c_ell, real_fluctuation_variance)
