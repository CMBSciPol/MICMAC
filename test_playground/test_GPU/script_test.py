import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from jax import random, dtypes
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
import scipy
import healpy as hp
import astropy.io.fits as fits
import camb
import numpyro.distributions as dist
import lineax as lx

import micmac as micmac

current_path = os.path.dirname(os.path.abspath('')) + '/'
sys.path.append(current_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('')))+'/tutorials/')
from func_tools_for_tests import *
from get_freq_maps_SO_64 import *

PRNGKey = random.PRNGKey(0)

repo_mask = "/gpfswork/rech/nih/commun/masks/"
path_mask = repo_mask + "mask_SO_SAT_apodized.fits"

# working_directory_path = os.path.abspath('') + '/'
working_directory_path = current_path + '/validation_chain_v7_JZ/'
directory_toml_file = working_directory_path + 'toml_params/'

path_toml_file = directory_toml_file + 'corr_v1cbcb.toml'

MICMAC_sampler_obj = micmac.create_MICMAC_sampler_from_toml_file(path_toml_file)

apod_mask = hp.ud_grade(hp.read_map(path_mask),nside_out=MICMAC_sampler_obj.nside)

mask = np.copy(apod_mask)
mask[apod_mask>0] = 1
mask[apod_mask==0] = 0

# mask = np.ones_like(apod_mask)

MICMAC_sampler_obj.mask = mask



freq_inverse_noise_masked = np.zeros((MICMAC_sampler_obj.number_frequencies,MICMAC_sampler_obj.number_frequencies,MICMAC_sampler_obj.npix))
freq_inverse_noise_0 = micmac.get_noise_covar(instrument['depth_p'], MICMAC_sampler_obj.nside) #MICMAC_sampler_obj.freq_inverse_noise

nb_pixels_mask = int(mask.sum())
freq_inverse_noise_masked[:,:,mask!=0] = np.repeat(freq_inverse_noise_0.ravel(order='F'), nb_pixels_mask).reshape((MICMAC_sampler_obj.number_frequencies,MICMAC_sampler_obj.number_frequencies,nb_pixels_mask), order='C')

freq_inverse_noise_masked = freq_inverse_noise_masked*mask
# freq_inverse_noise_masked = freq_inverse_noise_masked*nhits_mask

MICMAC_sampler_obj.freq_inverse_noise = freq_inverse_noise_masked
freq_inverse_noise = freq_inverse_noise_masked


np.random.seed(MICMAC_sampler_obj.seed)
freq_maps_fgs = get_observation(instrument, fgs_model, nside=MICMAC_sampler_obj.nside, noise=False)[:, 1:, :]   # keep only Q and U

init_mixing_matrix_obj = micmac.InitMixingMatrix(np.array(instrument['frequency']), MICMAC_sampler_obj.number_components, pos_special_freqs=MICMAC_sampler_obj.pos_special_freqs)
init_params = init_mixing_matrix_obj.init_params()

mixing_matrix_obj = micmac.MixingMatrix(instrument['frequency'], MICMAC_sampler_obj.number_components, init_params, pos_special_freqs=MICMAC_sampler_obj.pos_special_freqs)

mixing_matrix_sampled = mixing_matrix_obj.get_B()

BtinvNB = micmac.get_inv_BtinvNB(freq_inverse_noise, mixing_matrix_sampled)#*mask
# BtinvN_sqrt = micmac.get_BtinvN(scipy.linalg.sqrtm(freq_inverse_noise), mixing_matrix_sampled)#*mask
BtinvN_sqrt = micmac.get_BtinvN(np.sqrt(freq_inverse_noise), mixing_matrix_sampled)#*mask

BtinvN = micmac.get_BtinvN(freq_inverse_noise, mixing_matrix_sampled)#*mask

nb_pix_mask = int(np.sum(mask))




input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = MICMAC_sampler_obj.generate_input_freq_maps_from_fgs(freq_maps_fgs, return_only_freq_maps=False)

input_freq_maps_masked = input_freq_maps*mask
input_cmb_maps = input_cmb_maps*mask



red_cov_matrix_sample = theoretical_red_cov_r0_total + MICMAC_sampler_obj.r_true * theoretical_red_cov_r1_tensor


number_frequencies = MICMAC_sampler_obj.number_frequencies
number_components = MICMAC_sampler_obj.number_components
nstokes = MICMAC_sampler_obj.nstokes
lmin = MICMAC_sampler_obj.lmin
lmax = MICMAC_sampler_obj.lmax
nside = MICMAC_sampler_obj.nside
n_iter = MICMAC_sampler_obj.n_iter
npix = 12*nside**2



MICMAC_sampler_obj.limit_iter_cg = 1500
MICMAC_sampler_obj.tolerance_CG = 1e-6
MICMAC_sampler_obj.atol_CG = 1e-8


N_c_inv = jnp.copy(BtinvNB[0,0])
N_c_inv = N_c_inv.at[...,MICMAC_sampler_obj.mask!=0].set(1/BtinvNB[0,0,MICMAC_sampler_obj.mask!=0]/jhp.nside2resol(MICMAC_sampler_obj.nside)**2)
N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), MICMAC_sampler_obj.nstokes).reshape((MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix), order='F').ravel()

def second_term_left(x):
    return x*N_c_inv_repeat

red_cov_matrix_sqrt = micmac.get_sqrt_reduced_matrix_from_matrix_jax(red_cov_matrix_sample)
first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix)), red_cov_matrix_sqrt, nside=MICMAC_sampler_obj.nside, lmin=MICMAC_sampler_obj.lmin, n_iter=MICMAC_sampler_obj.n_iter).ravel()

# func = lambda x: maps_x_red_covariance_cell_JAX(x.reshape((MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix)), red_cov_matrix_sample, nside=MICMAC_sampler_obj.nside, lmin=MICMAC_sampler_obj.lmin, n_iter=MICMAC_sampler_obj.n_iter).ravel()
func = lambda x: x.ravel() + first_part_term_left(second_term_left(first_part_term_left(x)))
# func_left_term_tree_map = lambda x : jax.tree_map(func, x)
func_norm = lambda x : jnp.linalg.norm(x,ord=2)


func_lineax_test = lx.FunctionLinearOperator(func, jax.ShapeDtypeStruct((MICMAC_sampler_obj.nstokes*MICMAC_sampler_obj.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))


jax_key_PNRG, jax_key_PNRG_xi = random.split(PRNGKey+10) # Splitting of the random key to generate a new one

map_random_realization_xi = jax.random.normal(jax_key_PNRG_xi, shape=(MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix))/jhp.nside2resol(MICMAC_sampler_obj.nside)#*mask_to_use

jax_key_PNRG, *jax_key_PNRG_chi = random.split(jax_key_PNRG,MICMAC_sampler_obj.number_frequencies+1) # Splitting of the random key to generate a new one
def fmap(random_key):
    random_map = jax.random.normal(random_key, shape=(MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix))#/jhp.nside2resol(nside)
    return MICMAC_sampler_obj.get_band_limited_maps(random_map)
map_random_realization_chi = jax.vmap(fmap)(jnp.array(jax_key_PNRG_chi))

right_member_1 = map_random_realization_xi

## Computation of N_c^{-1/2} = (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi
right_member_2_part = jnp.einsum('kcp,cfp,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]*N_c_inv # [0] for selecting CMB component of the random variable
right_member_2 = maps_x_red_covariance_cell_JAX(right_member_2_part, red_cov_matrix_sqrt, nside=MICMAC_sampler_obj.nside, lmin=MICMAC_sampler_obj.lmin, n_iter=MICMAC_sampler_obj.n_iter)

right_member_b = (right_member_1 + right_member_2)#.ravel()
# options_dict = {'preconditoner':attempt_precond_lineax}
right_member_b = MICMAC_sampler_obj.get_band_limited_maps(right_member_b)


GMRES_solver = lx.GMRES(rtol=MICMAC_sampler_obj.tolerance_CG, atol=MICMAC_sampler_obj.atol_CG, max_steps=MICMAC_sampler_obj.limit_iter_cg, norm=func_norm)
CG_solver = lx.CG(rtol=MICMAC_sampler_obj.tolerance_CG, atol=MICMAC_sampler_obj.atol_CG, max_steps=MICMAC_sampler_obj.limit_iter_cg, norm=func_norm)

GMRES_solver = lx.GMRES(rtol=MICMAC_sampler_obj.tolerance_CG, atol=MICMAC_sampler_obj.atol_CG, max_steps=MICMAC_sampler_obj.limit_iter_cg, norm=func_norm, restart=50, stagnation_iters=5)

t0_CG = time.time()
solution_CG = lx.linear_solve(func_lineax_test, right_member_b.ravel(), solver=CG_solver, throw=False)
t1_CG = time.time()
print('CG stats :', solution_CG.stats, flush=True)
print('CG time :', t1_CG-t0_CG, flush=True)

t0_GMRES = time.time()
solution_GMRES = lx.linear_solve(func_lineax_test, right_member_b.ravel(), solver=GMRES_solver, throw=False)
t1_GMRES = time.time()
print('GMRES stats :', solution_GMRES.stats, flush=True)
print('GMRES time :', t1_GMRES-t0_GMRES, flush=True)

solution_lineax_CG = solution_CG.value.reshape((MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix))
solution_lineax_GMRES = solution_GMRES.value.reshape((MICMAC_sampler_obj.nstokes,MICMAC_sampler_obj.npix))

print("Difference -> max :", jnp.abs(solution_lineax_CG-solution_lineax_GMRES).max(), "mean :", (solution_lineax_CG-solution_lineax_GMRES).mean(), flush=True)
