import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jlax
import healpy as hp
from functools import partial

def get_reduced_matrix_from_c_ell_jax(c_ells_input):
    """ Expect c_ells_input to be sorted as TT, EE, BB, TE, TB, EB if 6 spectra are given

        Generate covariance matrix from c_ells assuming it's block diagonal

        -> TO RETEST ?
    """
    c_ells_array = jnp.copy(c_ells_input)
    number_correlations = c_ells_array.shape[0]
    # assert number_correlations == 1 or number_correlations == 3 or number_correlations == 6
    lmax_p1 = c_ells_array.shape[1]
    if number_correlations == 1:
        nstokes = 1
    elif number_correlations == 3:
        nstokes = 2
        # c_ells_array = np.vstack((c_ells_array, np.zeros(lmax)))
        # number_correlations = 3
    # elif number_correlations > 3:
    elif number_correlations == 4 or number_correlations == 6 :
        nstokes = 3
        if number_correlations != 6:
            # for i in range(6 - number_correlations):
                # c_ells_array = jnp.vstack((c_ells_array, jnp.zeros(lmax_p1)))
            c_ells_array = jnp.vstack((c_ells_array, jnp.repeat(jnp.zeros(lmax_p1), 6 - number_correlations)))
            number_correlations = 6
    else :
        raise Exception("C_ells must be given as TT for temperature only ; EE, BB, EB for polarization only ; TT, EE, BB, TE, (TB, EB) for both temperature and polarization")

    reduced_matrix = jnp.zeros((lmax_p1,nstokes,nstokes))

    for i in range(nstokes):
        # reduced_matrix[:,i,i] =  c_ells_array[i,:]
        reduced_matrix = reduced_matrix.at[:,i,i].set(c_ells_array[i,:])
    
    # for j in range(number_correlations-nstokes):
    if number_correlations > 1:
        # reduced_matrix[:,0,1] =  c_ells_array[nstokes,:]
        # reduced_matrix[:,1,0] =  c_ells_array[nstokes,:]
        reduced_matrix = reduced_matrix.at[:,0,1].set(c_ells_array[nstokes,:])
        reduced_matrix = reduced_matrix.at[:,1,0].set(c_ells_array[nstokes,:])
    if number_correlations == 6:
        # reduced_matrix[:,0,2] =  c_ells_array[4,:]
        # reduced_matrix[:,2,0] =  c_ells_array[4,:]

        # reduced_matrix[:,1,2] =  c_ells_array[5,:]
        # reduced_matrix[:,2,1] =  c_ells_array[5,:]

        # reduced_matrix[:,0,2] =  c_ells_array[5,:]
        # reduced_matrix[:,2,0] =  c_ells_array[5,:]

        # reduced_matrix[:,1,2] =  c_ells_array[4,:]
        # reduced_matrix[:,2,1] =  c_ells_array[4,:]

        reduced_matrix = reduced_matrix.at[:,0,2].set(c_ells_array[5,:])
        reduced_matrix = reduced_matrix.at[:,2,0].set(c_ells_array[5,:])

        reduced_matrix = reduced_matrix.at[:,1,2].set(c_ells_array[4,:])
        reduced_matrix = reduced_matrix.at[:,2,1].set(c_ells_array[4,:])

    return reduced_matrix

def get_c_ells_from_red_covariance_matrix_JAX(red_cov_mat, nstokes=0):
    """ Retrieve c_ell from red_cov_mat, which depending of nstokes will give :
            TT
            EE, BB, EB
            TT, EE, BB, TE, EB, TB
    """
    
    lmax = red_cov_mat.shape[0]
    nstokes = jnp.where(nstokes==0, red_cov_mat.shape[1], nstokes)

    number_correl = jnp.int32(jnp.ceil(nstokes**2/2) + jnp.floor(nstokes/2))
    # number_correl = jnp.array(jnp.ceil(nstokes**2/2) + jnp.floor(nstokes/2),int)
    c_ells = jnp.zeros((number_correl, lmax))

    for i in range(nstokes):
        c_ells = c_ells.at[i,:].set(red_cov_mat[:,i,i])
    if nstokes > 1:
        c_ells= c_ells.at[nstokes,:].set(red_cov_mat[:,0,1])
        if nstokes == 3:
            c_ells = c_ells.at[nstokes+2,:].set(red_cov_mat[:,0,2])
            c_ells = c_ells.at[nstokes+1,:].set(red_cov_mat[:,1,2])
    return c_ells

def get_sqrt_reduced_matrix_from_matrix_jax(red_matrix, tolerance=10**(-15)):
    """ Return L square root matrix
    """
    red_matrix = jnp.array(red_matrix, dtype=jnp.float64)
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    reduced_sqrtm = jnp.zeros_like(red_matrix)

    def func_map(ell):
        eigvals, eigvect = jnp.linalg.eigh(red_matrix[ell,:,:])
        inv_eigvect = jnp.array(jnp.linalg.pinv(eigvect),dtype=jnp.float64)
        return jnp.einsum('jk,km,m,mn->jn', eigvect, jnp.eye(nstokes), jnp.sqrt(jnp.abs(eigvals)), inv_eigvect)
    
    reduced_sqrtm = jax.vmap(func_map, in_axes=0)(jnp.arange(lmax))

    return reduced_sqrtm

def get_cell_from_map_jax(pixel_maps, lmax, n_iter=8):
    def wrapper_anafast(maps_, lmax=lmax, n_iter=n_iter):
        return hp.anafast(maps_, lmax=lmax, iter=n_iter)
        # return np.array([alm_T, alm_E, alm_B])
    
    @partial(jax.jit, static_argnums=1)
    def pure_call_anafast(maps_, lmax):
        # if jnp.size(maps_TQU_input.shape) == 1:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input)/12))
        # else:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input[0])/12))
        shape_output = (6,lmax+1)
        return jax.pure_callback(wrapper_anafast, jax.ShapeDtypeStruct(shape_output, np.float64), maps_)
    
    if jnp.size(pixel_maps.shape) == 1:
        nstokes = 1
    else:
        nstokes = pixel_maps.shape[0]
    
    if nstokes == 2:
        pixel_maps_for_Wishart = jnp.vstack((jnp.zeros_like(pixel_maps[0]), pixel_maps))
            # print("Test 5 :", pixel_maps_for_Wishart.shape, pixel_maps_for_Wishart[0].mean(), pixel_maps_for_Wishart[1].mean(), pixel_maps_for_Wishart[2].mean())
    else:
        pixel_maps_for_Wishart = jnp.copy(pixel_maps)

    # c_ells_Wishart = hp.anafast(pixel_maps_for_Wishart, lmax=lmax, iter=n_iter)
    c_ells_Wishart = pure_call_anafast(pixel_maps_for_Wishart, lmax=lmax)

    if nstokes == 2:
        polar_indexes = jnp.array([1,2,4])
        c_ells_Wishart = c_ells_Wishart[polar_indexes]
    return c_ells_Wishart

def get_MCMC_batch_error(sample_single_chain, batch_size):
    # number_iterations = np.size(sample_single_chain, axis=0)
    number_iterations = sample_single_chain.shape[0]
    assert number_iterations%batch_size == 0

    overall_mean = np.average(sample_single_chain, axis=0)
    standard_error = np.sqrt((batch_size/number_iterations)*((sample_single_chain-overall_mean)**2).sum())
    return standard_error


# @partial(jax.jit, static_argnums=(1,2,3,4))
def maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(maps_TQU_input, red_matrix_sqrt, nside, lmin, n_iter=8):
    lmax = red_matrix_sqrt.shape[0] - 1 + lmin
    nstokes = red_matrix_sqrt.shape[1]

    # all_params = jnp.int16(jnp.where(nstokes > 1, 3, 1))

    # if jnp.size(maps_TQU_input.shape) == 1:
    #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input)/12))
    # else:
    #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input[0])/12))
    all_params = 3

    red_sqrt_decomp = jnp.zeros((lmax+1,all_params,all_params))
    if nstokes != 1:
        # red_sqrt_decomp[lmin:,3-nstokes:,3-nstokes:] = red_matrix_sqrt
        red_sqrt_decomp = red_sqrt_decomp.at[lmin:,3-nstokes:,3-nstokes:].set(red_matrix_sqrt)
    else:
        # red_sqrt_decomp[lmin:,...] = red_matrix_sqrt
        red_sqrt_decomp = red_sqrt_decomp.at[lmin:].set(red_matrix_sqrt)

    if maps_TQU_input.shape[0] == 2:
        maps_TQU = jnp.vstack((jnp.zeros_like(maps_TQU_input[0]),jnp.copy(maps_TQU_input)))
    else:
        maps_TQU = jnp.copy(maps_TQU_input)

    def wrapper_map2alm(maps_, lmax=lmax, n_iter=n_iter, nside=nside):
        alm_T, alm_E, alm_B = hp.map2alm(maps_.reshape((3, 12*nside**2)), lmax=lmax, iter=n_iter)
        return np.array([alm_T, alm_E, alm_B])

    def wrapper_almxfl(alm_, matrix_ell):
        return hp.almxfl(alm_, matrix_ell, inplace=False)
    
    def wrapper_alm2map(alm_, lmax=lmax, nside=nside):
        return hp.alm2map(alm_, nside, lmax=lmax)

    @partial(jax.jit, static_argnums=(1,2))
    def pure_call_map2alm(maps_, lmax, nside):
        # if jnp.size(maps_TQU_input.shape) == 1:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input)/12))
        # else:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input[0])/12))
        shape_output = (3,(lmax+1)*(lmax//2+1))
        return jax.pure_callback(wrapper_map2alm, jax.ShapeDtypeStruct(shape_output, np.complex128), maps_.ravel())

    # @partial(jax.jit, static_argnames=['matrix_ell'])
    def pure_call_almxfl(alm_, matrix_ell):
        shape_output = [(lmax+1)*(lmax//2+1)]
        return jax.pure_callback(wrapper_almxfl, jax.ShapeDtypeStruct(shape_output, np.complex128), alm_, matrix_ell)

    @partial(jax.jit, static_argnums=(1,2))
    def pure_call_alm2map(alm_, lmax, nside):
        # if jnp.size(maps_TQU_input.shape) == 1:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input)/12))
        # else:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input[0])/12))
        shape_output = (3,12*nside**2)
        return jax.pure_callback(wrapper_alm2map, jax.ShapeDtypeStruct(shape_output, np.float64), alm_)

    alms_input = pure_call_map2alm(maps_TQU, lmax=lmax, nside=nside)
    # alms_output = jnp.zeros_like(alms_input)

    # for i in range(all_params):
    #     alms_j = jnp.zeros_like(alms_input[i])
    #     for j in range(all_params):
    #         # alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:,i,j], inplace=False)
    #         result_callback = pure_call_almxfl(alms_input[j], red_sqrt_decomp[:,i,j])
    #         alms_j += result_callback
    #     # alms_output[i] = jnp.copy(alms_j)
    #     alms_output = alms_output.at[i,...].set(jnp.copy(alms_j))
    
    def scan_func(carry, nstokes_j):
        val_alms_j, nstokes_i = carry
        result_callback = pure_call_almxfl(alms_input[nstokes_j], red_sqrt_decomp[:,nstokes_i,nstokes_j])
        new_carry = (val_alms_j + result_callback, nstokes_i)
        return new_carry, val_alms_j + result_callback

    def fmap(nstokes_i):
        return jlax.scan(scan_func, (jnp.zeros_like(alms_input[nstokes_i]),nstokes_i), jnp.arange(all_params))[0][0]

    # last_carry, all_alms = jax.vmap(fmap, in_axes=0)(jnp.arange(all_params))
    alms_output = jax.vmap(fmap, in_axes=0)(jnp.arange(all_params))
    # alms_output, indexe = last_carry

    # maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
    maps_output = pure_call_alm2map(alms_output, nside=nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3-nstokes:,...]
    return maps_output

def get_empirical_covariance_JAX(samples):
    """ Compute empirical covariance from samples
    """
    number_samples = jnp.size(samples, axis=0)
    mean_samples = jnp.mean(samples, axis=0)

    return (jnp.einsum('ti,tj->tij',samples,samples).sum(axis=0) - number_samples*jnp.einsum('i,j->ij',mean_samples,mean_samples))/(number_samples-1)
