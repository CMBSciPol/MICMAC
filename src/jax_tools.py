import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import healpy as hp
from functools import partial

def get_reduced_matrix_from_c_ell_jax(c_ells_input):
    """ Expect c_ells_input to be sorted as TT, EE, BB, TE, TB, EB if 6 spectra are given

        Generate covariance matrix from c_ells assuming it's block diagonal
    """
    c_ells_array = jnp.copy(c_ells_input)
    number_correlations = c_ells_array.shape[0]
    assert number_correlations == 1 or number_correlations == 3 or number_correlations == 6
    lmax = c_ells_array.shape[1]
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
            for i in range(6 - number_correlations):
                c_ells_array = jnp.vstack((c_ells_array, jnp.zeros(lmax)))
            number_correlations = 6
    else :
        raise Exception("C_ells must be given as TT for temperature only ; EE, BB, EB for polarization only ; TT, EE, BB, TE, (TB, EB) for both temperature and polarization")

    reduced_matrix = jnp.zeros((lmax,nstokes,nstokes))

    for i in range(nstokes):
        reduced_matrix[:,i,i] =  c_ells_array[i,:]
    
    # for j in range(number_correlations-nstokes):
    if number_correlations > 1:
        reduced_matrix[:,0,1] =  c_ells_array[nstokes,:]
        reduced_matrix[:,1,0] =  c_ells_array[nstokes,:]

    if number_correlations == 6:
        # reduced_matrix[:,0,2] =  c_ells_array[4,:]
        # reduced_matrix[:,2,0] =  c_ells_array[4,:]

        # reduced_matrix[:,1,2] =  c_ells_array[5,:]
        # reduced_matrix[:,2,1] =  c_ells_array[5,:]

        reduced_matrix[:,0,2] =  c_ells_array[5,:]
        reduced_matrix[:,2,0] =  c_ells_array[5,:]

        reduced_matrix[:,1,2] =  c_ells_array[4,:]
        reduced_matrix[:,2,1] =  c_ells_array[4,:]

    return reduced_matrix

def get_sqrt_reduced_matrix_from_matrix_jax(red_matrix, tolerance=10**(-15)):
    """ Return L square root matrix
    """
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    reduced_sqrtm = np.zeros_like(red_matrix)

    for ell in range(red_matrix.shape[0]):
        # if np.any(np.iscomplex(scipy.linalg.sqrtm(red_matrix[ell,:,:]))):
        #     print("COMPLEX ELEMENT IN SQRT RED MATRIX FOR ELL = {} !!!".format(ell), flush=True)
        # reduced_sqrtm[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
        eigvals, eigvect = jnp.linalg.eigh(red_matrix[ell,:,:])

        try:
            inv_eigvect = jnp.linalg.pinv(eigvect)
        except:
            raise Exception("Error for ell=",ell, "eigvals", eigvals, "eigvect", eigvect, "red_matrix", red_matrix[ell,:,:])

        if not(jnp.all(eigvals>0)) and (jnp.abs(eigvals[eigvals<0]) > tolerance):
            raise Exception("Covariance matrix not consistent with a negative eigval for ell=",ell, "eigvals", eigvals, "eigvect", eigvect, "red_matrix", red_matrix[ell,:,:])

        reduced_sqrtm[ell] = jnp.einsum('jk,km,m,mn->jn', eigvect, jnp.eye(nstokes), jnp.sqrt(np.abs(eigvals)), inv_eigvect)
    return reduced_sqrtm

# @partial(jax.jit, static_argnums=(1,2,3,4))
def maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(maps_TQU_input, red_matrix_sqrt, nside, lmin=0, n_iter=8):
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
        shape_output = (3,(lmax+1)*(nside+1))
        return jax.pure_callback(wrapper_map2alm, jax.ShapeDtypeStruct(shape_output, np.complex128), maps_.ravel())
    
    # @partial(jax.jit, static_argnames=['matrix_ell'])
    def pure_call_almxfl(alm_, matrix_ell):
        shape_output = [(lmax+1)*(nside+1)]
        return jax.pure_callback(wrapper_almxfl, jax.ShapeDtypeStruct(shape_output, np.complex128), alm_, matrix_ell)

    @partial(jax.jit, static_argnums=(1,2))
    def pure_call_alm2map(alm_, lmax, nside):
        # if jnp.size(maps_TQU_input.shape) == 1:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input)/12))
        # else:
        #     nside = jnp.int64(np.sqrt(jnp.size(maps_TQU_input[0])/12))
        shape_output = (3,12*nside**2)
        return jax.pure_callback(wrapper_alm2map, jax.ShapeDtypeStruct(shape_output, np.float64), alm_)

    # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    alms_input = pure_call_map2alm(maps_TQU, lmax=lmax, nside=nside)
    # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    alms_output = jnp.zeros_like(alms_input)

    for i in range(all_params):
        alms_j = jnp.zeros_like(alms_input[i])
        for j in range(all_params):
            # alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:,i,j], inplace=False)
            result_callback = pure_call_almxfl(alms_input[j], red_sqrt_decomp[:,i,j])
            alms_j += result_callback
        # alms_output[i] = jnp.copy(alms_j)
        alms_output = alms_output.at[i,...].set(jnp.copy(alms_j))
    # maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
    maps_output = pure_call_alm2map(alms_output, nside=nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3-nstokes:,...]
    return maps_output
