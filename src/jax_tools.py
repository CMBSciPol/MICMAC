import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import healpy as hp
from functools import partial

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
