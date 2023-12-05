import os, sys, time
import numpy as np
# import healpy as hp
from collections import namedtuple
from jax import random, dtypes
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
import chex as chx
from functools import partial
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC

from .jax_tools import *
from .noisecovar import *
from .mixingmatrix import *

MHState = namedtuple("MHState", ["u", "rng_key"])

class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "u"

    def __init__(self, potential_fn, step_size=0.1):
        """ potential_fn : should be a log proba so that exp(potential_fn) is the probability without the need of a sign change
        """
        self.potential_fn = potential_fn
        self.step_size = step_size

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        accept_prob = jnp.exp(-(self.potential_fn(u, **model_kwargs) - self.potential_fn(u_proposal, **model_kwargs)))
        u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
        return MHState(u_new, rng_key)

class MetropolisHastings_log(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "u"

    def __init__(self, potential_fn, step_size=0.1):
        self.potential_fn = potential_fn
        self.step_size = step_size

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        accept_prob = -(self.potential_fn(u, **model_kwargs) - self.potential_fn(u_proposal, **model_kwargs))
        u_new = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, u)
        return MHState(u_new, rng_key)


def single_Metropolis_Hasting_step(random_PRNGKey, old_sample, step_size, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        u_proposal = dist.Normal(jnp.ravel(old_sample,order='F'), step_size).sample(key_proposal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(u_proposal, **model_kwargs))
        new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, jnp.ravel(old_sample,order='F'))

        return new_sample.reshape(old_sample.shape,order='F')

def multivariate_Metropolis_Hasting_step(random_PRNGKey, old_sample, covariance_matrix, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        u_proposal = dist.MultivariateNormal(jnp.ravel(old_sample,order='F'), covariance_matrix).sample(key_proposal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(u_proposal, **model_kwargs))
        new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, jnp.ravel(old_sample,order='F'))

        return new_sample.reshape(old_sample.shape,order='F')

def single_Metropolis_Hasting_step_positive_constraint(random_PRNGKey, old_sample, step_size, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        u_proposal = dist.Normal(jnp.ravel(old_sample,order='F'), step_size).sample(key_proposal)

        u_proposal = jnp.where(u_proposal < 0, jnp.ravel(old_sample,order='F'), u_proposal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(u_proposal, **model_kwargs))
        new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, jnp.ravel(old_sample,order='F'))

        return new_sample.reshape(old_sample.shape,order='F')


class Sampling_functions(object):
    def __init__(self, nside, lmax, nstokes, 
                 frequency_array, freq_inverse_noise, pos_special_freqs=[0,-1], 
                 number_components=3, lmin=2,
                 n_iter=8, limit_iter_cg=2000, tolerance_CG=10**(-12)):

        # CMB parameters
        self.freq_inverse_noise = freq_inverse_noise
        # self.r_true = float(r_true)

        # Problem parameters
        # print("Test", jnp.int64(2**jnp.log2(nside)), flush=True)
        # chx.assert_equal(2**jnp.log2(nside), jnp.array(nside,dtype=jnp.float64))
        chx.assert_scalar_in(nstokes, 1, 3)
        self.nside = int(nside)
        self.lmax = int(lmax)
        self.nstokes = int(nstokes)
        self.lmin = int(lmin)
        self.n_iter = int(n_iter) # Number of iterations for Python estimation of alms
        self.frequency_array = frequency_array
        self.number_components = int(number_components)
        self.pos_special_freqs = pos_special_freqs

        # CG parameters
        self.limit_iter_cg = int(limit_iter_cg) # Maximum number of iterations for the different CGs
        self.tolerance_CG = float(tolerance_CG) # Tolerance for the different CGs

        # Tools
        fake_params = jnp.zeros((self.number_frequencies-jnp.size(self.pos_special_freqs),self.number_correlations-1))
        self._fake_mixing_matrix = MixingMatrix(self.frequency_array, self.number_components, fake_params, pos_special_freqs=self.pos_special_freqs)

    @property
    def npix(self):
        return 12*self.nside**2

    @property
    def number_correlations(self):
        """ Maximum number of correlations depending of the number of Stokes parameters : 
            6 (TT,EE,BB,TE,EB,TB) for 3 Stokes parameters ; 3 (EE,BB,EB) for 2 Stokes parameters ; 1 (TT) for 1 Stokes parameter"""
        return int(jnp.ceil(self.nstokes**2/2) + jnp.floor(self.nstokes/2))
        # return (jnp.ceil(self.nstokes**2/2) + jnp.floor(self.nstokes/2)).astype(int)

    @property
    def number_frequencies(self):
        return jnp.size(self.frequency_array)

    # @partial(jax.jit,static_argnames=['suppress_low_modes'])
    
    def get_sampling_eta_v2(self, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_x=jnp.empty(0), map_random_y=jnp.empty(0), suppress_low_modes=True):
        """ Solve sampling step 1 : sampling eta
            Solve CG for eta term with formulation : eta = C_approx^(-1/2) ( (E (B^t N^{-1} B)^{-1} B^t N^{-1/2}   x + C_approx^(1/2) y )

            Parameters
            ----------
            param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
            
            red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            BtinvNB : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
            # BtinvN_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

            map_random_x : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
            map_random_y : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
            
            lmin : minimum multipole to be considered, default 0
            
            n_iter : number of iterations for harmonic computations, default 8

            limit_iter_cg : maximum number of iterations for the CG, default 1000
            tolerance : CG tolerance, default 10**(-12)

            initial_guess : initial guess for the CG, default [] (which is a covnention for its initialization to 0)

            Returns
            -------
            eta maps [nstokes, npix]
        """

        # assert red_cov_approx_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
        # chx.assert_axis_dimension(red_cov_approx_matrix, 0, lmax + 1 - lmin)

        # Creation of the random maps if they are not given
        if jnp.size(map_random_x) == 0:
            print("Recalculating x !")
            # map_random_x = np.random.normal(loc=0, scale=1/jhp.nside2resol(nside), size=(number_frequencies,nstokes,npix))
            # map_random_x = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
            map_random_x = jax.random.normal(jax_key_PNRG, shape=(self.number_frequencies,self.nstokes,self.npix))#/jhp.nside2resol(nside)
        if jnp.size(map_random_y) == 0:
            print("Recalculating y !")
            # map_random_y = np.random.normal(loc=0, scale=1/jhp.nside2resol(nside), size=(nstokes,npix))
            map_random_y = jax.random.normal(jax_key_PNRG, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)

        # Computation of the right hand side member of the CG
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)
        # red_cov_approx_matrix_sqrt_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix))

        # First right member : C_approx^(1/2) (E (B^t N^{-1} B)^{-1} E^t)^{-1} C_approx^(1/2) x    
        # first_member = map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))/(BtinvNB[0,0])
        # first_member = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_x)[0]/BtinvNB[0,0] # Selecting CMB component of the random variable
        first_member = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_x)[0]/BtinvNB[0,0]/jhp.nside2resol(self.nside)**2 # Selecting CMB component of the random variable

        if suppress_low_modes:
            covariance_unity = jnp.zeros((self.lmax+1-self.lmin,self.nstokes,self.nstokes))
            # covariance_unity = covariance_unity.at[lmin:,...].set(jnp.eye(nstokes))
            covariance_unity = covariance_unity.at[:,...].set(jnp.eye(self.nstokes))
            first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(jnp.copy(first_member), covariance_unity, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        second_member = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_random_y, jnp.linalg.pinv(red_cov_approx_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        # second_member = micmac.maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_random_y, jnp.linalg.pinv(red_cov_approx_matrix_sqrt_sqrt), nside=nside, lmin=lmin, n_iter=n_iter)

        map_solution_0 = first_member + second_member
        # map_solution_0 = second_member
        # return map_solution_0
        map_solution = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_solution_0.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # if suppress_low_modes:
        #     map_solution = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(jnp.copy(map_solution), covariance_unity, nside=nside, lmin=lmin, n_iter=n_iter)
        return map_solution

    def get_fluctuating_term_maps(self, red_cov_matrix, BtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_realization_xi=jnp.empty(0), map_random_realization_chi=jnp.empty(0), initial_guess=jnp.empty(0)):
        """ 
            Solve fluctuation term with formulation (C^-1 + N^-1) for the left member :
            (C^{-1} + E^t (B^t N^{-1} B)^{-1} E) \zeta = C^{-1/2} xi + (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi

            Parameters
            ----------
            param_dict : dictionnary containing the following fields : nside, nstokes, lmax
            
            red_cov_matrix : covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            red_inverse_noise : matrices of inverse noise in harmonic domain (yet), dimension [lmin:lmax, nstokes, nstokes]

            map_white_noise_xi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            map_white_noise_chi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            
            lmin : minimum multipole to be considered, default 0
            
            n_iter : number of iterations for harmonic computations, default 8

            limit_iter_cg : maximum number of iterations for the CG, default 1000
            tolerance : CG tolerance, default 10**(-12)

            Returns
            -------
            Fluctuation maps [nstokes, npix]
        """

        # assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)

        red_inverse_cov_matrix = jnp.linalg.pinv(red_cov_matrix)
        

        # Creation of the random maps
        if jnp.size(map_random_realization_xi) == 0:
            print("Recalculating xi !")
            # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
            map_random_realization_xi = jax.random.normal(jax_key_PNRG, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)
        if jnp.size(map_random_realization_chi) == 0:
            print("Recalculating chi !")
            # map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))
            map_random_realization_chi = jax.random.normal(jax_key_PNRG, shape=(self.number_frequencies,self.nstokes,self.npix))#/jhp.nside2resol(self.nside)

        # Computation of the right side member of the CG
        red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_inverse_cov_matrix)

        # First right member : C^{-1/2} \xi
        right_member_1 = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_random_realization_xi, red_inv_cov_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        ## Left hand side term : (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi
        # right_member_2 = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]/BtinvNB[0,0] # Selecting CMB component of the random variable
        right_member_2 = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]/BtinvNB[0,0]/jhp.nside2resol(self.nside)**2 # Selecting CMB component of the random variable

        right_member = (right_member_1 + right_member_2).ravel()

        # Computation of the left side member of the CG
        
        # First left member : C^{-1} 
        first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(x.reshape((self.nstokes,self.npix)), red_inverse_cov_matrix, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        
        ## Second left member : (E^t (B^t N^{-1} B) E)
        def second_term_left(x, number_component=self.number_components):
            return x/BtinvNB[0,0]/jhp.nside2resol(self.nside)**2

        func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()
        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(map_random_realization_xi)

        # Actual start of the CG
        # fluctuating_map, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
        fluctuating_map, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)
        print("CG-Python-0 Fluct finished in ", number_iterations, "iterations !!")

        # if exit_code != 0:
        #     print("CG didn't converge with fluctuating term ! Exitcode :", exit_code, flush=True)
        return fluctuating_map.reshape((self.nstokes, self.npix))


    def solve_generalized_wiener_filter_term(self, s_cML, red_cov_matrix, BtinvNB, initial_guess=jnp.empty(0)):
        """ 
            Solve Wiener filter term with formulation (1 + C^1/2 N^-1 C^1/2) for the left member

            Parameters
            ----------
            param_dict : dictionnary containing the following fields : nside, nstokes, lmax

            s_cML : data maps, for Wiener filter CG ; dimensions [nstokes, npix]

            red_cov_matrix : covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            red_inverse_noise : matrices of inverse noise in harmonic domain (yet), dimension [lmin:lmax, nstokes, nstokes]

            map_white_noise_xi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            map_white_noise_chi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            
            lmin : minimum multipole to be considered, default 0
            
            n_iter : number of iterations for harmonic computations, default 8

            limit_iter_cg : maximum number of iterations for the CG, default 1000
            tolerance : CG tolerance, default 10**(-12)

            Returns
            -------
            Wiener filter maps [nstokes, npix]
        """

        # assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        if self.nstokes != 1:
            chx.assert_axis_dimension(s_cML, 0, self.nstokes)
            chx.assert_axis_dimension(s_cML, 1, self.npix)
            # assert s_cML.shape[0] == self.nstokes
            # assert s_cML.shape[1] == self.npix
        

        # Computation of the right side member of the CG
        # s_cML_extended = jnp.zeros((self.number_components, s_cML.shape[0], s_cML.shape[1]))
        # s_cML_extended = s_cML_extended.at[0,...].set(s_cML)
        # s_cML_extended[0,...] = s_cML
    
        # right_member = np.einsum('kc,csp->ksp', np.linalg.pinv(BtinvNB), s_cML_extended)[0].ravel() # Selecting CMB component of the
        # right_member = (s_cML/BtinvNB[0,0]/jhp.nside2resol(self.nside)**2).ravel()
        right_member = (s_cML/BtinvNB[0,0]/jhp.nside2resol(self.nside)**2).ravel()

        # Computation of the left side member of the CG
        first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(x.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        
        ## Second left member : (E^t (B^t N^{-1} B)^{-1} E)^{-1} x
        def second_term_left(x, number_component=self.number_components):
            return x/BtinvNB[0,0]/jhp.nside2resol(self.nside)**2

        func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(s_cML)

        # Actual start of the CG
        # wiener_filter_term, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
        wiener_filter_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)
    
        print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")

        # if exit_code != 0:
        #     print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
        return wiener_filter_term.reshape((self.nstokes, self.npix))


    def get_inverse_wishart_sampling_from_c_ells(self, sigma_ell, q_prior=0, option_ell_2=2, tol=1e-10):
        """ Solve sampling step 3 : inverse Wishart distribution with S_c
            Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
                1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
                2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
                3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            sigma_ell : initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [number_correlations, lmax+1]
            
            q_prior : choice of prior for the distribution : 0 means uniform prior ; 1 means Jeffrey prior
            
            lmin : minimum multipole to be considered, default 0

            option_ell_2 : option to choose how to sample ell=2, which is not defined by inverse Wishart distribution if nstokes=3 ; ignored if lmin>2 and/or nstokes<3
                        case 1 : sample ell=2 with Jeffrey prior (only for ell=2)
                        case 2 : sample ell=2 by sampling separately the TE and B blocks respectively, assumes TB and EB to be 0

            Returns
            -------
            Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """
        print("C sampling from Inverse Wishart does not have a 'JAX' version yet", flush=True)
        # raise NotImplemented

        # nstokes = jnp.where(jnp.size(sigma_ell.shape)==1, 1, 0)
        # nstokes = jnp.where(jnp.logical_and(nstokes==0,sigma_ell.shape[0] == 3), 2, 3)

        # lmax = jnp.where(nstokes==1, sigma_ell.shape[0] - 1, sigma_ell.shape[1] - 1)

        # if len(sigma_ell.shape) == 1:
        #     nstokes == 1
        #     lmax = len(sigma_ell) - 1
        # elif sigma_ell.shape[0] == 6:
        #     nstokes = 3
        #     lmax = sigma_ell.shape[1] - 1
        # elif sigma_ell.shape[0] == 3:
        #     nstokes = 2
        #     lmax = sigma_ell.shape[1] - 1

        # for i in range(nstokes):
        #     sigma_ell[i] *= 2*np.arange(lmax+1) + 1
        c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        invert_parameter_Wishart_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(invert_parameter_Wishart)
        
        lmin = self.lmin

        # assert invert_parameter_Wishart.shape[0] == lmax + 1 #- lmin
        chx.assert_shape(invert_parameter_Wishart, (self.lmax + 1, self.nstokes, self.nstokes))
        
        sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)
        # assert (option_ell_2 == 0) or (option_ell_2 == 1) or (option_ell_2 == 2)
        chx.assert_scalar_in(option_ell_2, 0, 2)

        # Option sampling without caring about inverse Wishart not defined
        ell_2 = 2
        # PRNGKey, rng_key_lmin_2 = random.split(PRNGKey)
        if self.lmin <= 2 and (2*ell_2 + 1 - 2*self.nstokes + 2*q_prior <= 0):
            
            # 2*ell_2 + 1 - 2*nstokes + 2*q_prior <= 0) correspond to the definition condition of the inverse Wishart distribution

            # # Option sampling with brute force inverse Wishart
            # if option_ell_2 == -1:
            #     print("~~Not caring about inverse Wishart distribution not defined !", flush=True)
            #     mean = np.zeros(nstokes)
            #     sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*q_prior))
            #     sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)

            # Option sampling with Jeffrey prior
            if option_ell_2 == 0:
                Jeffrey_prior = 1
                print("~~Applying Jeffry prior for ell=2 !", flush=True)
                # mean = np.zeros(nstokes)
                # sample_gaussian = np.random.multivariate_normal(mean, invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*Jeffrey_prior))
                # sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)

                sample_gaussian = random.multivariate_normal(jnp.zeros(self.nstokes), invert_parameter_Wishart[ell_2], size=(2*ell_2 - self.nstokes + 2*Jeffrey_prior), check_valid='warn', tol=tol)
                sampling_Wishart = sampling_Wishart.at[ell_2].set(jnp.einsum('lk,lm->km',sample_gaussian,sample_gaussian))
            # Option sampling separately TE and B
            elif option_ell_2 == 1:
                print("~~Sampling separately TE and B for ell=2 !", flush=True)
                reduced_matrix_2 = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)[ell_2]
                invert_parameter_Wishart_2 = jnp.zeros((self.nstokes,self.nstokes))

                invert_parameter_Wishart_2 = invert_parameter_Wishart_2.at[:self.nstokes-1,:self.nstokes-1].set(jnp.linalg.pinv(reduced_matrix_2[:self.nstokes-1,:self.nstokes-1]))
                invert_parameter_Wishart_2 = invert_parameter_Wishart_2.at[self.nstokes-1,self.nstokes-1].set(1/reduced_matrix_2[self.nstokes-1,self.nstokes-1])

                eigvals, eigvect = jnp.linalg.eigh(invert_parameter_Wishart_2)
                inv_eigvect = jnp.array(jnp.linalg.pinv(eigvect),dtype=jnp.float64)
                invert_parameter_Wishart_2_sqrt = jnp.einsum('jk,km,m,mn->jn', eigvect, jnp.eye(self.nstokes), jnp.sqrt(jnp.abs(eigvals)), inv_eigvect)
                
                sample_gaussian_TE = random.multivariate_normal(np.zeros(self.nstokes-1), invert_parameter_Wishart_2[:self.nstokes-1,:self.nstokes-1], size=(2*ell_2 - (self.nstokes-1)))
                sample_gaussian_B = random.normal(loc=0, scale=invert_parameter_Wishart_2[self.nstokes-1,self.nstokes-1], size=(2*ell_2 - 1))

                sampling_Wishart = sampling_Wishart.at[ell_2,:self.nstokes-1,:self.nstokes-1].set(jnp.einsum('lk,lm->km',sample_gaussian_TE,sample_gaussian_TE))
                sampling_Wishart = sampling_Wishart.at[ell_2,self.nstokes-1,self.nstokes-1].set(jnp.einsum('li,li->i',sample_gaussian_B,sample_gaussian_B))

            lmin = 3

        for ell in range(max(lmin,2),self.lmax+1):
            sample_gaussian = jnp.random.multivariate_normal(np.zeros(self.nstokes), invert_parameter_Wishart[ell], size=(2*ell - self.nstokes + 2*q_prior))
            sampling_Wishart[ell] = np.dot(sample_gaussian.T,sample_gaussian)
        # sampling_Wishart[max(lmin,2):,...] = np.einsum('lkj,lkm->ljm',sample_gaussian,sample_gaussian)
        return np.linalg.pinv(sampling_Wishart)

    def get_conditional_proba_C_from_r(self, r_param, red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total):
        # red_sigma_ell = model_kwargs['red_sigma_ell']

        # red_cov_matrix_sampled = model_kwargs['red_cov_matrix_sampled']
        # red_cov_matrix_sampled = r_param * model_kwargs['theoretical_red_cov_r1_tensor'] + model_kwargs['theoretical_red_cov_r0_total']
        red_cov_matrix_sampled = r_param * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        sum_dets = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_cov_matrix_sampled)) ).sum()
        
        return -( jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets)/2

    # @partial(jax.jit)
    def get_conditional_proba_spectral_likelihood_JAX(self, complete_mixing_matrix, full_data_without_CMB, suppress_low_modes=False):
        """ Get conditional probability of spectral likelihood by sampling it using emcee

            The associated conditional probability is given by : 
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        """

        # Building the spectral_likelihood : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        complete_mixing_matrix_fg = complete_mixing_matrix[:,1:]

        BtinvNB_fg = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)
        BtinvN_fg = get_BtinvN(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)

        full_data_without_CMB_with_noise = jnp.einsum('cf,fsp->csp', BtinvN_fg, full_data_without_CMB)
        if suppress_low_modes:
            covariance_unity = jnp.zeros((self.lmax+1-self.lmin,self.nstokes,self.nstokes))
            # covariance_unity = covariance_unity.at[lmin:,...].set(jnp.eye(nstokes))
            covariance_unity = covariance_unity.at[:,...].set(jnp.eye(self.nstokes))
            for i in range(self.number_components-1):
                full_data_without_CMB_with_noise = full_data_without_CMB_with_noise.at[i].set(maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(jnp.copy(full_data_without_CMB_with_noise[i]), covariance_unity, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter))

        first_term_complete = jnp.einsum('psc,cm,msp', full_data_without_CMB_with_noise.T, BtinvNB_fg, full_data_without_CMB_with_noise)
        return -(-first_term_complete + 0)/2.

    def get_conditional_proba_spectral_likelihood_JAX_alt(self, complete_mixing_matrix, full_data_without_CMB, suppress_low_modes=False):
        """ Get conditional probability of spectral likelihood by sampling it using emcee

            The associated conditional probability is given by : 
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        """

        # Building the spectral_likelihood : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        complete_mixing_matrix_fg = complete_mixing_matrix[:,1:]

        BtinvNB_fg = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)*jhp.nside2resol(self.nside)**2
        BtinvN_fg = get_BtinvN(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)/jhp.nside2resol(self.nside)**2

        full_data_without_CMB_with_noise = jnp.einsum('cf,fsp->csp', BtinvN_fg, full_data_without_CMB)
        if suppress_low_modes:
            covariance_unity = jnp.zeros((self.lmax+1-self.lmin,self.nstokes,self.nstokes))
            # covariance_unity = covariance_unity.at[lmin:,...].set(jnp.eye(nstokes))
            covariance_unity = covariance_unity.at[:,...].set(jnp.eye(self.nstokes))
            for i in range(self.number_components-1):
                full_data_without_CMB_with_noise = full_data_without_CMB_with_noise.at[i].set(maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(jnp.copy(full_data_without_CMB_with_noise[i]), covariance_unity, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter))

        first_term_complete = jnp.einsum('psc,cm,msp', full_data_without_CMB_with_noise.T, BtinvNB_fg, full_data_without_CMB_with_noise)
        return -(-first_term_complete + 0)/2.

    # @partial(jax.jit, static_argnames=['with_prints'])
    def get_conditional_proba_perturbation_likelihood_JAX_v2_slow(self, complete_mixing_matrix, modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False):
        """ Get conditional probability of perturbation likelihood by sampling it using numpyro

            The associated conditional probability is given by : 
        """

        new_BtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix, jax_use=True)
        # new_BtinvN_sqrt = micmac.get_BtinvN(jnp.array(jsp.linalg.sqrtm(freq_inverse_noise), dtype=jnp.float64), complete_mixing_matrix, jax_use=True)

        # N_c_sqrt = jnp.einsum('ck,kf->cf', new_BtinvNB, new_BtinvN_sqrt)[0,:]
        # modified_sample_eta_maps_2 = np.copy(modified_sample_eta_maps)
        red_cov_approx_matrix_msqrt = jnp.linalg.pinv(get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix))
        modified_sample_eta_maps_2 = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(modified_sample_eta_maps, red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        operator_harmonic = jnp.linalg.pinv(red_cov_approx_matrix)
        operator_pixel = 1/new_BtinvNB[0,0]/jhp.nside2resol(self.nside)**2
        # operator_harmonic = red_cov_approx_matrix
        # operator_pixel = new_BtinvNB[0,0]
        first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(x.reshape((self.nstokes,self.npix)), operator_harmonic, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        def second_term_left(x):
            return operator_pixel*x.reshape((self.nstokes,self.npix))

        func_left_term = lambda x : (first_term_left(x) + second_term_left(x)).ravel()

        initial_guess = jnp.zeros((self.nstokes,self.npix))
        right_member = jnp.copy(modified_sample_eta_maps_2)    
        inverse_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)
        if with_prints:
            print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")

        modified_sample_eta_maps_3 = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(inverse_term.reshape(self.nstokes,self.npix), red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # And finally \eta^t N_c^{1/2] (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
        second_term_complete = jnp.einsum('sk,sk', modified_sample_eta_maps, modified_sample_eta_maps_3)

        return -(-0 + second_term_complete)/2.

    # def get_conditional_proba_perturbation_likelihood_JAX_v2_harm(self, complete_mixing_matrix, modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False):
    #     """ Get conditional probability of perturbation likelihood by computing it directly in harmonic space

    #         The associated conditional probability is given by :
    #     """

    #     new_BtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix, jax_use=True)

    #     _cl_noise_harm = new_BtinvNB[0,0]*jhp.nside2resol(self.nside)**2

    #     # cl_noise_harm_CMB = jnp.zeros((self.number_correlations, self.lmax+1-self.lmin))
    #     # cl_noise_harm_CMB = cl_noise_harm_CMB.at[0].set(_cl_noise_harm[0,0,:])
    #     # cl_noise_harm_CMB = cl_noise_harm_CMB.at[1].set(_cl_noise_harm[0,0,:])
    #     red_cl_noise_harm_CMB = jnp.zeros_like(red_cov_approx_matrix)
    #     red_cl_noise_harm_CMB = red_cl_noise_harm_CMB.at[:,0,0].set(_cl_noise_harm)
    #     red_cl_noise_harm_CMB = red_cl_noise_harm_CMB.at[:,1,1].set(_cl_noise_harm)

    #     red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)
    #     # red_cl_noise_harm_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cl_noise_harm_CMB)

    #     red_correction_term = jnp.linalg.pinv(jnp.einsum('lkm,lmn,lno->lko',red_cov_approx_matrix_sqrt, 
    #                                                 jnp.linalg.pinv(red_cov_approx_matrix) + jnp.linalg.pinv(red_cl_noise_harm_CMB),
    #                                                 red_cov_approx_matrix_sqrt))

    #     # red_correction_term_2 = jnp.einsum('lkm,lmn,lno->lko',red_cl_noise_harm_sqrt, 
    #     #                                             np.linalg.pinv(red_cov_approx_matrix + red_cl_noise_harm_CMB),
    #     #                                             red_cl_noise_harm_sqrt)

    #     log_det_correction = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_correction_term)) ).sum()
    #     # log_det_correction_2 = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_correction_term_2)) ).sum()
    #     # print('Log det :', log_det_correction, flush=True)
    #     # print('Log det :', log_det_correction_2, flush=True)

    #     return -log_det_correction/2.
    
    def get_conditional_proba_perturbation_likelihood_JAX_v1_harm(self, complete_mixing_matrix, modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False):
        """ Get conditional probability of perturbation likelihood by computing it directly in harmonic space

            The associated conditional probability is given by :
        """

        new_BtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix, jax_use=True)

        _cl_noise_harm = new_BtinvNB[0,0]*jhp.nside2resol(self.nside)**2

        # cl_noise_harm_CMB = jnp.zeros((self.number_correlations, self.lmax+1-self.lmin))
        # cl_noise_harm_CMB = cl_noise_harm_CMB.at[0].set(_cl_noise_harm[0,0,:])
        # cl_noise_harm_CMB = cl_noise_harm_CMB.at[1].set(_cl_noise_harm[0,0,:])
        red_cl_noise_harm_CMB = jnp.zeros_like(red_cov_approx_matrix)
        red_cl_noise_harm_CMB = red_cl_noise_harm_CMB.at[:,0,0].set(_cl_noise_harm)
        red_cl_noise_harm_CMB = red_cl_noise_harm_CMB.at[:,1,1].set(_cl_noise_harm)

        # red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)
        red_cl_noise_harm_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cl_noise_harm_CMB)

        red_correction_term = jnp.einsum('lkm,lmn,lno->lko',red_cl_noise_harm_sqrt, 
                                                    jnp.linalg.pinv(red_cov_approx_matrix + red_cl_noise_harm_CMB),
                                                    red_cl_noise_harm_sqrt)

        # red_correction_term_2 = jnp.einsum('lkm,lmn,lno->lko',red_cl_noise_harm_sqrt, 
        #                                             np.linalg.pinv(red_cov_approx_matrix + red_cl_noise_harm_CMB),
        #                                             red_cl_noise_harm_sqrt)

        log_det_correction = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_correction_term)) ).sum()
        # log_det_correction_2 = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_correction_term_2)) ).sum()
        # print('Log det :', log_det_correction, flush=True)
        # print('Log det :', log_det_correction_2, flush=True)

        return -log_det_correction/2.


    
    def get_conditional_proba_mixing_matrix_v2_slow_JAX_alt(self, new_params_mixing_matrix, full_data_without_CMB, modified_sample_eta_maps, red_cov_approx_matrix):
        params_mixing_matrix = jnp.copy(new_params_mixing_matrix)

        # new_mixing_matrix = create_mixing_matrix_jax(params_mixing_matrix, self.number_components, self.number_frequencies, pos_special_freqs=self.pos_special_freqs)
        self._fake_mixing_matrix.update_params(params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        # log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_alt(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))

        log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v2_slow(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)

        return log_proba_spectral_likelihood + log_proba_perturbation_likelihood

    def get_conditional_proba_mixing_matrix_v1_slow_JAX_alt_harm(self, new_params_mixing_matrix, full_data_without_CMB, modified_sample_eta_maps, red_cov_approx_matrix):
        params_mixing_matrix = jnp.copy(new_params_mixing_matrix)

        # new_mixing_matrix = create_mixing_matrix_jax(params_mixing_matrix, self.number_components, self.number_frequencies, pos_special_freqs=self.pos_special_freqs)
        self._fake_mixing_matrix.update_params(params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        # log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
        # log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_alt_harm(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
        # log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_alt_harm_wrestling(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_alt(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
        
        # log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v2_slow(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)
        log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v1_harm(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)
        # log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v2_c_fast(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)
        return log_proba_spectral_likelihood + log_proba_perturbation_likelihood


    def get_biased_conditional_proba_mixing_matrix_v2_slow_JAX(self, new_params_mixing_matrix, full_data_without_CMB, modified_sample_eta_maps, red_cov_approx_matrix):
        params_mixing_matrix = jnp.copy(new_params_mixing_matrix)

        # new_mixing_matrix = create_mixing_matrix_jax(params_mixing_matrix, self.number_components, self.number_frequencies, pos_special_freqs=self.pos_special_freqs)
        self._fake_mixing_matrix.update_params(params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))

        # log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v2_slow(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)
        return log_proba_spectral_likelihood #+ log_proba_perturbation_likelihood

    # def get_biased_conditional_proba_mixing_matrix_v2_slow_JAX_alt(self, new_params_mixing_matrix, full_data_without_CMB, modified_sample_eta_maps, red_cov_approx_matrix):
    #     params_mixing_matrix = jnp.copy(new_params_mixing_matrix)

    #     # new_mixing_matrix = create_mixing_matrix_jax(params_mixing_matrix, self.number_components, self.number_frequencies, pos_special_freqs=self.pos_special_freqs)
    #     self._fake_mixing_matrix.update_params(params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
    #     new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)

    #     # log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
    #     log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_alt(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))

    #     # log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v2_slow(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)
    #     return log_proba_spectral_likelihood #+ log_proba_perturbation_likelihood
    
    # def get_biased_conditional_proba_mixing_matrix_v1_slow_JAX_alt(self, new_params_mixing_matrix, full_data_without_CMB, modified_sample_eta_maps, red_cov_approx_matrix):
    #     params_mixing_matrix = jnp.copy(new_params_mixing_matrix)

    #     # new_mixing_matrix = create_mixing_matrix_jax(params_mixing_matrix, self.number_components, self.number_frequencies, pos_special_freqs=self.pos_special_freqs)
    #     self._fake_mixing_matrix.update_params(params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
    #     new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)

    #     # log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))
    #     log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_alt_harm_wrestling(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))

    #     # log_proba_perturbation_likelihood = self.get_conditional_proba_perturbation_likelihood_JAX_v2_slow(jnp.copy(new_mixing_matrix), modified_sample_eta_maps, red_cov_approx_matrix, with_prints=False)
    #     return log_proba_spectral_likelihood #+ log_proba_perturbation_likelihood


# def get_sample_parameter(mcmc_kernel, full_initial_guess, random_PRNGKey=random.PRNGKey(100), **model_kwargs):
#     """ The mcmc_kernel provided must be provided with a log_proba function which aims at be maximised !!! Not minimised

#         One must then provide it a log L function, not a -2 log L
#     """
#     mcmc_kernel.run(random_PRNGKey, init_params=full_initial_guess, **model_kwargs)    
#     mcmc_kernel.print_summary()
#     return mcmc_kernel.get_samples(group_by_chain=True)





#### Not to consider yet
def get_sampling_eta_prime_JAX(number_frequencies, nstokes, nside, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=jnp.empty(0), map_random_y=jnp.empty(0), jax_key_PNRG=jax.random.PRNGKey(1), lmin=0, n_iter=8):
    """ Solve sampling step 1 : sampling eta'
        Solve CG for eta term with formulation : eta' = C_approx^(1/2) x + (B^t N^{-1} B)^{-1} B^T N^{-1/2} y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
        
        red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        BtinvNB : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        # BtinvN_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

        map_random_x : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
        map_random_y : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        limit_iter_cg : maximum number of iterations for the CG, default 1000
        tolerance : CG tolerance, default 10**(-12)

        initial_guess : initial guess for the CG, default [] (which is a covnention for its initialization to 0)

        Returns
        -------
        eta maps [nstokes, npix]
    """

    # assert red_cov_approx_matrix.shape[0] == param_dict['lmax'] + 1 - lmin


    # Creation of the random maps if they are not given
    if jnp.size(map_random_x) == 0:
        print("Recalculating x !")
        # map_random_x = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
        map_random_x = jax.random.normal(jax_key_PNRG, shape=(nstokes,12*nside**2))/jhp.nside2resol(nside)
    if jnp.size(map_random_y) == 0:
        print("Recalculating y !")
        # map_random_y = np.random.normal(loc=0, scale=1/jhp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))
        map_random_y = jax.random.normal(jax_key_PNRG+1, shape=(number_frequencies,nstokes,12*nside**2))/jhp.nside2resol(nside)
    # Computation of the right hand side member of the CG
    red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)

    # First right member : C_approx^{1/2} x
    # first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix_sqrt, lmin=lmin, n_iter=n_iter)
    first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_random_x.reshape((nstokes,12*nside**2)), red_cov_approx_matrix_sqrt, nside=nside, lmin=lmin, n_iter=n_iter)
    # # Second right member : E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2}
    second_member = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_y)[0] # Selecting CMB component of the random variable

    return first_member + second_member

def get_sampling_eta_JAX(number_frequencies, nstokes, nside, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=jnp.empty(0), map_random_y=jnp.empty(0), jax_key_PNRG=jax.random.PRNGKey(1), lmin=0, n_iter=8):
    """ Solve sampling step 1 : sampling eta'
        Solve CG for eta term with formulation : eta' = C_approx^(1/2) x + (B^t N^{-1} B)^{-1} B^T N^{-1/2} y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
        
        red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        BtinvNB : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        # BtinvN_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

        map_random_x : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
        map_random_y : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        limit_iter_cg : maximum number of iterations for the CG, default 1000
        tolerance : CG tolerance, default 10**(-12)

        initial_guess : initial guess for the CG, default [] (which is a covnention for its initialization to 0)

        Returns
        -------
        eta maps [nstokes, npix]
    """

    # assert red_cov_approx_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    eta_prime_jax = get_sampling_eta_prime_JAX(number_frequencies, nstokes, nside, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, jax_key_PNRG=jax_key_PNRG, lmin=lmin, n_iter=n_iter)

    # eta_prime_jax_extended_frequencies = jnp.repeat(eta_prime_jax, number_frequencies).reshape((number_frequencies,nstokes,12*nside**2),order='F')

    # Transform into eta maps by applying N_c^{-1/2} = N_c^{-1} N_c^{1/2} = (E^t (B^t N^{-1} B)^{-1} E)^{-1} E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2}
    # First applying N_c^{1/2}
    # first_part = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, eta_prime_jax_extended_frequencies)
    # first_part = jnp.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, eta_prime_jax_extended_frequencies)[0]
    first_part = jnp.einsum('fc,c,sp->fsp', BtinvN_sqrt.T, BtinvNB.T[0,:], eta_prime_jax/(BtinvNB[0,0]))

    # Then applying N_c^{-1}
    # return 1/(BtinvNB[0,0])*first_part
    return first_part
    # return jnp.einsum('kc,csp', jnp.linalg.pinv(BtinvNB), first_part)[0]

