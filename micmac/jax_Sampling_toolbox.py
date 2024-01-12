import os, sys, time
from jax import random, dtypes
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jlax
import jax_healpy as jhp
import chex as chx
import lineax as lx
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC

from .jax_tools import *
from .noisecovar import *
from .mixingmatrix import *


class Sampling_functions(object):
    def __init__(self, nside, lmax, nstokes, 
                 frequency_array, freq_inverse_noise, pos_special_freqs=[0,-1],
                 mask=None,
                 number_components=3, lmin=2,
                 n_iter=8, 
                 limit_iter_cg=2000, limit_iter_cg_eta=50, 
                 tolerance_CG=1e-10, atol_CG=1e-8,
                 restrict_to_mask=False):
        """ Sampling functions object
            Contains all the functions needed for the sampling step of the Gibbs sampler

            Parameters
            ----------
            :param nside: nside of the maps
            :param lmax: maximum ell to be considered
            :param nstokes: number of Stokes parameters
            :param frequency_array: set of frequencies in GHz
            :param freq_inverse_noise: inverse noise in uK^2
            :param pos_special_freqs: position of the special frequencies (0 and -1 by default)
            :param mask: mask to be applied on the maps (default None if no mask)
            :param number_components: number of components to be considered (default 3)
            :param lmin: minimum ell to be considered (default 2)
            :param n_iter: number of iterations for harmonic computations (default 8)
            :param limit_iter_cg: maximum number of iterations for the CG (default 2000)
            :param tolerance_CG: CG tolerance (default 10**(-12))
        """
        # Tests parameters
        self.restrict_to_mask = bool(restrict_to_mask)

        # Problem parameters
        self.freq_inverse_noise = freq_inverse_noise
        self.frequency_array = frequency_array
        chx.assert_scalar_in(nstokes, 1, 3)
        self.nstokes = int(nstokes)
        self.nside = int(nside)
        self.lmax = int(lmax)
        assert lmin >= 2
        self.lmin = int(lmin)
        self.number_components = int(number_components)
        self.pos_special_freqs = pos_special_freqs
        if mask is None:
            self.mask = jnp.ones(12*self.nside**2)
        else:
            self.mask = mask
        
        # CG and harmonic parameters
        self.n_iter = int(n_iter) # Number of iterations for estimation of alms
        self.limit_iter_cg = int(limit_iter_cg) # Maximum number of iterations for the different CGs
        self.tolerance_CG = float(tolerance_CG) # Tolerance for the different CGs
        self.atol_CG = float(atol_CG) # Absolute tolerance for the different CGs
        self.limit_iter_cg_eta = float(limit_iter_cg_eta) # Maximum number of iterations for the CG of eta

        # Tools
        fake_params = jnp.zeros((self.number_frequencies-jnp.size(self.pos_special_freqs),self.number_correlations-1))
        self._fake_mixing_matrix = MixingMatrix(self.frequency_array, self.number_components, fake_params, pos_special_freqs=self.pos_special_freqs)

    @property
    def npix(self):
        """ Number of pixels
        """
        return 12*self.nside**2

    @property
    def number_correlations(self):
        """ Maximum number of correlations depending of the number of Stokes parameters : 
            6 (TT,EE,BB,TE,EB,TB) for 3 Stokes parameters ; 3 (EE,BB,EB) for 2 Stokes parameters ; 1 (TT) for 1 Stokes parameter
        """
        return int(jnp.ceil(self.nstokes**2/2) + jnp.floor(self.nstokes/2))

    @property
    def number_frequencies(self):
        """ Return number of frequencies
        """
        return jnp.size(self.frequency_array)

    def get_band_limited_maps(self, input_map):
        """ Get band limited maps from input maps between lmin and lmax
            :param input_map: input maps to be band limited ; dimension [nstokes, npix]
            :return: band limited maps ; dimension [nstokes, npix]
        """

        covariance_unity = jnp.zeros((self.lmax+1-self.lmin,self.nstokes,self.nstokes))
        covariance_unity = covariance_unity.at[:,...].set(jnp.eye(self.nstokes))

        return maps_x_red_covariance_cell_JAX(jnp.copy(input_map), covariance_unity, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
    
    def get_sampling_eta_v2(self, red_cov_approx_matrix, invBtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_x=None, map_random_y=None, suppress_low_modes=True):
        """ Sampling step 1 : eta maps
            Solve CG for eta term with formulation : eta =  C_approx^(1/2) ( N_c^{-1/2} x + C_approx^(-1/2) y )
            Or :
                eta = C_approx^(1/2) ( (E (B^t N^{-1} B)^{-1} E) E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x + C_approx^(-1/2) y )

            Parameters
            ----------            
            :param red_cov_approx_matrix : covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB : matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            :param BtinvN_sqrt : matrix B^T N^{-1/2}, dimension [component, frequencies, npix]

            :param map_random_x : set of maps 0 with mean and variance 1, which will be used to compute eta, default None (then computed in the routine) ; dimension [nfreq, nstokes, npix]
            :param map_random_y : set of maps 0 with mean and variance 1, which will be used to compute eta, default None (then computed in the routine) ; dimension [nstokes, npix]

            :param suppress_low_modes : if True, suppress low modes in the CG between lmin and lmax, default True
            Returns
            -------
            :return: eta maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_approx_matrix, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.npix)

        # Creation of the random maps if they are not given
        jax_key_PNRG, jax_key_PNRG_x = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        # if jnp.size(map_random_x) == 0:
        if map_random_x is None:
            print("Recalculating x !")
            map_random_x = jax.random.normal(jax_key_PNRG_x, shape=(self.number_frequencies,self.nstokes,self.npix))#/jhp.nside2resol(nside)

        jax_key_PNRG, jax_key_PNRG_y = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        # if jnp.size(map_random_y) == 0:
        if map_random_y is None:
            print("Recalculating y !")
            map_random_y = jax.random.normal(jax_key_PNRG_y, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)#*self.mask

        # Computation of the right hand side member of the equation
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix) # Getting sqrt of the covariance matrix
        
        ## First right member N_c^{-1/2} x
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        first_member = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_x)[0]*N_c_inv # Selecting CMB component of the random variable

        if suppress_low_modes:
            first_member = self.get_band_limited_maps(first_member)

        ## Second right member C_approx^(-1/2) y
        second_member = maps_x_red_covariance_cell_JAX(map_random_y, jnp.linalg.pinv(red_cov_approx_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # Getting partial solution
        map_solution_0 = first_member + second_member

        # Getting final solution : C_approx^(1/2) ( N_c^{-1/2} x + C_approx^(-1/2) y)
        map_solution = maps_x_red_covariance_cell_JAX(map_solution_0.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        return map_solution

    def get_sampling_eta_v1(self, red_cov_approx_matrix, invBtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_x=None, map_random_y=None, suppress_low_modes=True):
        """ !!! Doesn't seem to work properly !!!

            Sampling step 1 : eta maps
            Solve CG for eta term with formulation : eta =  N_c^(-1/2) ( N_c^{1/2} x + C_approx^(1/2) y )
            Or :
                eta = (E (B^t N^{-1} B)^{-1} E) E (B^t N^{-1} B)^{-1} B^t N^{-1/2} ( (E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x + C_approx^(-1/2) y )

            Parameters
            ----------            
            :param red_cov_approx_matrix : covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB : matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            :param BtinvN_sqrt : matrix B^T N^{-1/2}, dimension [component, frequencies, npix]

            :param map_random_x : set of maps 0 with mean and variance 1, which will be used to compute eta, default None (then computed in the routine) ; dimension [nfreq, nstokes, npix]
            :param map_random_y : set of maps 0 with mean and variance 1, which will be used to compute eta, default None (then computed in the routine) ; dimension [nstokes, npix]

            :param suppress_low_modes : if True, suppress low modes in the CG between lmin and lmax, default True
            Returns
            -------
            :return: eta maps [nstokes, npix]
        """
        
        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_approx_matrix, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.npix)


        # Creation of the random maps if they are not given
        jax_key_PNRG, jax_key_PNRG_x = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        # if jnp.size(map_random_x) == 0:
        if map_random_x is None:
            print("Recalculating x !")
            map_random_x = jax.random.normal(jax_key_PNRG_x, shape=(self.number_frequencies,self.nstokes,self.npix))

        jax_key_PNRG, jax_key_PNRG_y = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        if map_random_y is None:
            print("Recalculating y !")
            map_random_y = jax.random.normal(jax_key_PNRG_y, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)

        # Computation of the right hand side member of the equation
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix) # Getting sqrt of the covariance matrix

        # First right member : (E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x    
        first_member = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_x)[0] # Selecting CMB component of the random variable

        # Second right member : C_approx^(1/2) y
        second_member = maps_x_red_covariance_cell_JAX(map_random_y, red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        map_solution_0 = first_member + second_member

        # Getting final solution : N_c^(-1/2) ( N_c^{1/2} x + C_approx^(1/2) y )
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)

        map_solution = jnp.einsum('cfp,cp,sp->fsp', BtinvN_sqrt, invBtinvNB[:,0], map_solution_0)*N_c_inv*jhp.nside2resol(self.nside)
        return map_solution


    def get_fluctuating_term_maps(self, red_cov_matrix, invBtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_realization_xi=None, map_random_realization_chi=None, initial_guess=jnp.empty(0)):
        """ Sampling step 2 : fluctuating term

            Solve fluctuation term with formulation (C^-1 + N^-1) for the left member :
            (C^{-1} + E^t (B^t N^{-1} B)^{-1} E) \zeta = C^{-1/2} xi + (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi

            Parameters
            ----------
            :param red_cov_matrix: term C, covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            :param BtinvN_sqrt: matrix B^T N^{-1/2}, dimension [component, frequencies, npix]

            :param jax_key_PNRG: random key for JAX PNRG
            
            :param map_white_noise_xi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            :param map_white_noise_chi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nfreq, nstokes, npix]

            :param initial_guess: initial guess for the CG, default None (then set to 0) ; dimension [nstokes, npix]

            Returns
            -------
            :return: Fluctuation maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.npix)


        jax_key_PNRG, jax_key_PNRG_xi = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        
        # Creation of the random maps if they are not given
        # if jnp.size(map_random_realization_xi) == 0:
        if map_random_realization_xi is None:
            print("Recalculating xi !")
            # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
            map_random_realization_xi = jax.random.normal(jax_key_PNRG_xi, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)#*mask_to_use
        
        jax_key_PNRG, *jax_key_PNRG_chi = random.split(jax_key_PNRG,self.number_frequencies+1) # Splitting of the random key to generate a new one
        # if jnp.size(map_random_realization_chi) == 0:
        if map_random_realization_chi is None:
            print("Recalculating chi !")
            def fmap(random_key):
                random_map = jax.random.normal(random_key, shape=(self.nstokes,self.npix))#/jhp.nside2resol(nside)
                return self.get_band_limited_maps(random_map)
            map_random_realization_chi = jax.vmap(fmap)(jnp.array(jax_key_PNRG_chi))

        # Computation of the right side member of the CG
        red_inverse_cov_matrix = jnp.linalg.pinv(red_cov_matrix)
        red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_inverse_cov_matrix)

        # First right member : C^{-1/2} xi
        right_member_1 = maps_x_red_covariance_cell_JAX(map_random_realization_xi, red_inv_cov_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # Second right member :
        ## Computation of N_c^{-1/2}
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        ## Computation of N_c^{-1/2} = (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi
        right_member_2 = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]*N_c_inv # [0] for selecting CMB component of the random variable

        right_member = (right_member_1 + right_member_2).ravel()

        # Computation of the left side member of the CG

        # Operator in harmonic domain : C^{-1}
        first_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_inverse_cov_matrix, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        ## Operator in pixel domain : (E^t (B^t N^{-1} B) E)
        def second_term_left(x):
            return x.reshape((self.nstokes,self.npix))*N_c_inv

        # Full operator to inverse : C^{-1} + 1/(E^t (B^t N^{-1} B) E)
        func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()

        # Initial guess for the CG
        # if jnp.size(initial_guess) == 0:
        if initial_guess.size == 0:
            initial_guess = jnp.zeros_like(map_random_realization_xi)

        # Actual start of the CG
        fluctuating_map, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
                                                                    tol=self.tolerance_CG, atol=self.atol_CG, maxiter=self.limit_iter_cg)
        print("CG-Python-0 Fluct finished in ", number_iterations, "iterations !!")

        return fluctuating_map.reshape((self.nstokes, self.npix))


    def solve_generalized_wiener_filter_term(self, s_cML, red_cov_matrix, invBtinvNB, initial_guess=jnp.empty(0)):
        """ 
            Solve Wiener filter term with CG : (C^{-1} + N_c^-1) s_c,WF = N_c^{-1} s_c,ML

            Parameters
            ----------
            :param s_cML: Maximum Likelihood solution of component separation from input frequency maps ; dimensions [nstokes, npix]
            :param red_cov_matrix: covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            
            :param initial_guess: initial guess for the CG, default None (then set to 0) ; dimension [nstokes, npix]

            Returns
            -------
            :return: Wiener filter maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        if self.nstokes != 1:
            chx.assert_axis_dimension(s_cML, 0, self.nstokes)
            chx.assert_axis_dimension(s_cML, 1, self.npix)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)

        # Computation of the right side member of the CG : N_c^{-1} s_c,ML
        ## First, comutation of N_c^{-1} (putting it to 0 outside the mask)
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        ## Then, computation of N_c^{-1} s_c,ML
        right_member = (s_cML*N_c_inv).ravel()

        # Preparation of the harmonic operator C^{-1} for the LHS of the CG
        first_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        
        ## Second left member pixel operator : (E^t (B^t N^{-1} B)^{-1} E) x
        def second_term_left(x, number_component=self.number_components):
            return x*N_c_inv_repeat

        # Full operator to inverse : C^{-1} + (E^t (B^t N^{-1} B) E)
        func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()

        # Initial guess for the CG
        # if jnp.size(initial_guess) == 0:
        if initial_guess.size == 0:
            initial_guess = jnp.zeros_like(s_cML)

        # Actual start of the CG
        wiener_filter_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.atol_CG, maxiter=self.limit_iter_cg)
        print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")

        return wiener_filter_term.reshape((self.nstokes, self.npix))

    def get_fluctuating_term_maps_v2(self, red_cov_matrix, invBtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_realization_xi=None, map_random_realization_chi=None, initial_guess=jnp.empty(0)):
        """ Sampling step 2 : fluctuating term

            Solve fluctuation term with formulation (C^-1 + N^-1) for the left member :
            (C^{-1} + E^t (B^t N^{-1} B)^{-1} E) \zeta = C^{-1/2} xi + (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi

            Parameters
            ----------
            :param red_cov_matrix: term C, covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            :param BtinvN_sqrt: matrix B^T N^{-1/2}, dimension [component, frequencies, npix]

            :param jax_key_PNRG: random key for JAX PNRG
            
            :param map_white_noise_xi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            :param map_white_noise_chi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nfreq, nstokes, npix]

            :param initial_guess: initial guess for the CG, default None (then set to 0) ; dimension [nstokes, npix]

            Returns
            -------
            :return: Fluctuation maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.npix)


        jax_key_PNRG, jax_key_PNRG_xi = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        
        # Creation of the random maps if they are not given
        # if jnp.size(map_random_realization_xi) == 0:
        if map_random_realization_xi is None:
            print("Recalculating xi !")
            # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
            map_random_realization_xi = jax.random.normal(jax_key_PNRG_xi, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)#*mask_to_use

        jax_key_PNRG, *jax_key_PNRG_chi = random.split(jax_key_PNRG,self.number_frequencies+1) # Splitting of the random key to generate a new one
        # if jnp.size(map_random_realization_chi) == 0:
        if map_random_realization_chi is None:
            print("Recalculating chi !")
            def fmap(random_key):
                random_map = jax.random.normal(random_key, shape=(self.nstokes,self.npix))#/jhp.nside2resol(nside)
                return self.get_band_limited_maps(random_map)
            map_random_realization_chi = jax.vmap(fmap)(jnp.array(jax_key_PNRG_chi))

        # Computation of the right side member of the CG
        # red_inverse_cov_matrix = jnp.linalg.pinv(red_cov_matrix)
        # red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_inverse_cov_matrix)
        red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_matrix)

        # First right member : C^{-1/2} xi
        # right_member_1 = maps_x_red_covariance_cell_JAX(map_random_realization_xi, red_inv_cov_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        right_member_1 = map_random_realization_xi

        # Second right member :
        ## Computation of N_c^{-1/2}
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        ## Computation of N_c^{-1/2} = (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi
        right_member_2_part = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]*N_c_inv # [0] for selecting CMB component of the random variable
        right_member_2 = maps_x_red_covariance_cell_JAX(right_member_2_part, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        right_member = (right_member_1 + right_member_2).ravel()

        # Computation of the left side member of the CG

        # Operator in harmonic domain : C^{-1}
        first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        ## Operator in pixel domain : (E^t (B^t N^{-1} B) E)
        def second_part_term_left(x):
            return x.reshape((self.nstokes,self.npix))*N_c_inv

        third_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        transform_func = lambda x : third_part_term_left(second_part_term_left(first_part_term_left(x)))

        # Full operator to inverse : C^{-1} + 1/(E^t (B^t N^{-1} B) E)
        # func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()
        func_left_term = lambda x : x.ravel() + transform_func(x).ravel()

        # Initial guess for the CG
        # if jnp.size(initial_guess) == 0:
        if initial_guess.size == 0:
            initial_guess = jnp.zeros_like(map_random_realization_xi)
        else:
            initial_guess = maps_x_red_covariance_cell_JAX(initial_guess.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # Actual start of the CG
        # fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
        #                                                             tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)

        fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
                                                                    tol=self.tolerance_CG, maxiter=self.limit_iter_cg)    
        print("CG-Python-0 Fluct finished in ", number_iterations, "iterations !!")

        fluctuating_map = maps_x_red_covariance_cell_JAX(fluctuating_map_z.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        return fluctuating_map.reshape((self.nstokes, self.npix))


    def solve_generalized_wiener_filter_term_v2(self, s_cML, red_cov_matrix, invBtinvNB, initial_guess=jnp.empty(0)):
        """ 
            Solve Wiener filter term with CG : (C^{-1} + N_c^-1) s_c,WF = N_c^{-1} s_c,ML

            Parameters
            ----------
            :param s_cML: Maximum Likelihood solution of component separation from input frequency maps ; dimensions [nstokes, npix]
            :param red_cov_matrix: covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            
            :param initial_guess: initial guess for the CG, default None (then set to 0) ; dimension [nstokes, npix]

            Returns
            -------
            :return: Wiener filter maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        if self.nstokes != 1:
            chx.assert_axis_dimension(s_cML, 0, self.nstokes)
            chx.assert_axis_dimension(s_cML, 1, self.npix)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)

        red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_matrix)

        # Computation of the right side member of the CG : N_c^{-1} s_c,ML
        ## First, comutation of N_c^{-1} (putting it to 0 outside the mask)
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        ## Then, computation of N_c^{-1} s_c,ML
        # right_member = (s_cML*N_c_inv).ravel()
        right_member = maps_x_red_covariance_cell_JAX(s_cML*N_c_inv, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        

        # Preparation of the harmonic operator C^{-1} for the LHS of the CG
        first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        
        ## Second left member pixel operator : (E^t (B^t N^{-1} B)^{-1} E) x
        def second_part_term_left(x):
            return x*N_c_inv_repeat

        third_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        transform_func = lambda x : third_part_term_left(second_part_term_left(first_part_term_left(x)))

        # Full operator to inverse : C^{-1} + (E^t (B^t N^{-1} B) E)
        # func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()
        func_left_term = lambda x : x.ravel() + transform_func(x).ravel()

        # Initial guess for the CG
        # if jnp.size(initial_guess) == 0:
        if initial_guess.size == 0:
            initial_guess = jnp.zeros_like(s_cML)
        else:
            initial_guess = maps_x_red_covariance_cell_JAX(initial_guess.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # Actual start of the CG
        # wiener_filter_term_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)
        wiener_filter_term_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, maxiter=self.limit_iter_cg)
        print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")

        wiener_filter_term = maps_x_red_covariance_cell_JAX(wiener_filter_term_z.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        return wiener_filter_term.reshape((self.nstokes, self.npix))

    def get_fluctuating_term_maps_v2c(self, red_cov_matrix, invBtinvNB, BtinvN_sqrt, jax_key_PNRG, map_random_realization_xi=None, map_random_realization_chi=None, initial_guess=jnp.empty(0)):
        """ Sampling step 2 : fluctuating term

            Solve fluctuation term with formulation (C^-1 + N^-1) for the left member :
            (C^{-1} + E^t (B^t N^{-1} B)^{-1} E) \zeta = C^{-1/2} xi + (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi

            Parameters
            ----------
            :param red_cov_matrix: term C, covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            :param BtinvN_sqrt: matrix B^T N^{-1/2}, dimension [component, frequencies, npix]

            :param jax_key_PNRG: random key for JAX PNRG
            
            :param map_white_noise_xi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
            :param map_white_noise_chi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nfreq, nstokes, npix]

            :param initial_guess: initial guess for the CG, default jnp.empty(0) (then set to 0) ; dimension [nstokes, npix]

            Returns
            -------
            :return: Fluctuation maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)
        chx.assert_axis_dimension(BtinvN_sqrt, 1, self.number_frequencies)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.npix)


        jax_key_PNRG, jax_key_PNRG_xi = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one
        
        # Creation of the random maps if they are not given
        if map_random_realization_xi is None:
            print("Recalculating xi !")
            # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
            map_random_realization_xi = jax.random.normal(jax_key_PNRG_xi, shape=(self.nstokes,self.npix))/jhp.nside2resol(self.nside)#*mask_to_use

        jax_key_PNRG, *jax_key_PNRG_chi = random.split(jax_key_PNRG,self.number_frequencies+1) # Splitting of the random key to generate a new one
        if map_random_realization_chi is None:
            print("Recalculating chi !")
            def fmap(random_key):
                random_map = jax.random.normal(random_key, shape=(self.nstokes,self.npix))#/jhp.nside2resol(nside)
                return self.get_band_limited_maps(random_map)
            map_random_realization_chi = jax.vmap(fmap)(jnp.array(jax_key_PNRG_chi))
            chx.assert_shape(map_random_realization_chi, (self.number_frequencies, self.nstokes, self.npix))

        # Computation of the right side member of the CG
        # red_inverse_cov_matrix = jnp.linalg.pinv(red_cov_matrix)
        # red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_inverse_cov_matrix)
        red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_matrix)

        # First right member : C^{-1/2} xi
        # right_member_1 = maps_x_red_covariance_cell_JAX(map_random_realization_xi, red_inv_cov_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        right_member_1 = map_random_realization_xi

        # Second right member :
        ## Computation of N_c^{-1/2}
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        ## Computation of N_c^{-1/2} = (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi
        right_member_2_part = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]*N_c_inv # [0] for selecting CMB component of the random variable
        right_member_2 = maps_x_red_covariance_cell_JAX(right_member_2_part, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        right_member = (right_member_1 + right_member_2).ravel()

        # Computation of the left side member of the CG

        # Operator in harmonic domain : C^{-1}
        first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        ## Operator in pixel domain : (E^t (B^t N^{-1} B) E)
        def second_part_term_left(x):
            # return x.reshape((self.nstokes,self.npix))*N_c_inv
            return x*N_c_inv_repeat

        # third_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        transform_func = lambda x : first_part_term_left(second_part_term_left(first_part_term_left(x).ravel()))

        # Full operator to inverse : C^{-1} + 1/(E^t (B^t N^{-1} B) E)
        # func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()
        func_left_term = lambda x : x.ravel() + transform_func(x).ravel()

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(map_random_realization_xi)
        # else:
        #     # initial_guess = maps_x_red_covariance_cell_JAX(initial_guess.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        #     initial_guess = maps_x_red_covariance_cell_JAX(initial_guess.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        # # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        # ## Second left member pixel operator : (E^t (B^t N^{-1} B)^{-1} E) x
        # def inv_second_part_term_left(x):
        #     return jnp.einsum('sp,p->sp',x.reshape((self.nstokes,self.npix)),invBtinvNB[0,0,...]*jhp.nside2resol(self.nside)**2)
        # def inv_second_part_term_left(x):
        #     return jnp.einsum('sp,p->sp',x.reshape((self.nstokes,self.npix)),N_c_inv)

        # precond_func = lambda x : inv_first_part_term_left(x).ravel() + inv_second_part_term_left(x).ravel()
        # precond_func = lambda x : inv_second_part_term_left(x).ravel()
        # precond_func = lambda x : inv_first_part_term_left(x).ravel()
        # precond_func = lambda x : x.ravel() -  inv_first_part_term_left(inv_second_part_term_left(inv_first_part_term_left(x))).ravel()/(N_c_inv).sum()

        # Actual start of the CG
        # fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
        #                                                             tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)

        func_lineax = lx.FunctionLinearOperator(func_left_term, jax.ShapeDtypeStruct((self.nstokes*self.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))

        # mask_binary = jnp.copy(self.mask)
        # if mask_binary is None:
        #     mask_binary = jnp.ones_like(self.mask)
        # mask_to_use = jnp.repeat(mask_binary[:].ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()
        # Maybe try directly the ravel order='F' ?

        # func_norm = lambda x : jnp.sqrt(jnp.sum((x.reshape((self.nstokes,self.npix))*self.mask)**2))
        # func_norm = lambda x : jnp.sqrt(jnp.sum((x*mask_to_use)**2))
        # func_norm = lambda x : jnp.linalg.norm((x.reshape((self.nstokes,self.npix))*self.mask).ravel(),ord=2)
        func_norm = lambda x : jnp.linalg.norm(x,ord=2)
        # func_norm = lambda x : jnp.linalg.norm((x*mask_to_use).ravel(),ord=2)

        # func_norm = lambda x : jnp.sqrt(jnp.sum((x.reshape((self.nstokes,self.npix)))**2))


        # precond_lineax = lx.FunctionLinearOperator(precond_func, jax.ShapeDtypeStruct((self.nstokes*self.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))

        CG_obj = lx.CG(rtol=self.tolerance_CG, atol=self.atol_CG, max_steps=self.limit_iter_cg, norm=func_norm)

        time_start = time.time()
        # fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
        #                                                             tol=self.tolerance_CG, maxiter=self.limit_iter_cg, M=precond_func)
        # options_dict = {"y0":initial_guess.ravel(), "preconditioner":precond_lineax}
        options_dict = {"y0":initial_guess.ravel()}

        solution = lx.linear_solve(func_lineax, right_member.ravel(), solver=CG_obj, throw=False, options=options_dict)
        fluctuating_map_z = solution.value
        # fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
        #                                                             tol=self.tolerance_CG, maxiter=self.limit_iter_cg, M=precond_func)    
        print("CG Fluct finished in ", time.time()-time_start, "seconds !!")

        # print("CG-Python-0 Fluct finished in ", number_iterations, "iterations !!")
        print("CG-Python-0 Fluct finished with ", solution.result, solution.stats)

        fluctuating_map = maps_x_red_covariance_cell_JAX(fluctuating_map_z.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        return fluctuating_map.reshape((self.nstokes, self.npix))


    def solve_generalized_wiener_filter_term_v2c(self, s_cML, red_cov_matrix, invBtinvNB, initial_guess=jnp.empty(0)):
        """ 
            Solve Wiener filter term with CG : (C^{-1} + N_c^-1) s_c,WF = N_c^{-1} s_c,ML

            Parameters
            ----------
            :param s_cML: Maximum Likelihood solution of component separation from input frequency maps ; dimensions [nstokes, npix]
            :param red_cov_matrix: covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, npix]
            
            :param initial_guess: initial guess for the CG, default jnp.empty(0) (then set to 0) ; dimension [nstokes, npix]

            Returns
            -------
            :return: Wiener filter maps [nstokes, npix]
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix, 0, self.lmax + 1 - self.lmin)
        if self.nstokes != 1:
            chx.assert_axis_dimension(s_cML, 0, self.nstokes)
            chx.assert_axis_dimension(s_cML, 1, self.npix)
        chx.assert_axis_dimension(invBtinvNB, 2, self.npix)

        red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_matrix)

        # Computation of the right side member of the CG : N_c^{-1} s_c,ML
        ## First, comutation of N_c^{-1} (putting it to 0 outside the mask)
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        N_c_repeat = jnp.repeat((invBtinvNB[0,0]*jhp.nside2resol(self.nside)**2).ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()
        ## Then, computation of N_c^{-1} s_c,ML
        # right_member = (s_cML*N_c_inv).ravel()
        right_member = maps_x_red_covariance_cell_JAX(s_cML*N_c_inv, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        

        # Preparation of the harmonic operator C^{-1} for the LHS of the CG
        first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        
        ## Second left member pixel operator : (E^t (B^t N^{-1} B)^{-1} E) x
        def second_part_term_left(x):
            return x*N_c_inv_repeat
            # return x.reshape((self.nstokes,self.npix))*N_c_inv

        transform_func = lambda x : first_part_term_left(second_part_term_left(first_part_term_left(x).ravel()))

        # Full operator to inverse : C^{-1} + (E^t (B^t N^{-1} B) E)
        # func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()
        func_left_term = lambda x : x.ravel() + transform_func(x).ravel()

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(s_cML)
        # else:
        #     initial_guess = maps_x_red_covariance_cell_JAX(initial_guess.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), jnp.linalg.pinv(red_cov_matrix), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_matrix, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        #         # Second left member pixel operator : (E^t (B^t N^{-1} B)^{-1} E) x
        # def inv_second_part_term_left(x):
        #     # return jnp.einsum('sp,p->sp',x.reshape((self.nstokes,self.npix)),invBtinvNB[0,0,...]*jhp.nside2resol(self.nside)**2)
        #     return x*N_c_repeat
        # # def inv_second_part_term_left(x):
        # #     return jnp.einsum('sp,p->sp',x.reshape((self.nstokes,self.npix)),N_c_inv)

        # precond_func = lambda x : x.ravel() -  inv_first_part_term_left(inv_second_part_term_left(inv_first_part_term_left(x))).ravel()/(N_c_inv).sum()
        # precond_func = lambda x : inv_first_part_term_left(x).ravel() + inv_second_part_term_left(x).ravel()
        # precond_func = lambda x : inv_first_part_term_left(x).ravel() + inv_second_part_term_left(x).ravel()
        # precond_func = lambda x : inv_second_part_term_left(x).ravel()
        # precond_func = lambda x : inv_first_part_term_left(x).ravel()


        # Actual start of the CG
        # wiener_filter_term_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg)
        # wiener_filter_term_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
        #                                                                 tol=self.tolerance_CG, maxiter=self.limit_iter_cg, M=precond_func)
        func_lineax = lx.FunctionLinearOperator(func_left_term, jax.ShapeDtypeStruct((self.nstokes*self.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))

        # precond_lineax = lx.FunctionLinearOperator(precond_func, jax.ShapeDtypeStruct((self.nstokes*self.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))

        # mask_binary = jnp.copy(self.mask)
        # if mask_binary is None:
        #     mask_binary = jnp.ones_like(self.mask)
        # mask_to_use = jnp.repeat(mask_binary[:].ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()


        # func_norm = lambda x : jnp.sqrt(jnp.sum((x.reshape((self.nstokes,self.npix))*self.mask)**2))
        # func_norm = lambda x : jnp.sqrt(jnp.sum((x*mask_to_use)**2))
        # func_norm = lambda x : jnp.linalg.norm(x*mask_to_use,ord=2)
        # func_norm = lambda x : jnp.linalg.norm((x.reshape((self.nstokes,self.npix))*self.mask).ravel(),ord=2)
        func_norm = lambda x : jnp.linalg.norm(x,ord=2)
        # func_norm = lambda x : jnp.linalg.norm((x*mask_to_use).ravel(),ord=2)
        # def func_norm(x, _mask_to_use=mask_to_use):
        #     # return jnp.sqrt(jnp.sum((x*_mask_to_use)**2))
        #     return jnp.sqrt(jnp.sum((x[_mask_to_use!=0])**2))
        # func_norm = lambda x : jnp.sqrt(jnp.sum((x.reshape((self.nstokes,self.npix))**2)))

        CG_obj = lx.CG(rtol=self.tolerance_CG, atol=self.atol_CG, max_steps=self.limit_iter_cg, norm=func_norm)
        # CG_obj = lx.CG(rtol=self.tolerance_CG, atol=1e-8, max_steps=self.limit_iter_cg)

        time_start = time.time()
        # fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), 
        #                                                             tol=self.tolerance_CG, maxiter=self.limit_iter_cg, M=precond_func)
        # options_dict = {"y0":initial_guess.ravel(), "preconditioner":precond_lineax}
        options_dict = {"y0":initial_guess.ravel()}

        solution = lx.linear_solve(func_lineax, right_member.ravel(), solver=CG_obj, throw=False, options=options_dict)
        wiener_filter_term_z = solution.value
        # print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")
        print("CG-Python-0 Fluct finished with ", solution.result, solution.stats)
        
        wiener_filter_term = maps_x_red_covariance_cell_JAX(wiener_filter_term_z.reshape((self.nstokes,self.npix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        return wiener_filter_term.reshape((self.nstokes, self.npix))


    def get_inverse_wishart_sampling_from_c_ells(self, sigma_ell, PRNGKey):
        """ Solve sampling step 3 : inverse Wishart distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
                1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
                2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
                3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [number_correlations, lmax+1]
            :param PRNGKey: random key for JAX PNRG
            
            Returns
            -------
            :return: Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)
        c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        
        sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)

        def map_sampling_Wishart(ell_PNRGKey, ell):
            """ Compute the sampling of the Wishart distribution for a given ell
            """

            sample_gaussian = random.multivariate_normal(ell_PNRGKey, jnp.zeros(self.nstokes), invert_parameter_Wishart[ell], shape=(2*(self.lmax+1) - self.nstokes,))

            weighting = jnp.where(ell >= (jnp.arange(2*(self.lmax+1)-self.nstokes)+self.nstokes)/2, 1, 0)

            sample_to_return = jnp.einsum('lk,l,lm->km',sample_gaussian,weighting,sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return
        
        PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1)) 
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Wishart = sampling_Wishart.at[self.lmin:].set(sampling_Wishart_map)
        return jnp.linalg.pinv(sampling_Wishart)

    def get_conditional_proba_C_from_previous_sample(self, red_sigma_ell, red_cov_matrix_sampled):
        """ Compute log-proba of C parametrized by r_param. The associated log proba is :
                -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

            Parameters
            ----------
            :param r_param: parameter of the covariance C to be sampled
            :param red_sigma_ell: covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param theoretical_red_cov_r1_tensor: tensor mode covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param theoretical_red_cov_r0_total: scalar mode covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: log-proba of C parametrized by r_param
        """
        chx.assert_equal_shape(red_sigma_ell, red_cov_matrix_sampled)

        # Getting determinant of the covariance matrix
        sum_dets = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_cov_matrix_sampled)) ).sum()
        
        return -( jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets)/2


    def get_inverse_wishart_sampling_from_c_ells_v2(self, sigma_ell, PRNGKey, red_cov_ell_2=None):
        """ Solve sampling step 3 : inverse Wishart distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
                1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
                2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
                3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [number_correlations, lmax+1]
            :param PRNGKey: random key for JAX PNRG
            
            Returns
            -------
            :return: Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)
        c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))

        PRNGKey, subkey = random.split(PRNGKey)
        ell_2 = 2
        red_cov_ell_2 
        new_sample_ell_2 = multivariate_Metropolis_Hasting_step(PRNGKey, old_sample, covariance_matrix, self.get_conditional_proba_C_from_previous_sample, **model_kwargs)

        sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)

        def map_sampling_Wishart(ell_PNRGKey, ell):
            """ Compute the sampling of the Wishart distribution for a given ell
            """

            sample_gaussian = random.multivariate_normal(ell_PNRGKey, jnp.zeros(self.nstokes), invert_parameter_Wishart[ell], shape=(2*(self.lmax+1) - self.nstokes,))

            weighting = jnp.where(ell >= (jnp.arange(2*(self.lmax+1)-self.nstokes)+self.nstokes)/2, 1, 0)

            sample_to_return = jnp.einsum('lk,l,lm->km',sample_gaussian,weighting,sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return
        
        min_value_ell = jnp.array([3, self.lmin]).max()
        PRNGKey_map = random.split(PRNGKey, self.lmax-min_value_ell+1) # Prepare lmax+1-lmin PRNGKeys to be used
        sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(min_value_ell,self.lmax+1)) 
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Wishart = sampling_Wishart.at[min_value_ell:].set(sampling_Wishart_map)
        return jnp.linalg.pinv(sampling_Wishart)

    def get_inverse_gamma_sampling_from_c_ells(self, sigma_ell, PRNGKey):
        """ Solve sampling step 3 : inverse Gamma distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Gamma (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Gamma distribution ; must be of dimension [number_correlations, lmax+1]
            :param PRNGKey: random key for JAX PNRG
            
            Returns
            -------
            :return: Matrices following an inverse Gamma distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)

        # c_ells_gamma_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)

        c_ells_gamma_modified = jnp.zeros_like(sigma_ell)
        c_ells_gamma_modified = c_ells_gamma_modified.at[:self.nstokes].set(jnp.copy(sigma_ell[:self.nstokes])*(2*jnp.arange(self.lmax+1) + 1))
        red_c_ell_Gamma_b_factor = get_reduced_matrix_from_c_ell_jax(c_ells_gamma_modified)/2

        
        sampling_Gamma = jnp.zeros_like(red_c_ell_Gamma_b_factor)

        def map_sampling_Gamma(ell_PNRGKey, ell):
            """ Compute the sampling of the Wishart distribution for a given ell
            """
            return jnp.diag(jnp.diag(red_c_ell_Gamma_b_factor[ell])/random.gamma(ell_PNRGKey, a=(2*ell+1-2)/2, shape=(1,)))

        PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        sampling_Gamma_map = jax.vmap(map_sampling_Gamma)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1)) 
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Gamma = sampling_Gamma.at[self.lmin:].set(sampling_Gamma_map)
        return sampling_Gamma


    def get_conditional_proba_C_from_r(self, r_param, red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total):
        """ Compute log-proba of C parametrized by r_param. The associated log proba is :
                -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

            Parameters
            ----------
            :param r_param: parameter of the covariance C to be sampled
            :param red_sigma_ell: covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param theoretical_red_cov_r1_tensor: tensor mode covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param theoretical_red_cov_r0_total: scalar mode covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: log-proba of C parametrized by r_param
        """

        # Getting the covariance matrix parametrized by r_param
        red_cov_matrix_sampled = r_param * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        # Getting determinant of the covariance matrix
        sum_dets = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_cov_matrix_sampled)) ).sum()
        
        return -( jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets)/2

    def get_conditional_proba_spectral_likelihood_JAX(self, complete_mixing_matrix, full_data_without_CMB, suppress_low_modes=True):
        """ Get conditional probability of spectral likelihood from the full mixing matrix

            The associated conditional probability is given by : 
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

            Parameters
            ----------
            :param complete_mixing_matrix: mixing matrix of dimension [component, frequencies]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, npix]

            :param suppress_low_modes: if True, suppress low modes of full_data_without_CMB in the CG between lmin and lmax, default True

            Returns
            -------
            :return: computation of spectral likelihood
        """

        # Building the spectral_likelihood : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

        ## Getting B_fg, mixing matrix part for foregrounds
        complete_mixing_matrix_fg = complete_mixing_matrix[:,1:]

        invBtinvNB_fg = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)
        BtinvN_fg = get_BtinvN(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)

        ## Prepraring the data without CMB
        full_data_without_CMB_with_noise = jnp.einsum('cfp,fsp->csp', BtinvN_fg, full_data_without_CMB)
        if suppress_low_modes:
            def fmap(index):
                return self.get_band_limited_maps(full_data_without_CMB_with_noise[index])
            full_data_without_CMB_with_noise = jax.vmap(fmap)(jnp.arange(self.number_components-1))

        ## Computation of the spectral likelihood
        first_term_complete = jnp.einsum('psc,cmp,msp', full_data_without_CMB_with_noise.T, invBtinvNB_fg, full_data_without_CMB_with_noise)
        return -(-first_term_complete + 0)/2.


    def get_conditional_proba_correction_likelihood_JAX_v2(self, complete_mixing_matrix, component_eta_maps, red_cov_approx_matrix):
        """ Get conditional probability of correction term in the likelihood from the full mixing matrix

            The associated conditional probability is given by : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            Or :
            - (eta^t C_approx^{-1/2} ( C_approx^{-1} + (E^t (B^t N^{-1} B)^{-1} E) ^{-1}) C_approx^{-1/2} \eta

            Parameters
            ----------
            :param complete_mixing_matrix: full mixing matrix of dimension [component, frequencies]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of correction term to the likelihood
        """

        # Building the correction term to the likelihood : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} \eta

        ## Preparing the mixing matrix and C_approx^{-1/2}
        invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2
        red_cov_approx_matrix_msqrt = jnp.linalg.pinv(get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix))

        ## Computation of C_approx^{-1/2} eta
        component_eta_maps_2 = maps_x_red_covariance_cell_JAX(component_eta_maps, red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        ## Preparing the operator ( C_approx^{-1} + N_c^{-1} )^{-1}
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        # N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0])

        operator_harmonic = jnp.linalg.pinv(red_cov_approx_matrix)
        operator_pixel = N_c_inv
        
        first_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), operator_harmonic, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        def second_term_left(x):
            return x.reshape((self.nstokes,self.npix))*operator_pixel

        func_left_term = lambda x : (first_term_left(x) + second_term_left(x)).ravel()

        ## Computation of ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        initial_guess = jnp.zeros((self.nstokes,self.npix))
        right_member = jnp.copy(component_eta_maps_2)
        inverse_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.atol_CG, maxiter=self.limit_iter_cg)

        ## Computation of C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        component_eta_maps_3 = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.npix), red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # And finally eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        second_term_complete = jnp.einsum('sk,sk', component_eta_maps, component_eta_maps_3)
        return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2

    def get_conditional_proba_correction_likelihood_JAX_v2b(self, complete_mixing_matrix, component_eta_maps, red_cov_approx_matrix, previous_inverse=jnp.empty(0), return_inverse=False):
        """ Get conditional probability of correction term in the likelihood from the full mixing matrix

            The associated conditional probability is given by : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            Or :
            - (eta^t C_approx^{-1/2} ( C_approx^{-1} + (E^t (B^t N^{-1} B)^{-1} E) ^{-1}) C_approx^{-1/2} \eta

            Parameters
            ----------
            :param complete_mixing_matrix: full mixing matrix of dimension [component, frequencies]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of correction term to the likelihood
        """

        # Building the correction term to the likelihood : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} \eta

        ## Preparing the mixing matrix and C_approx^{-1/2}
        invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2        
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)

        ## Preparing the operator ( C_approx^{-1} + N_c^{-1} )^{-1}
        N_c_inv = jnp.zeros_like(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0])
        
        first_part_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        
        def second_part_left(x):
            return x.reshape((self.nstokes,self.npix))*N_c_inv

        third_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        func_left_term = lambda x : x.ravel() + third_part_term_left(second_part_left(first_part_left(x))).ravel()

        ## Computation of ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        initial_guess = jnp.copy(component_eta_maps)

        if previous_inverse.size != 0:
            initial_guess = jnp.copy(previous_inverse)

        right_member = jnp.copy(component_eta_maps)
        inverse_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.atol_CG, maxiter=self.limit_iter_cg_eta)

        ## Computation of C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        # component_eta_maps_3 = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.npix), red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # And finally eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        # second_term_complete = jnp.einsum('sk,sk', component_eta_maps, inverse_term.reshape(self.nstokes,self.npix))
        if self.restrict_to_mask:
            central_term = self.mask
        else:
            central_term = jnp.ones_like(self.mask)
        second_term_complete = jnp.einsum('sp,p,sp', component_eta_maps, central_term, inverse_term.reshape(self.nstokes,self.npix))
        if return_inverse:
            return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2, inverse_term.reshape(self.nstokes,self.npix)
        return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2

    def get_conditional_proba_correction_likelihood_JAX_v2c(self, complete_mixing_matrix, component_eta_maps, red_cov_approx_matrix, previous_inverse=jnp.empty(0), return_inverse=False):
        """ Get conditional probability of correction term in the likelihood from the full mixing matrix

            The associated conditional probability is given by : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            Or :
            - (eta^t C_approx^{-1/2} ( C_approx^{-1} + (E^t (B^t N^{-1} B)^{-1} E) ^{-1}) C_approx^{-1/2} \eta

            Parameters
            ----------
            :param complete_mixing_matrix: full mixing matrix of dimension [component, frequencies]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of correction term to the likelihood
        """

        # Building the correction term to the likelihood : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} \eta

        ## Preparing the mixing matrix and C_approx^{-1/2}
        invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2        
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)
        red_cov_approx_matrix_msqrt = jnp.linalg.pinv(red_cov_approx_matrix_sqrt)

        ## Preparing the operator ( C_approx^{-1} + N_c^{-1} )^{-1}
        N_c_inv = jnp.zeros_like(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0])
        N_c_inv_repeat = jnp.repeat(N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        N_c_repeat = jnp.repeat(invBtinvNB[0,0].ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()
        
        first_part_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        
        def second_part_left(x):
            # return x.reshape((self.nstokes,self.npix))*N_c_inv
            return x*N_c_inv_repeat

        # inv_first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        func_left_term = lambda x : x.ravel() + first_part_left(second_part_left(first_part_left(x))).ravel()

        ## Computation of ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        initial_guess = jnp.copy(component_eta_maps)

        if previous_inverse.size != 0:
            initial_guess = jnp.copy(previous_inverse)

        right_member = jnp.copy(component_eta_maps)
    
        func_lineax = lx.FunctionLinearOperator(func_left_term, jax.ShapeDtypeStruct((self.nstokes*self.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))

        # mask_binary = jnp.copy(self.mask)
        # if mask_binary is None:
        #     mask_binary = jnp.ones_like(self.mask)
        # mask_to_use = jnp.repeat(mask_binary[:].ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        # func_norm = lambda x : jnp.linalg.norm((x.reshape((self.nstokes,self.npix))*self.mask).ravel(),ord=2)
        # func_norm = lambda x : jnp.linalg.norm((x*mask_to_use).ravel(),ord=2)
        # func_norm = lambda x : jnp.sqrt(jnp.sum((x.reshape((self.nstokes,self.npix)))**2))
        func_norm = lambda x : jnp.linalg.norm(x,ord=2)

        # def inv_sqrt_second_part_left(x):
        #     # return x.reshape((self.nstokes,self.npix))*N_c_inv
        #     return x*jnp.sqrt(N_c_repeat)

        # precond_func = lambda x : x.ravel() - third_part_term_left(second_part_left(first_part_left(x))).ravel()
        # precond_func = lambda x : inv_sqrt_second_part_left(inv_first_part_term_left(x).ravel()).ravel()

        # precond_lineax = lx.FunctionLinearOperator(precond_func, jax.ShapeDtypeStruct((self.nstokes*self.npix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))

        CG_obj = lx.CG(rtol=self.tolerance_CG, atol=self.atol_CG, max_steps=self.limit_iter_cg_eta, norm=func_norm)

        time_start = time.time()
        options_dict = {"y0":initial_guess.ravel()}
        # options_dict = {"y0":initial_guess.ravel(), "precond":precond_lineax}
        
        # inverse_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=self.tolerance_CG, atol=self.tolerance_CG, maxiter=self.limit_iter_cg_eta)
        solution = lx.linear_solve(func_lineax, right_member.ravel(), solver=CG_obj, throw=False, options=options_dict)

        inverse_term = solution.value
        print("CG-Python-0 Fluct finished with ", solution.result, solution.stats)
        ## Computation of C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        # component_eta_maps_3 = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.npix), red_cov_approx_matrix_msqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        # And finally eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
        # second_term_complete = jnp.einsum('sk,sk', component_eta_maps, inverse_term.reshape(self.nstokes,self.npix))
        if self.restrict_to_mask:
            central_term = self.mask
        else:
            central_term = jnp.ones_like(self.mask)
        second_term_complete = jnp.einsum('sp,p,sp', component_eta_maps, central_term, inverse_term.reshape(self.nstokes,self.npix))
        if return_inverse:
            return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2, inverse_term.reshape(self.nstokes,self.npix)
        return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2


    def get_conditional_proba_correction_likelihood_JAX_v2d(self, old_params_mixing_matrix, new_params_mixing_matrix, inverse_term, component_eta_maps, red_cov_approx_matrix):
        """ Get conditional probability of correction term in the likelihood from the full mixing matrix

            The associated conditional probability is given by : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            Or :
            - (eta^t C_approx^{-1/2} ( C_approx^{-1} + (E^t (B^t N^{-1} B)^{-1} E) ^{-1}) C_approx^{-1/2} \eta

            Parameters
            ----------
            :param complete_mixing_matrix: full mixing matrix of dimension [component, frequencies]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of correction term to the likelihood
        """

        self._fake_mixing_matrix.update_params(old_params_mixing_matrix, jax_use=True)
        old_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)

        self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape(old_params_mixing_matrix.shape,order='F'), jax_use=True))
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)


        ## Preparing the mixing matrix and C_approx^{-1/2}
        old_invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, old_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2
        new_invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2

        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)

        old_N_c_inv = jnp.zeros_like(old_invBtinvNB[0,0])
        old_N_c_inv = old_N_c_inv.at[...,self.mask!=0].set(1/old_invBtinvNB[0,0,self.mask!=0])
        old_N_c_inv_repeat = jnp.repeat(old_N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        new_N_c_inv = jnp.zeros_like(new_invBtinvNB[0,0])
        new_N_c_inv = new_N_c_inv.at[...,self.mask!=0].set(1/new_invBtinvNB[0,0,self.mask!=0])
        new_N_c_inv_repeat = jnp.repeat(new_N_c_inv.ravel(order='C'), self.nstokes).reshape((self.nstokes,self.npix), order='F').ravel()

        first_part_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.npix)), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        def second_part_left(x):
            return x*(new_N_c_inv_repeat-old_N_c_inv_repeat)

        func_to_apply = lambda x : first_part_left(second_part_left(first_part_left(x))).ravel()

        ## Getting new inverse
        if self.restrict_to_mask:
            central_term = self.mask
        else:
            central_term = jnp.ones_like(self.mask)

        perturbation_term = func_to_apply(inverse_term).reshape(self.nstokes,self.npix)
        previous_inverse_x_eta = inverse_term.reshape(self.nstokes,self.npix)

        # new_log_proba = jnp.einsum('sp,p,sp', component_eta_maps, central_term, previous_inverse_x_eta) - jnp.einsum('sp,p,sp', previous_inverse_x_eta, central_term, perturbation_term)
        new_log_proba = jnp.einsum('sp,p,sp', component_eta_maps - perturbation_term, central_term, previous_inverse_x_eta)
        print("First order :", jnp.einsum('sp,p,sp', component_eta_maps, central_term, previous_inverse_x_eta))
        print("Perturbation :", -jnp.einsum('sp,p,sp', perturbation_term, central_term, previous_inverse_x_eta))

        return -(-0 + new_log_proba)/2.*jhp.nside2resol(self.nside)**2



    def get_conditional_proba_mixing_matrix_v2_JAX(self, new_params_mixing_matrix, full_data_without_CMB, component_eta_maps, red_cov_approx_matrix):
        """ Get conditional probability of the conditional probability associated with the B_f parameters
            
            The associated conditional probability is given by : 
                - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            
            Parameters
            ----------
            :param new_params_mixing_matrix: new B_f parameters of the mixing matrix to compute the log-proba, dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, npix]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of the conditional probability of the mixing matrix
        """

        self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        # Compute spectral likelihood : (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(new_mixing_matrix, jnp.array(full_data_without_CMB))

        # Compute correction term to the likelihood : (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
        log_proba_perturbation_likelihood = self.get_conditional_proba_correction_likelihood_JAX_v2(new_mixing_matrix, component_eta_maps, red_cov_approx_matrix)

        return (log_proba_spectral_likelihood + log_proba_perturbation_likelihood)

    def get_conditional_proba_mixing_matrix_v2b_JAX(self, new_params_mixing_matrix, full_data_without_CMB, component_eta_maps, red_cov_approx_matrix, previous_inverse):
        """ Get conditional probability of the conditional probability associated with the B_f parameters
            
            The associated conditional probability is given by : 
                - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            
            Parameters
            ----------
            :param new_params_mixing_matrix: new B_f parameters of the mixing matrix to compute the log-proba, dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, npix]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of the conditional probability of the mixing matrix
        """

        self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        # Compute spectral likelihood : (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(new_mixing_matrix, jnp.array(full_data_without_CMB))

        # Compute correction term to the likelihood : (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
        # log_proba_perturbation_likelihood, inverse_term = self.get_conditional_proba_correction_likelihood_JAX_v2b(new_mixing_matrix, component_eta_maps, red_cov_approx_matrix,previous_inverse=previous_inverse,return_inverse=True)
        log_proba_perturbation_likelihood, inverse_term = self.get_conditional_proba_correction_likelihood_JAX_v2c(new_mixing_matrix, component_eta_maps, red_cov_approx_matrix,previous_inverse=previous_inverse,return_inverse=True)

        return (log_proba_spectral_likelihood + log_proba_perturbation_likelihood), inverse_term
    
    def get_conditional_proba_mixing_matrix_v3_JAX(self, new_params_mixing_matrix, old_params_mixing_matrix, full_data_without_CMB, component_eta_maps, red_cov_approx_matrix, previous_inverse):
        """ Get conditional probability of the conditional probability associated with the B_f parameters
            
            The associated conditional probability is given by : 
                - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            
            Parameters
            ----------
            :param new_params_mixing_matrix: new B_f parameters of the mixing matrix to compute the log-proba, dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, npix]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of the conditional probability of the mixing matrix
        """

        self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        # Compute spectral likelihood : (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(new_mixing_matrix, jnp.array(full_data_without_CMB))

        # Compute correction term to the likelihood : (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
        # log_proba_perturbation_likelihood, inverse_term = self.get_conditional_proba_correction_likelihood_JAX_v2b(new_mixing_matrix, component_eta_maps, red_cov_approx_matrix,previous_inverse=previous_inverse,return_inverse=True)
        log_proba_perturbation_likelihood = self.get_conditional_proba_correction_likelihood_JAX_v2d(old_params_mixing_matrix, new_params_mixing_matrix, previous_inverse, component_eta_maps, red_cov_approx_matrix)

        return log_proba_spectral_likelihood + log_proba_perturbation_likelihood

 
    def get_biased_conditional_proba_mixing_matrix_v2_JAX(self, new_params_mixing_matrix, full_data_without_CMB, component_eta_maps, red_cov_approx_matrix):
        """ Get biased conditional probability of the conditional probability associated with the B_f parameters, without the correction term

            The associated conditional probability is given by :
                - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
            
            Parameters
            ----------
            :param new_params_mixing_matrix: new B_f parameters of the mixing matrix to be tested, dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, npix]
            :param component_eta_maps: set of eta maps of dimension [component, npix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of the conditional probability of the mixing matrix
        """
        self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        
        # Compute spectral likelihood : (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(jnp.copy(new_mixing_matrix), jnp.array(full_data_without_CMB))

        return log_proba_spectral_likelihood


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

def get_log_pdf_lognormal(x, mean, scale):
    return -(jnp.log(x) - mean)**2/(2*scale**2) - jnp.log(x*scale*jnp.sqrt(2*jnp.pi))

def single_lognormal_Metropolis_Hasting_step(random_PRNGKey, old_sample, step_size, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        goal_mean = jnp.ravel(old_sample,order='F')
        goal_step_size = step_size

        mean_lognormal = 2*jnp.log(goal_mean) - 0.5*jnp.log(goal_step_size**2 + goal_mean**2)
        std_lognormal = jnp.sqrt(jnp.log(1 + goal_step_size**2/goal_mean**2))
        u_proposal = dist.LogNormal(loc=mean_lognormal, scale=std_lognormal).sample(key_proposal)

        proposal_mean_lognormal = 2*jnp.log(u_proposal) - 0.5*jnp.log(goal_step_size**2 + u_proposal**2)
        proposal_std_lognormal = jnp.sqrt(jnp.log(1 + goal_step_size**2/u_proposal**2))

        diff_lognormal = get_log_pdf_lognormal(u_proposal, mean_lognormal, std_lognormal) - get_log_pdf_lognormal(jnp.ravel(old_sample,order='F'), proposal_mean_lognormal, proposal_std_lognormal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(u_proposal, **model_kwargs)) - diff_lognormal
        new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, jnp.ravel(old_sample,order='F'))

        return new_sample.reshape(old_sample.shape,order='F')

def separate_single_MH_step_index(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, **model_kwargs):
    
    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry[0], 3)

        u_proposal = dist.Normal(carry[1][index_Bf], step_size[index_Bf]).sample(key_proposal)
        
        
        proposal_params = jnp.copy(carry[1])
        proposal_params = proposal_params.at[index_Bf].set(u_proposal)

        accept_prob = -(log_proba(carry[1], **model_kwargs) - log_proba(proposal_params, **model_kwargs))
        new_param = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, carry[1][index_Bf])
        
        proposal_params = proposal_params.at[index_Bf].set(new_param)
        return (rng_key, proposal_params), new_param

    carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F')), indexes_Bf)
    new_sample = jnp.copy(jnp.ravel(old_sample,order='F'))
    new_sample = new_sample.at[indexes_Bf].set(new_params)
    latest_PRNGKey = carry[0]
    return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F')

def separate_single_MH_step_index_accelerated(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, previous_inverse, **model_kwargs):
    
    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry[0], 3)

        u_proposal = dist.Normal(carry[1][index_Bf], step_size[index_Bf]).sample(key_proposal)

        old_inverse = carry[2]

        proposal_params = jnp.copy(carry[1])
        proposal_params = proposal_params.at[index_Bf].set(u_proposal)

        accept_prob_0, inverse_term_0 = log_proba(carry[1], previous_inverse=old_inverse, **model_kwargs)
        accept_prob_1, inverse_term_1 = log_proba(proposal_params, previous_inverse=old_inverse, **model_kwargs)
        accept_prob = -(accept_prob_0 - accept_prob_1)
        new_param = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, carry[1][index_Bf])
        new_inverse_term = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, inverse_term_1, inverse_term_0)
        
        proposal_params = proposal_params.at[index_Bf].set(new_param)
        return (rng_key, proposal_params, new_inverse_term), new_param

    carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F'), previous_inverse), indexes_Bf)
    new_sample = jnp.copy(jnp.ravel(old_sample,order='F'))
    new_sample = new_sample.at[indexes_Bf].set(new_params)
    latest_PRNGKey = carry[0]
    new_inverse_term = carry[2]
    return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F'), new_inverse_term


# def log_proba_proposal(new_sample, old_sample, step_size, grad_proba, **model_kwargs):
#     """ Return log proba proposal distribution MALA
#     """
#     return -1/(4*step_size)*jnp.linalg.norm(new_sample - old_sample - step_size*grad_proba(x, **model_kwargs),ord=2)**2

# def get_MALA_step(random_PRNGKey, old_sample, step_size, log_proba, grad_proba, indexes_Bf, **model_kwargs):
#     """ Compute Metropolis-adjusted Langevin (MALA) step for a given log-probability function and its gradient.
#     """

#     rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

#     fluctuation = dist.Normal(jnp.ravel(old_sample,order='F'), jnp.ones(jnp.size(old_sample))).sample(key_proposal)*jnp.sqrt(2*step_size)

#     u_proposal = old_sample + step_size*grad_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) + fluctuation

#     diff_proposal = -(log_proba_proposal(u_proposal, jnp.ravel(old_sample,order='F'), step_size, grad_proba, **model_kwargs) 
#                         - log_proba_proposal(jnp.ravel(old_sample,order='F'), u_proposal, step_size, grad_proba, **model_kwargs))

#     accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(u_proposal, **model_kwargs)) + diff_proposal
#     new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, u_proposal, jnp.ravel(old_sample,order='F'))

#     return new_sample[indexes_Bf].reshape(old_sample.shape,order='F')
