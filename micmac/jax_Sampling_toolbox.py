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


class Sampling_functions(MixingMatrix):
    def __init__(self, nside, lmax, nstokes, 
                 frequency_array, freq_inverse_noise, pos_special_freqs=[0,-1],
                 spv_nodes_b=None,
                 freq_noise_c_ell=None,
                 mask=None,
                 n_components=3, lmin=2,
                 n_iter=8, 
                 limit_iter_cg=200, limit_iter_cg_eta=200, 
                 tolerance_CG=1e-10, atol_CG=1e-8,
                 bin_ell_distribution=None):
        """ Sampling functions object
            Contains all the functions needed for the sampling step of the Gibbs sampler

            Parameters
            ----------
            :param nside: nside of the input maps
            :param lmax: maximum ell to be considered
            :param nstokes: number of Stokes parameters
            :param frequency_array: array of frequencies in GHz
            :param freq_inverse_noise: inverse noise in uK^2, expected to be given with pixel dependency, with dimensions (n_frequencies, n_frequencies, n_pix)
            :param freq_noise_c_ell: optional, harmonic dependency of the noise, with dimensions (n_frequencies, n_frequencies, lmax+1-lmin)
            :param pos_special_freqs: position of the special frequencies for dust and synchrotron (0 and -1 by default)
            :param spv_nodes_b: nodes for the spv function (default None)
            :param mask: mask to be considered in the Gibbs sampling (default None if no mask) ; It is not reapplied to the input frequency maps
            :param n_components: number of components to be considered (default 3 for CMB, synchrotron and dust)
            :param lmin: minimum ell to be considered (default 2)
            :param n_iter: number of iterations for Healpy spherical harmonic computations (default 8)
            :param limit_iter_cg: maximum number of iterations for the CGs of the CMB map sampling (default 2000)
            :param limit_iter_cg_eta: maximum number of iterations for the CG of the eta map sampling (default 200)
            :param tolerance_CG: CG tolerance (default 10**(-10))
            :param atol_CG: CG absolute tolerance (default 10**(-8))
            :param bin_ell_distribution: array of the bounds of the bins if binned Inverse Wishart considered, of size nbins+1 (default None for no binning)
        """

        # Inheritance from MixingMatrix
        super(Sampling_functions,self).__init__(nside=nside, frequency_array=frequency_array, n_components=n_components, 
                                                params=None, pos_special_freqs=pos_special_freqs, spv_nodes_b=spv_nodes_b)

        # Tests parameters
        # self.restrict_to_mask = bool(restrict_to_mask)

        # Problem parameters
        assert freq_inverse_noise.shape == (self.n_frequencies, self.n_frequencies, self.n_pix)
        self.freq_inverse_noise = freq_inverse_noise
        if freq_noise_c_ell is not None:
            assert freq_inverse_noise.shape == (self.n_frequencies, self.n_frequencies, self.lmax+1-self.lmin)
        self.freq_noise_c_ell = freq_noise_c_ell # If noise expected to be white, c_ell of the noise
        assert nstokes == 2 # Focus on polarisation for now
        self.nstokes = int(nstokes)
        self.n_correlations = int(np.ceil(self.nstokes**2/2) + np.floor(self.nstokes/2)) 
        # Number of correlations for the power spectrum, so if nstokes==2, it should 3 corresponding to [EE, BB, EB]
        self.nside = int(nside) # nside of the input maps
        self.lmax = int(lmax) # maximum ell to be considered
        assert lmin >= 2
        self.lmin = int(lmin) # minimum ell to be considered
        if mask is None:
            # If no mask is given, then the mask is set to 1
            self.mask = np.ones(12*self.nside**2)
        else:
            self.mask = mask
        if bin_ell_distribution is None:
            # If no binning, then the bin_ell_distribution is set to the full range of ell
            self.bin_ell_distribution = np.arange(self.lmin, self.lmax+1)
        else:
            self.bin_ell_distribution = bin_ell_distribution # Expects array of the bounds of the bins, of size nbins+1
        self.maximum_number_dof = int(self.bin_ell_distribution[-1]**2 - self.bin_ell_distribution[-2]**2)
        # Maximum number of degrees of freedom for the binned Inverse Wishart distribution


        # CG and harmonic parameters
        self.n_iter = int(n_iter) # Number of iterations for estimation of alms with Healpy
        self.limit_iter_cg = int(limit_iter_cg) # Maximum number of iterations for the CGs used in the CMB maps sampling
        self.limit_iter_cg_eta = float(limit_iter_cg_eta) # Maximum number of iterations for the CG used in the eta sampling
        self.tolerance_CG = float(tolerance_CG) # Tolerance for the different CGs
        self.atol_CG = float(atol_CG) # Absolute tolerance for the different CGs

    @property
    def n_pix(self):
        """ Number of pixels of 1 map, for a given Stokes parameter
        """
        return 12*self.nside**2
    
    @property
    def number_bins(self):
        """ Return number of bins
        """
        return jnp.size(self.bin_ell_distribution)-1

    def number_dof(self, bin_index):
        """ Return number of degrees of freedom for a given bin, for inverse Wishart distribution
        """
        return (self.bin_ell_distribution[bin_index+1])**2 - self.bin_ell_distribution[bin_index]**2
 
    def get_band_limited_maps(self, input_map):
        """ Get band limited maps from input maps between lmin and lmax
            :param input_map: input maps to be band limited ; dimension [nstokes, n_pix]
            :return: band limited maps ; dimension [nstokes, n_pix]
        """

        covariance_unity = jnp.zeros((self.lmax+1-self.lmin,self.nstokes,self.nstokes))
        covariance_unity = covariance_unity.at[:,...].set(jnp.eye(self.nstokes))
        return maps_x_red_covariance_cell_JAX(jnp.copy(input_map), covariance_unity, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
    

    def get_sampling_eta_v2(self, 
                            red_cov_approx_matrix_sqrt, 
                            invBtinvNB, 
                            BtinvN_sqrt, 
                            jax_key_PNRG, 
                            map_random_x=None, 
                            map_random_y=None, 
                            suppress_low_modes=True):
        """ Sampling step 1 : eta maps
            Solve CG for eta term with formulation:
                eta = \tilde{C}^(1/2) N_c^{-1/2} x + y

            Or, fully developped:
                eta = \tilde{C}^(1/2) ( (E (B^t N^{-1} B)^{-1} E) E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x + \tilde{C}^(-1/2) y )
            This computation is performed assuming \tilde{C} (or C_approx) is block diagonal

            Parameters
            ----------
            :param red_cov_approx_matrix_sqrt : matrix square root of the covariance matrice (\tilde{C} or C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB : matrix (B^t N^{-1} B)^{-1}, dimension [component, component, n_pix]
            :param BtinvN_sqrt : matrix B^T N^{-1/2}, dimension [component, frequencies, n_pix]
            :param jax_key_PNRG : random key for JAX PNRG

            :param map_random_x : set of maps 0 with mean and variance 1, which will be used to compute eta, default None (to have them computed in the routine) ; dimension [nfreq, nstokes, n_pix]
            :param map_random_y : set of maps 0 with mean and variance 1, which will be used to compute eta, default None (to have them computed in the routine) ; dimension [nstokes, n_pix]

            :param suppress_low_modes : if True, suppress low modes in the CG between lmin and lmax, default True
            Returns
            -------
            :return: eta maps [nstokes, n_pix]
        """

        # Chex test for arguments
        chx.assert_shape(red_cov_approx_matrix_sqrt, (self.lmax+1-self.lmin, self.nstokes, self.nstokes))
        chx.assert_shape(invBtinvNB, (self.n_components, self.n_components, self.n_pix))
        chx.assert_shape(BtinvN_sqrt, (self.n_components, self.n_frequencies, self.n_pix))

        # Creation of the random maps if they are not given
        jax_key_PNRG, jax_key_PNRG_x = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one

        if map_random_x is None:
            # If no random maps are provided, then it is computed within the routine
            print("Recalculating x !")
            map_random_x = jax.random.normal(jax_key_PNRG_x, shape=(self.n_frequencies,self.nstokes,self.n_pix))#/jhp.nside2resol(nside)

        jax_key_PNRG, jax_key_PNRG_y = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one

        if map_random_y is None:
            # If no random maps are provided, then it is computed within the routine
            print("Recalculating y !")
            map_random_y = jax.random.normal(jax_key_PNRG_y, shape=(self.nstokes,self.n_pix))/jhp.nside2resol(self.nside)#*self.mask

        # Computation of the right hand side member of the equation

        ## First right member N_c^{-1/2} x = (E (B^t N^{-1} B)^{-1} E) E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        first_member = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_x)[0]*N_c_inv # Selecting CMB component of the random variable

        ## Second right member is  C_approx^(1/2) C_approx^(-1/2) y, so no need to compute it

        # Getting final solution : C_approx^(1/2) ( N_c^{-1/2} x + C_approx^(-1/2) y)

        ## First getting the full first member C_approx^(1/2) N_c^{-1/2} x
        full_first_member = maps_x_red_covariance_cell_JAX(first_member, red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

        ## The second member is simply C_approx^(1/2) C_approx^(-1/2) y

        # Retrieving the solution
        map_solution = map_random_y + full_first_member

        # If needed, making it band-limited
        if suppress_low_modes:
            map_solution = self.get_band_limited_maps(map_solution)
        return map_solution

    def get_fluctuating_term_maps_v2d(self, 
                                      red_cov_matrix_sqrt, 
                                      invBtinvNB, 
                                      BtinvN_sqrt, 
                                      jax_key_PNRG, 
                                      map_random_realization_xi=None, 
                                      map_random_realization_chi=None, 
                                      initial_guess=jnp.empty(0),
                                      precond_func=None):
        """ Sampling step 2 : fluctuating term

            Solve fluctuation term:
                (Id + C^{1/2} N_c^{-1} C^{1/2}) C^{-1/2} \zeta = xi + C^{1/2} N_c^{-1/2} \chi

            Or, fully developped:
                (Id + C^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C^{1/2}) \zeta = C^{1/2} C^{-1/2} xi + C^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi

            This ensures:
            <\zeta \zeta^t> = (C^{-1} + N_c^{-1})^{-1}
            < \zeta > = 0

            Note C is assumed to be block diagonal

            Parameters
            ----------
            :param red_cov_matrix_sqrt: term C^{1/2}, matrix square root of CMB covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, n_pix]
            :param BtinvN_sqrt: matrix B^T N^{-1/2}, dimension [component, frequencies, n_pix]
            :param jax_key_PNRG: random key for JAX PNRG
            :param map_white_noise_xi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, n_pix]
            :param map_white_noise_chi: set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nfreq, nstokes, n_pix]
            :param initial_guess: initial guess for the CG, default jnp.empty(0) (then set to 0) ; dimension [nstokes, n_pix]
            :param precond_func: function preconditioner for the CG, default None

            Returns
            -------
            :return: Fluctuation maps [nstokes, n_pix] for s_c sampling
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix_sqrt, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.n_pix)
        chx.assert_axis_dimension(BtinvN_sqrt, 1, self.n_frequencies)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.n_pix)

        jax_key_PNRG, jax_key_PNRG_xi = random.split(jax_key_PNRG) # Splitting of the random key to generate a new one

        # Creation of the random maps if they are not given
        if map_random_realization_xi is None:
            # If no random maps are provided, then it is computed within the routine
            print("Recalculating xi !")
            map_random_realization_xi = jax.random.normal(jax_key_PNRG_xi, shape=(self.nstokes,self.n_pix))/jhp.nside2resol(self.nside)

        jax_key_PNRG, *jax_key_PNRG_chi = random.split(jax_key_PNRG,self.n_frequencies+1) # Splitting of the random key to generate a new one
        if map_random_realization_chi is None:
            # If no random maps are provided, then it is computed within the routine
            print("Recalculating chi !")
            def fmap(random_key):
                random_map = jax.random.normal(random_key, shape=(self.nstokes,self.n_pix))#/jhp.nside2resol(nside)
                # return self.get_band_limited_maps(random_map)
                return random_map
            map_random_realization_chi = jax.vmap(fmap)(jnp.array(jax_key_PNRG_chi)) # Generating a different random Gaussian map for each frequency

            chx.assert_shape(map_random_realization_chi, (self.n_frequencies, self.nstokes, self.n_pix))

        # Computation of the right side member of the CG

        # First right member : xi
        right_member_1 = map_random_realization_xi

        # Second right member :
        ## Computation of C^{1/2} N_c^{-1/2} \chi
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.broadcast_to(N_c_inv, (self.nstokes,self.n_pix)).ravel() 
        # Repeat N_c_inv for each Stokes parameter, for speed-up afterwards
    
        ## Computation of N_c^{-1/2} \chi = (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi
        right_member_2_part = jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]*N_c_inv # [0] for selecting CMB component of the random variable
        # First compute N_c^{-1/2} \chi 
        right_member_2 = maps_x_red_covariance_cell_JAX(right_member_2_part, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
        # Then apply C^{1/2} to N_c^{-1/2} \chi

        # right_member = (right_member_1 + right_member_2).ravel()
        right_member = self.get_band_limited_maps(right_member_1 + right_member_2).ravel()

        # Computation of the left side member of the equation

        # Operator in harmonic domain : C^{1/2}
        first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        ## Operator in pixel domain : (E^t (B^t N^{-1} B) E)^{-1}
        def second_part_term_left(x):
            return x*N_c_inv_repeat

        ## Defining the function to inverse with the CG
        transform_func = lambda x : first_part_term_left(second_part_term_left(first_part_term_left(x)))
        func_left_term = lambda x : x.ravel() + transform_func(x).ravel()

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(map_random_realization_xi)

        # Actual start of the CG
        # func_lineax = lx.FunctionLinearOperator(func_left_term, jax.ShapeDtypeStruct((self.nstokes*self.n_pix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))
        # func_norm = lambda x : jnp.linalg.norm(x,ord=2)
        # CG_obj = lx.CG(rtol=self.tolerance_CG, atol=self.atol_CG, max_steps=self.limit_iter_cg, norm=func_norm)
        # options_dict = {"y0":initial_guess.ravel(), "preconditioner":precond_lineax}
        
        # Start of the CG
        time_start = time.time()
        fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, 
                                                                    right_member.ravel(), 
                                                                    x0=initial_guess.ravel(), 
                                                                    tol=self.tolerance_CG,
                                                                    maxiter=self.limit_iter_cg, 
                                                                    M=precond_func)
        # fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, 
        #                                                             right_member.ravel(), 
        #                                                             x0=initial_guess.ravel(), 
        #                                                             tol=self.tolerance_CG,
        #                                                             atol=self.atol_CG,
        #                                                             maxiter=self.limit_iter_cg, 
        #                                                             M=precond_func)
        ## Computing the term C^{-1/2} \zeta

        # solution = lx.linear_solve(func_lineax, right_member.ravel(), solver=CG_obj, throw=False, options=options_dict)
        # fluctuating_map_z = solution.value
        print("CG Fluct finished in ", time.time()-time_start, "seconds !!")

        
        fluctuating_map = maps_x_red_covariance_cell_JAX(fluctuating_map_z.reshape((self.nstokes,self.n_pix)), 
                                                            red_cov_matrix_sqrt, 
                                                            nside=self.nside, 
                                                            lmin=self.lmin, 
                                                            n_iter=self.n_iter)

        ## Retrieving \zeta from C^{-1/2} \zeta
        return fluctuating_map.reshape((self.nstokes, self.n_pix))


    def solve_generalized_wiener_filter_term_v2d(self, 
                                                 s_cML, 
                                                 red_cov_matrix_sqrt, 
                                                 invBtinvNB, 
                                                 initial_guess=jnp.empty(0),
                                                 precond_func=None):
        """ 
            Solve Wiener filter term with CG:
                 (Id + C^{1/2} N_c^{-1} C^{1/2}) C^{-1/2} s_{c,WF} = C^{1/2} N_c^{-1} s_{c,ML}

            This ensures:
                s_{c,WF} = (C^{-1} + N_c^{-1})^{-1} N_c^{-1} s_{c,ML}
            Note C is assumed to be block diagonal

            Parameters
            ----------
            :param s_cML: Maximum Likelihood solution of component separation from input frequency maps ; dimensions [nstokes, n_pix]
            :param red_cov_matrix_sqrt: term C^{1/2}, matrix square root of CMB covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param invBtinvNB: matrix (B^t N^{-1} B)^{-1}, dimension [component, component, n_pix]
            :param initial_guess: initial guess for the CG, default jnp.empty(0) (then set to 0) ; dimension [nstokes, n_pix]
            :param precond_func: function preconditioner for the CG, default None

            Returns
            -------
            :return: Wiener filter maps [nstokes, n_pix]
        """

        # Chex test for arguments
        chx.assert_shape(red_cov_matrix_sqrt, (self.lmax+1-self.lmin, self.nstokes, self.nstokes))
        if self.nstokes != 1:
            chx.assert_shape(s_cML, (self.nstokes, self.n_pix))
        chx.assert_shape(invBtinvNB, (self.n_components, self.n_components, self.n_pix))

        # Computation of the right side member of the CG : C^{1/2} N_c^{-1} s_c,ML
        ## First, computation of N_c^{-1} taking into account the mask
        N_c_repeat = jnp.broadcast_to(invBtinvNB[0,0]*jhp.nside2resol(self.nside)**2, (self.nstokes,self.n_pix)).ravel() 
        ## Repeat N_c for each Stokes parameter, for speed-up afterwards
        N_c_inv = jnp.copy(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0]/jhp.nside2resol(self.nside)**2)
        N_c_inv_repeat = jnp.broadcast_to(N_c_inv, (self.nstokes,self.n_pix)).ravel() ## Repeat N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Then, computation of C^{1/2} N_c^{-1} s_c,ML
        right_member = maps_x_red_covariance_cell_JAX(s_cML*N_c_inv, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
        

        # Preparation of the harmonic operator C^{1/2} for the LHS of the CG
        first_part_term_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), 
                                                                         red_cov_matrix_sqrt, 
                                                                         nside=self.nside, 
                                                                         lmin=self.lmin, 
                                                                         n_iter=self.n_iter).ravel()

        ## Second left member pixel operator : (E^t (B^t N^{-1} B)^{-1} E) x
        def second_part_term_left(x):
            return x*N_c_inv_repeat

        transform_func = lambda x : first_part_term_left(second_part_term_left(first_part_term_left(x).ravel()))
        # Full operator to inverse : Id + C^{1/2} N_c^{-1} C^{1/2}
        func_left_term = lambda x : x.ravel() + transform_func(x).ravel()

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(s_cML)


        # Actual start of the CG
        # func_lineax = lx.FunctionLinearOperator(func_left_term, jax.ShapeDtypeStruct((self.nstokes*self.n_pix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))
        # func_norm = lambda x : jnp.linalg.norm(x,ord=2)
        # CG_obj = lx.CG(rtol=self.tolerance_CG, atol=self.atol_CG, max_steps=self.limit_iter_cg, norm=func_norm)
        # # precond_lineax = lx.FunctionLinearOperator(precond_func, jax.ShapeDtypeStruct((self.nstokes*self.n_pix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))
        # options_dict = {"y0":initial_guess.ravel(), "preconditioner":precond_lineax}
        
        time_start = time.time()
        wiener_filter_term_z, number_iterations = jsp.sparse.linalg.cg(func_left_term, 
                                                                       right_member.ravel(), 
                                                                       x0=initial_guess.ravel(), 
                                                                       tol=self.tolerance_CG,
                                                                       atol=self.atol_CG,
                                                                       maxiter=self.limit_iter_cg, 
                                                                       M=precond_func)
        ## Computing the term C^{-1/2} s_{c,WF}

        # solution = lx.linear_solve(func_lineax, right_member.ravel(), solver=CG_obj, throw=False, options=options_dict)
        # wiener_filter_term_z = solution.value
        # print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")
        # print("CG-Python-0 Fluct finished with ", solution.result, solution.stats)
        print("CG WF finished in ", time.time()-time_start, "seconds !!")

        wiener_filter_term = maps_x_red_covariance_cell_JAX(wiener_filter_term_z.reshape((self.nstokes,self.n_pix)), 
                                                            red_cov_matrix_sqrt, 
                                                            nside=self.nside, 
                                                            lmin=self.lmin, 
                                                            n_iter=self.n_iter)
        ## Retrieving s_{c,WF} from C^{-1/2} s_{c,WF}

        return wiener_filter_term.reshape((self.nstokes, self.n_pix))

    def get_inverse_wishart_sampling_from_c_ells(self, sigma_ell, PRNGKey, old_sample=None, acceptance_posdef=False):
        """ Solve sampling step 3 : inverse Wishart distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
                1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
                2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
                3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [n_correlations, lmax+1]
            :param PRNGKey: random key for JAX PNRG
            
            Returns
            -------
            :return: Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1-self.lmin)
        c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmin,self.lmax+1) + 1)
        invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        
        sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)

        def map_sampling_Wishart(ell_PNRGKey, ell):
            """ Compute the sampling of the Wishart distribution for a given ell
            """

            sample_gaussian = random.multivariate_normal(ell_PNRGKey, jnp.zeros(self.nstokes), invert_parameter_Wishart[ell], shape=(2*self.lmax - self.nstokes,))

            weighting = jnp.where(ell >= (jnp.arange(2*self.lmax-self.nstokes)+self.nstokes)/2, 1, 0)

            sample_to_return = jnp.einsum('lk,l,lm->km',sample_gaussian,weighting,sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return
        
        PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        # sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        sampling_Wishart = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        # sampling_Wishart = sampling_Wishart.at[self.lmin:].set(sampling_Wishart_map)
        # return jnp.linalg.pinv(sampling_Wishart)
        reconstructed_spectra = jnp.linalg.pinv(sampling_Wishart)
        new_sample = reconstructed_spectra
        if acceptance_posdef and old_sample is not None:
            print("Only positive definite matrices are accepted for inv Wishart !")    
            eigen_prod = jnp.prod(jnp.linalg.eigvalsh(reconstructed_spectra), axis=(1))
            acceptance = jnp.where(eigen_prod<0, 0, 1)
            acceptance_reversed = jnp.where(eigen_prod<0, 1, 0)
            # new_sample = jnp.copy(old_sample)
            # new_sample = new_sample.at[acceptance==1,...].set(reconstructed_spectra[acceptance==1,...])
            new_sample = jnp.einsum('lik,l->lik', reconstructed_spectra, acceptance) + jnp.einsum('lik,l->lik', old_sample, acceptance_reversed)
        return new_sample

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
        chx.assert_equal_shape((red_sigma_ell, red_cov_matrix_sampled))

        # Getting determinant of the covariance matrix
        sum_dets = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_cov_matrix_sampled)) ).sum()
        
        return -( jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets)/2

    def get_inverse_gamma_sampling_from_c_ells(self, sigma_ell, PRNGKey):
        """ Solve sampling step 3 : inverse Gamma distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Gamma (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Gamma distribution ; must be of dimension [n_correlations, lmax+1]
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
            """ Compute the sampling of the Gamma distribution for a given ell
            """
            return jnp.diag(jnp.diag(red_c_ell_Gamma_b_factor[ell])/random.gamma(ell_PNRGKey, a=(2*ell+1-2)/2, shape=(1,)))

        PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        sampling_Gamma_map = jax.vmap(map_sampling_Gamma)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1)) 
        # Map over PRNGKeys and ells to create samples of the Gamma distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Gamma = sampling_Gamma.at[self.lmin:].set(sampling_Gamma_map)
        return sampling_Gamma

    def get_binned_red_c_ells_v2(self, red_c_ells_to_bin):
        """ Bin the power spectrum to get the binned power spectrum
            
                Parameters
                ----------
                :param red_c_ells_to_bin: power spectrum to bin ; must be of dimension [lmax+1, nstokes, nstokes]

                Returns
                -------
                :return: Binned power spectrum, of dimension [number_bins, nstokes, nstokes]
        """
        chx.assert_axis_dimension(red_c_ells_to_bin, 0, self.lmax+1 - self.lmin)

        ell_distribution = jnp.arange(red_c_ells_to_bin.shape[0]) + self.lmin

        # number_bins = self.bin_ell_distribution.shape[0]-1

        def map_binned_red_c_ells(bin_ell):
            """ Compute the binned power spectrum for a given ell
            """
            cond = jnp.logical_and(self.bin_ell_distribution[bin_ell]<=ell_distribution, self.bin_ell_distribution[bin_ell+1]>ell_distribution)
            cond_quantitative = jnp.where(cond, 1, 0)
            modes_to_bin = jnp.einsum('lij, l, l->lij', red_c_ells_to_bin, cond_quantitative, ell_distribution*(ell_distribution+1))

            return modes_to_bin.sum(axis=0)#/(self.bin_ell_distribution[bin_ell+1]-self.bin_ell_distribution[bin_ell])

        binned_red_c_ells = jax.vmap(map_binned_red_c_ells)(jnp.arange(self.number_bins))
        return binned_red_c_ells

    def get_binned_inverse_wishart_sampling_from_c_ells_v2(self, sigma_ell, PRNGKey, old_sample=None, acceptance_posdef=False):
        """ Solve sampling step 3 : inverse Wishart distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
                1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
                2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
                3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [n_correlations, lmax+1]
            :param PRNGKey: random key for JAX PNRG
            
            Returns
            -------
            :return: Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        # chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)
        chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1-self.lmin)
        # c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*(jnp.arange(self.lmax+1-self.lmin)+self.lmin) + 1)
        # invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        binned_invert_parameter_Wishart = jnp.linalg.pinv(self.get_binned_red_c_ells_v2(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)))

        # sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)
        sampling_Wishart = jnp.zeros_like(binned_invert_parameter_Wishart)
        # number_dof = lambda x: 2*x+1

        def map_sampling_Wishart(ell_PNRGKey, binned_ell):
            """ Compute the sampling of the Wishart distribution for a given ell
            """

            sample_gaussian = random.multivariate_normal(ell_PNRGKey, jnp.zeros(self.nstokes), binned_invert_parameter_Wishart[binned_ell], shape=(self.maximum_number_dof - self.nstokes-1,))

            weighting = jnp.where(self.number_dof(binned_ell)-self.nstokes-1 >= jnp.arange(self.maximum_number_dof-self.nstokes-1), 1, 0)

            sample_to_return = jnp.einsum('lk,l,lm->km',sample_gaussian,weighting,sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return

        # distribution_bins_over_lmin = self.bin_ell_distributed_min[self.bin_ell_distributed_min>=self.lmin]
        distribution_bins_over_lmin = jnp.copy(self.bin_ell_distribution)[:-1]

        # number_bins_over_lmin = distribution_bins_over_lmin.shape[0]
        # number_bins_over_lmin = jnp.size(distribution_bins_over_lmin)

        # PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        PRNGKey_map = random.split(PRNGKey, self.number_bins) # Prepare lmax+1-lmin PRNGKeys to be used

        # sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.number_bins))
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Wishart = sampling_Wishart.at[-self.number_bins:].set(sampling_Wishart_map)
        sampling_invWishart = jnp.linalg.pinv(sampling_Wishart)

        def reconstruct_spectra(ell):
            cond = jnp.logical_and(self.bin_ell_distribution[:-1]<=ell, self.bin_ell_distribution[1:]>ell)
            bin_contribution = jnp.where(cond, 1, 0)
            return jnp.einsum('lij,l->ij',sampling_invWishart,bin_contribution)/(ell*(ell+1))
        
        # reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.lmin,self.lmax+1))
        reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.lmin,self.bin_ell_distribution[-1]))

        if acceptance_posdef and old_sample is not None:
            print("Only positive definite matrices are accepted for inv Wishart !")
            eigen_prod = jnp.prod(jnp.linalg.eigvalsh(reconstructed_spectra), axis=(1))
            acceptance = jnp.where(eigen_prod<0, 0, 1)
            acceptance_reversed = jnp.where(eigen_prod<0, 1, 0)
            # new_sample = jnp.copy(old_sample)
            # new_sample = new_sample.at[acceptance==1,...].set(reconstructed_spectra[acceptance==1,...])
            new_sample = jnp.einsum('lik,l->lik', reconstructed_spectra, acceptance) + jnp.einsum('lik,l->lik', old_sample, acceptance_reversed)

        chx.assert_tree_all_finite(new_sample)
        return new_sample
        # return reconstructed_spectra

    def get_binned_red_c_ells_v3(self, red_c_ells_to_bin):
        """ Bin the power spectrum to get the binned power spectrum
            
                Parameters
                ----------
                :param red_c_ells_to_bin: power spectrum to bin ; must be of dimension [lmax+1, nstokes, nstokes]

                Returns
                -------
                :return: Binned power spectrum, of dimension [number_bins, nstokes, nstokes]
        """
        chx.assert_axis_dimension(red_c_ells_to_bin, 0, self.lmax+1 - self.lmin)

        ell_distribution = jnp.arange(red_c_ells_to_bin.shape[0]) + self.lmin

        # number_bins = self.bin_ell_distribution.shape[0]-1

        def map_binned_red_c_ells(bin_ell):
            """ Compute the binned power spectrum for a given ell
            """
            cond = jnp.logical_and(self.bin_ell_distribution[bin_ell]<=ell_distribution, 
                                    self.bin_ell_distribution[bin_ell+1]>ell_distribution)
            cond_quantitative = jnp.where(cond, 1, 0)
            modes_to_bin = jnp.einsum('lij, l->lij', red_c_ells_to_bin, cond_quantitative)

            return modes_to_bin.sum(axis=0)#/(self.bin_ell_distribution[bin_ell+1]-self.bin_ell_distribution[bin_ell])

        binned_red_c_ells = jax.vmap(map_binned_red_c_ells)(jnp.arange(self.number_bins))
        return binned_red_c_ells
    
    def get_binned_red_c_ells_v4(self, red_c_ells_to_bin):
        """ Bin the power spectrum to get the binned power spectrum
            
                Parameters
                ----------
                :param red_c_ells_to_bin: power spectrum to bin ; must be of dimension [lmax+1, nstokes, nstokes]

                Returns
                -------
                :return: Binned power spectrum, of dimension [number_bins, nstokes, nstokes]
        """
        chx.assert_axis_dimension(red_c_ells_to_bin, 0, self.lmax+1 - self.lmin)

        ell_distribution = jnp.arange(red_c_ells_to_bin.shape[0]) + self.lmin

        # number_bins = self.bin_ell_distribution.shape[0]-1

        def map_binned_red_c_ells(bin_sup, bin_inf):
            """ Compute the binned power spectrum for a given ell
            """
            cond = jnp.logical_and(bin_inf<=ell_distribution, 
                                    bin_sup>ell_distribution)
            cond_quantitative = jnp.where(cond, 1, 0)
            modes_to_bin = jnp.einsum('lij, l->lij', red_c_ells_to_bin, cond_quantitative)

            return modes_to_bin.sum(axis=0)#/(self.bin_ell_distribution[bin_ell+1]-self.bin_ell_distribution[bin_ell])

        binned_red_c_ells = jax.vmap(map_binned_red_c_ells)(self.bin_ell_distribution[1:],self.bin_ell_distribution[:-1])

        chx.assert_axis_dimension(binned_red_c_ells, 0, self.number_bins)
        return binned_red_c_ells

    def get_binned_inverse_wishart_sampling_from_c_ells_v3(self, sigma_ell, PRNGKey, old_sample=None, acceptance_posdef=False):
        """ Solve sampling step 3 : inverse Wishart distribution with C

            sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

            Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
                1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
                2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
                3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

            Also assumes the monopole and dipole to be 0

            Parameters
            ----------
            :param sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [n_correlations, lmax+1]
            :param PRNGKey: random key for JAX PNRG
            
            Returns
            -------
            :return: Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        # chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)
        chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1-self.lmin)
        # c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        c_ells_Wishart_modified = (sigma_ell*(2*jnp.arange(self.lmin, self.lmax+1) + 1))[...,:self.bin_ell_distribution[-1]-self.lmin] #-1 ?
    

        chx.assert_axis_dimension(c_ells_Wishart_modified, 1, self.bin_ell_distribution[-1]-self.lmin) #-1 ?

        # invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        # binned_invert_parameter_Wishart = jnp.linalg.pinv(self.get_binned_red_c_ells_v3(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)))
        binned_invert_parameter_Wishart = jnp.linalg.pinv(self.get_binned_red_c_ells_v4(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)))

        # sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)
        sampling_Wishart = jnp.zeros_like(binned_invert_parameter_Wishart)
        # number_dof = lambda x: 2*x+1

        all_number_dof = jnp.array((self.bin_ell_distribution[1:])**2 - self.bin_ell_distribution[:-1]**2)

        def map_sampling_Wishart(ell_PNRGKey, binned_ell):
            """ Compute the sampling of the Wishart distribution for a given ell
            """

            sample_gaussian = random.multivariate_normal(ell_PNRGKey, 
                                                        jnp.zeros(self.nstokes), 
                                                        binned_invert_parameter_Wishart[binned_ell], 
                                                        shape=(self.maximum_number_dof - self.nstokes-1,))

            # weighting = jnp.where(self.number_dof(binned_ell)-self.nstokes-1 >= jnp.arange(self.maximum_number_dof-self.nstokes-1), 1, 0)
            weighting = jnp.where(all_number_dof[binned_ell]-self.nstokes-1 >= jnp.arange(self.maximum_number_dof-self.nstokes-1), 1, 0)

            sample_to_return = jnp.einsum('lk,l,lm->km',sample_gaussian,weighting,sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return

        # distribution_bins_over_lmin = self.bin_ell_distributed_min[self.bin_ell_distributed_min>=self.lmin]
        distribution_bins_over_lmin = jnp.copy(self.bin_ell_distribution)[:-1]

        # number_bins_over_lmin = distribution_bins_over_lmin.shape[0]
        # number_bins_over_lmin = jnp.size(distribution_bins_over_lmin)

        # PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        PRNGKey_map = random.split(PRNGKey, self.number_bins) # Prepare lmax+1-lmin PRNGKeys to be used

        # sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.number_bins))
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Wishart = sampling_Wishart.at[:].set(sampling_Wishart_map)
        sampling_invWishart = jnp.linalg.pinv(sampling_Wishart)

        def reconstruct_spectra(ell):
            cond = jnp.logical_and(self.bin_ell_distribution[:-1]<=ell, self.bin_ell_distribution[1:]>ell)
            bin_contribution = jnp.where(cond, 1, 0)
            # return jnp.einsum('lij,l->ij',sampling_invWishart,bin_contribution)#/(ell*(ell+1))
            return jnp.einsum('lij,l->ij',sampling_invWishart,bin_contribution)/bin_contribution.sum()
        
        # reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.lmin,self.lmax+1))
        reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.bin_ell_distribution[0],self.bin_ell_distribution[-1])-1)

        new_sample = reconstructed_spectra
        if acceptance_posdef and old_sample is not None:
            print("Only positive definite matrices are accepted for inv Wishart !")
            eigen_prod = jnp.prod(jnp.linalg.eigvalsh(reconstructed_spectra), axis=(1))
            acceptance = jnp.where(eigen_prod<=0, 0, 1)
            acceptance_reversed = jnp.where(eigen_prod<=0, 1, 0)
            # new_sample = jnp.copy(old_sample)
            # new_sample = new_sample.at[acceptance==1,...].set(reconstructed_spectra[acceptance==1,...])
            new_sample = jnp.einsum('lik,l->lik', reconstructed_spectra, acceptance) + jnp.einsum('lik,l->lik', old_sample, acceptance_reversed)

        # chx.assert_tree_all_finite(new_sample)
        return new_sample
    # return reconstructed_spectra


    def get_conditional_proba_C_from_r(self, 
                                       r_param, 
                                       red_sigma_ell, 
                                       theoretical_red_cov_r1_tensor, 
                                       theoretical_red_cov_r0_total):
        """ Compute log-proba of C parametrized by r_param. 
            The parametrisation is given by : C(r) = r * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total
        
            The associated log proba is :
                -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

            Parameters
            ----------
            :param r_param: parameter of the covariance C to be sampled, float
            :param red_sigma_ell: covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param theoretical_red_cov_r1_tensor: tensor mode covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param theoretical_red_cov_r0_total: scalar mode covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: log-proba of C parametrized by r_param
        """
        chx.assert_shape(theoretical_red_cov_r1_tensor, (self.lmax+1-self.lmin, self.nstokes, self.nstokes))
        chx.assert_equal_shape((red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total))
        
        # Getting the CMB covariance matrix parametrized by r_param
        red_cov_matrix_sampled = r_param * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        # Getting determinant of the covariance matrix log det C(r) ; taking into account the factor 2ell+1 for the multiples m
        sum_dets = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_cov_matrix_sampled)) ).sum()

        return -(jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets)/2 # -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)


    def get_conditional_proba_spectral_likelihood_JAX(self, 
                                                      complete_mixing_matrix, 
                                                      full_data_without_CMB):
        """ Get conditional probability of spectral likelihood from the full mixing matrix

            The associated conditional probability is given by : 
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
            
            with d = full_data_without_CMB, B_c = complete_mixing_matrix, B_f = complete_mixing_matrix[:,1:,:]
            d is assumed to be band-limited

            Parameters
            ----------
            :param complete_mixing_matrix: mixing matrix of dimension [component, frequencies]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, n_pix] ; assumed to be band-limited

            Returns
            -------
            :return: computation of spectral likelihood
        """

        # Building the spectral_likelihood : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

        chx.assert_shape(complete_mixing_matrix, (self.n_frequencies, self.n_components, self.n_pix))
        chx.assert_shape(full_data_without_CMB, (self.n_frequencies, self.nstokes, self.n_pix))
        chx.assert_shape(self.freq_inverse_noise, (self.n_frequencies, self.n_frequencies, self.n_pix))

        ## Getting B_fg, foreground part of the mixing matrix
        complete_mixing_matrix_fg = complete_mixing_matrix[:,1:,:]

        # Computing (B_f^t N^{-1} B_f)^{-1}
        invBtinvNB_fg = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)
        # Computing B_f^t N^{-1}
        BtinvN_fg = get_BtinvN(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)

        # Computing B_f^t N^{-1} (d - B_c s_c)
        full_data_without_CMB_with_noise = jnp.einsum('cfp,fsp->csp', BtinvN_fg, full_data_without_CMB)
        chx.assert_shape(full_data_without_CMB_with_noise, (self.n_components-1, self.nstokes, self.n_pix))

        ## Computation of the spectral likelihood : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        first_term_complete = jnp.einsum('csp,cmp,msp', full_data_without_CMB_with_noise, invBtinvNB_fg, full_data_without_CMB_with_noise)
        return -(-first_term_complete + 0)/2.



    def get_conditional_proba_correction_likelihood_JAX_v2d(self, 
                                                            invBtinvNB_, 
                                                            component_eta_maps, 
                                                            red_cov_approx_matrix_sqrt, 
                                                            first_guess=None, 
                                                            return_inverse=False,
                                                            precond_func=None):
        """ Get conditional probability of correction term in the likelihood from the full mixing matrix

            Noting C_approx = \tilde{C}, the associated conditional probability is given by: 
                - (eta ^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} eta
            Or, with the developped version:
                - (eta^t ( Id + C_approx^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C_approx^{1/2}) \eta

            Parameters
            ----------
            :param invBtinvNB_: pixel covariance matrix (B^t N^{-1} B)^{-1}, of dimension [component, component, n_pix]
            :param component_eta_maps: set of eta maps of dimension [component, n_pix]            
            :param red_cov_approx_matrix_sqrt: square root of the approximate covariance matrix, of dimension [component, component, n_pix]
            :param first_guess: initial guess for the CG algorithm, of dimension [component, n_pix]
            :param return_inverse: boolean to return the inverse term or not (default is False)
            :param precond_func: preconditioning function to be used in the CG algorithm

            Returns
            -------
            :return: computation of log-proba correction term to the likelihood
        """

        # Building the correction term to the likelihood : - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} \eta

        ## Preparing the mixing matrix and C_approx^{-1/2}
        invBtinvNB = invBtinvNB_*jhp.nside2resol(self.nside)**2 # Rescaling the pixel covariance matrix into a weighting

        ## Preparing the operator ( C_approx^{-1} + N_c^{-1} )^{-1}
        N_c_repeat = jnp.broadcast_to(invBtinvNB[0,0], (self.nstokes,self.n_pix)).ravel() ## Repeat N_c for each Stokes parameter, for speed-up afterwards
        
        N_c_inv = jnp.zeros_like(invBtinvNB[0,0])
        N_c_inv = N_c_inv.at[...,self.mask!=0].set(1/invBtinvNB[0,0,self.mask!=0])
        N_c_inv_repeat = jnp.broadcast_to(N_c_inv, (self.nstokes,self.n_pix)).ravel() ## Repeat N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Preparing the operator C_approx^{1/2}
        first_part_left = lambda x : maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), 
                                                                    red_cov_approx_matrix_sqrt, 
                                                                    nside=self.nside, 
                                                                    lmin=self.lmin, 
                                                                    n_iter=self.n_iter).ravel()
        ## Preparing the operator N_c^{-1}
        def second_part_left(x):
            return x*N_c_inv_repeat
        ## Preparing the CG function (Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2})
        func_left_term = lambda x : x.ravel() + first_part_left(second_part_left(first_part_left(x))).ravel()

        ## Preparation of the initial guess
        initial_guess = jnp.copy(component_eta_maps)
        if first_guess is not None:
            initial_guess = jnp.copy(first_guess)

        ## Preparation of the right member of the CG
        right_member = component_eta_maps
    
        # func_lineax = lx.FunctionLinearOperator(func_left_term, jax.ShapeDtypeStruct((self.nstokes*self.n_pix,),jnp.float64), tags=(lx.symmetric_tag,lx.positive_semidefinite_tag))
        # func_norm = lambda x : jnp.linalg.norm(x,ord=2)
        # CG_obj = lx.CG(rtol=self.tolerance_CG, atol=self.atol_CG, max_steps=self.limit_iter_cg_eta, norm=func_norm)
        # options_dict = {"y0":initial_guess.ravel(), "precond":precond_lineax}

        time_start = time.time()
        inverse_term, iterations = jsp.sparse.linalg.cg(func_left_term, 
                                                        right_member.ravel(), 
                                                        x0=initial_guess.ravel(), 
                                                        tol=self.tolerance_CG, 
                                                        maxiter=self.limit_iter_cg_eta,
                                                        M=precond_func)
        ## Computing (Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2})^{-1} \eta
        
        # solution = lx.linear_solve(func_lineax, right_member.ravel(), solver=CG_obj, throw=False, options=options_dict)
        # inverse_term = solution.value
        # print("CG-Python-0 Fluct finished with ", solution.result, solution.stats)
        print("CG-Inverse term eta finished in ",time.time()-time_start, " seconds with ", iterations, " iterations", flush=True)

        ## Computing the log-proba of the correction term
        second_term_complete = jnp.einsum('sp,p,sp', component_eta_maps, self.mask, inverse_term.reshape(self.nstokes,self.n_pix))
        if return_inverse:
            return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2, inverse_term.reshape(self.nstokes,self.n_pix)
        return -(-0 + second_term_complete)/2.*jhp.nside2resol(self.nside)**2


    def get_conditional_proba_correction_likelihood_JAX_v2db(self, 
                                                             old_params_mixing_matrix, 
                                                             new_params_mixing_matrix, 
                                                             inverse_term, 
                                                             component_eta_maps, 
                                                             red_cov_approx_matrix_sqrt, 
                                                             inverse_term_x_Capprox_root=None):
        """ Get conditional probability of correction term in the likelihood from the full mixing matrix,
            assuming the difference between the old and new mixing matrix is small

            With notation C_approx instead of \tilde{C}, the associated conditional probability is given by: 
                - (\eta ^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} \eta
            Or:
                - (\eta^t ( Id + C_approx^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C_approx^{1/2}) \eta

            The computation proceeds as follows:
                Knowing A^{-1} = ( Id + C_approx^{1/2} N_{c,old}^{-1} C_approx^{1/2} )^{-1} ; 
                We compute ( Id + C_approx^{1/2} N_{c,new}^{-1} C_approx^{1/2} )^{-1} eta
                From:  \eta (A + C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2})^{-1} \eta
                        =  \eta (A^{-1} - A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1}) \eta
                Which holds if || N_{c,new}^{-1} - N_{c,old}^{-1} || is small enough

            As we have:
                - inverse_term = A^{-1} eta
                - inverse_term_x_Capprox_root = C_approx^{1/2} A^{-1} eta (optional, otherwise it can be computed in the routine)
                - The B_{old} and B_{new} mixing matrices to reconstruct easily N_{c,old}^{-1} and N_{c,new}^{-1}
            We can thus easily compute:
                \eta (A^{-1} - A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1}) \eta

            Parameters
            ----------
            :param old_params_mixing_matrix: B_{old} to generate N_{c,old}, old mixing matrix of dimension [component, frequencies]
            :param new_params_mixing_matrix: B_{new} to generate N_{c,new}, new mixing matrix of dimension [component, frequencies]
            :param inverse_term: previous inverse term computed with N_{c,old}, of dimension [component, n_pix]
            :param component_eta_maps: set of eta maps of dimension [component, n_pix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
            :param inverse_term_x_Capprox_root: optional, C_approx^{1/2} A^{-1} eta, of dimension [component, n_pix] (otherwise it will be recomputed here)

            Returns
            -------
            :return: computation of the log-proba of the correction term to the likelihood
        """

        ## Retrieving the mixing matrix B_{old} from the old set of parameters
        self.update_params(old_params_mixing_matrix, jax_use=True)
        old_mixing_matrix = self.get_B(jax_use=True)
        

        ## Retrieving the mixing matrix B_{new} from the new set of parameters
        self.update_params(new_params_mixing_matrix, jax_use=True)
        new_mixing_matrix = self.get_B(jax_use=True)

        ## Preparing both mixing matrix
        old_invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, old_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2
        new_invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)*jhp.nside2resol(self.nside)**2

        ## Preparing N_c^{-1} for both mixing matrix
        old_N_c_inv = jnp.zeros_like(old_invBtinvNB[0,0])
        old_N_c_inv = old_N_c_inv.at[...,self.mask!=0].set(1/old_invBtinvNB[0,0,self.mask!=0])
        old_N_c_inv_repeat = jnp.broadcast_to(old_N_c_inv, (self.nstokes,self.n_pix)).ravel()

        new_N_c_inv = jnp.zeros_like(new_invBtinvNB[0,0])
        new_N_c_inv = new_N_c_inv.at[...,self.mask!=0].set(1/new_invBtinvNB[0,0,self.mask!=0])
        new_N_c_inv_repeat = jnp.broadcast_to(new_N_c_inv, (self.nstokes,self.n_pix)).ravel()

        ## Preparing the operator (N_{c,new}^{-1} - N_{c,old}^{-1})
        def second_part_left(x):
            return x*(new_N_c_inv_repeat-old_N_c_inv_repeat)
        func_to_apply = lambda x : second_part_left(x)

        ## If not precomputed, we compute C_approx^{1/2} A^{-1} eta
        if inverse_term_x_Capprox_root is None:
            inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.n_pix), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()

        ## Applying the operator (N_{c,new}^{-1} - N_{c,old}^{-1}) to C_approx^{1/2} A^{-1} eta
        perturbation_term = func_to_apply(inverse_term_x_Capprox_root).reshape(self.nstokes,self.n_pix)
        

        # previous_inverse_x_eta = inverse_term.reshape(self.nstokes,self.n_pix)
        # previous_inverse_x_Capprox_root_x_eta = inverse_term_x_Capprox_root.reshape(self.nstokes,self.n_pix)

        ## Computing contribution of \eta A^{-1} \eta
        first_order_term = jnp.einsum('sp,p,sp', component_eta_maps, self.mask, inverse_term.reshape(self.nstokes,self.n_pix))

        ## Computing contribution of \eta A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1} \eta
        perturbation_term = jnp.einsum('sp,p,sp', perturbation_term, self.mask, inverse_term_x_Capprox_root.reshape(self.nstokes,self.n_pix))

        # new_log_proba = jnp.einsum('sp,p,sp', component_eta_maps, self.mask, previous_inverse_x_eta) - jnp.einsum('sp,p,sp', perturbation_term, self.mask, previous_inverse_x_Capprox_root_x_eta)
        ## Assembling everything
        new_log_proba = first_order_term - perturbation_term
        print("First order :", first_order_term)
        print("Perturbation :", -perturbation_term)

        return -(-0 + new_log_proba)/2.*jhp.nside2resol(self.nside)**2



    def get_conditional_proba_mixing_matrix_v2b_JAX(self, 
                                                    new_params_mixing_matrix, 
                                                    full_data_without_CMB,  
                                                    red_cov_approx_matrix_sqrt, 
                                                    component_eta_maps=None,
                                                    first_guess=None, 
                                                    biased_bool=False,
                                                    precond_func=None):
        """ Get conditional probability of the conditional probability associated with the B_f parameters
            
            With notation C_approx instead of \tilde{C}, the associated conditional probability is given by : 
                - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + \eta^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} \eta
            
            Parameters
            ----------
            :param new_params_mixing_matrix: new B_f parameters of the mixing matrix to compute the log-proba, dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, n_pix]
            :param red_cov_approx_matrix_sqrt: matrix square root of the covariance of C_approx, of dimension [lmin:lmax, nstokes, nstokes]
            :param component_eta_maps: set of eta maps of dimension [component, n_pix]
            :param first_guess: initial guess for the CG

            Returns
            -------
            :return: computation of the conditional probability of the mixing matrix
        """

        # self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)
        # new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)

        # self.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)
        self.update_params(new_params_mixing_matrix,jax_use=True)
        new_mixing_matrix = self.get_B(jax_use=True)
        
        # Compute spectral likelihood : (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(new_mixing_matrix, 
                                                                                           full_data_without_CMB)

        # Compute correction term to the likelihood : (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
        if biased_bool:
            log_proba_perturbation_likelihood = 0
            inverse_term = 0
        else:
            invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
            log_proba_perturbation_likelihood, inverse_term = self.get_conditional_proba_correction_likelihood_JAX_v2d(invBtinvNB, 
                                                                                                                       component_eta_maps, 
                                                                                                                       red_cov_approx_matrix_sqrt, 
                                                                                                                       first_guess=first_guess,
                                                                                                                       return_inverse=True)

        return (log_proba_spectral_likelihood + log_proba_perturbation_likelihood), inverse_term
    

    def get_conditional_proba_mixing_matrix_v3_JAX(self, 
                                                   new_params_mixing_matrix, 
                                                   old_params_mixing_matrix, 
                                                   full_data_without_CMB, 
                                                   red_cov_approx_matrix_sqrt, 
                                                   component_eta_maps=None,
                                                   first_guess=None,
                                                   previous_inverse_x_Capprox_root=None, 
                                                   biased_bool=False):
        """ Get conditional probability of the conditional probability associated with the B_f parameters
            
            The associated conditional probability is given by : 
                - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta
            
            Parameters
            ----------
            :param new_params_mixing_matrix: new B_f parameters of the mixing matrix to compute the log-proba, dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            :param full_data_without_CMB: data without from which the CMB (sample) was substracted, of dimension [frequencies, n_pix]
            :param component_eta_maps: set of eta maps of dimension [component, n_pix]
            :param red_cov_approx_matrix: covariance matrice approx (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

            Returns
            -------
            :return: computation of the conditional probability of the mixing matrix
        """

        # self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)
        # new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)

        # self.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)
        self.update_params(new_params_mixing_matrix,jax_use=True)
        new_mixing_matrix = self.get_B(jax_use=True)
        
        # Compute spectral likelihood : (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(new_mixing_matrix, 
                                                                                           jnp.array(full_data_without_CMB))

        # Compute correction term to the likelihood : (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
        if biased_bool:
            log_proba_perturbation_likelihood = 0
        else:    
            log_proba_perturbation_likelihood = self.get_conditional_proba_correction_likelihood_JAX_v2db(old_params_mixing_matrix, 
                                                                                                          new_params_mixing_matrix, 
                                                                                                          first_guess, 
                                                                                                          component_eta_maps, 
                                                                                                          red_cov_approx_matrix_sqrt, 
                                                                                                          inverse_term_x_Capprox_root=previous_inverse_x_Capprox_root)

        return log_proba_spectral_likelihood + log_proba_perturbation_likelihood
    

    def harmonic_marginal_probability(self, sample_B_f_r, noise_weighted_alm_data, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total, red_cov_approx_matrix):
        """ Get marginal probability of the full likelihood, over s_c
        """

        chx.assert_axis_dimension(sample_B_f_r, 0, 2*(self.n_frequencies-jnp.size(self.pos_special_freqs))+1)
        chx.assert_axis_dimension(red_cov_approx_matrix, 0, self.lmax+1-self.lmin)
        chx.assert_axis_dimension(theoretical_red_cov_r1_tensor, 0, self.lmax+1-self.lmin)
        chx.assert_axis_dimension(theoretical_red_cov_r0_total, 0, self.lmax+1-self.lmin)
        chx.assert_axis_dimension(noise_weighted_alm_data, 0, self.n_frequencies)
        chx.assert_axis_dimension(noise_weighted_alm_data, 1, self.nstokes)

        r_param = sample_B_f_r[-1]
        B_f = sample_B_f_r[:-1]#.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F')

        self.update_params(B_f, jax_use=True)
        mixing_matrix_sample = self.get_B(jax_use=True).mean(axis=2)

        red_CMB_cell = theoretical_red_cov_r0_total + r_param*theoretical_red_cov_r1_tensor

        inv_BtinvNB_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sample)
        effective_noise_CMB_c_ell = inv_BtinvNB_c_ell[0,0]
        red_noise_CMB = jnp.einsum('l,sk->lsk', effective_noise_CMB_c_ell, jnp.eye(self.nstokes))

        ## Computation of the first term d^t P d
        central_term_1_ = jnp.einsum('fc,ckl,gk->fgl',
                                    mixing_matrix_sample, 
                                    inv_BtinvNB_c_ell, 
                                    mixing_matrix_sample)

        central_term_1 = jnp.einsum('fnl,sk->fnlsk', central_term_1_, jnp.eye(self.nstokes))

        frequency_alm_central_term_1 = frequency_alms_x_obj_red_covariance_cell_JAX(
                                                            noise_weighted_alm_data, 
                                                            central_term_1, 
                                                            lmin=self.lmin)


        ## Computation of the second term s_{c,ML}^t (C^{-1} + N_c^{-1}) s_{c,ML}
        # frequency_noise_Stokes = jnp.einsum('fgl,sk->fglsk', red_freq_inverse_noise, jnp.eye(self.nstokes))
    
        multiplicative_term_s_cML_ = jnp.einsum('ckl,fk->cfl',
                            inv_BtinvNB_c_ell,
                            mixing_matrix_sample)
        multiplicative_term_s_cML = jnp.einsum('cfl,sk->cflsk', multiplicative_term_s_cML_, jnp.eye(self.nstokes))
        s_cML = frequency_alms_x_obj_red_covariance_cell_JAX(noise_weighted_alm_data, multiplicative_term_s_cML, lmin=self.lmin)[0,...]


        central_term_2 = jnp.linalg.pinv(red_noise_CMB + red_CMB_cell)
        alm_central_term_2 = alms_x_red_covariance_cell_JAX(s_cML, central_term_2, lmin=self.lmin)

        ## Computation of the third term ln | (C + N_c) (C_approx + N_c)^-1 |
        red_contribution = jnp.einsum('lsk,lkm->lsm', 
                                        red_CMB_cell + red_noise_CMB, 
                                        jnp.linalg.pinv(red_noise_CMB + red_cov_approx_matrix))


        first_term_complete = -alm_dot_product_JAX(noise_weighted_alm_data, frequency_alm_central_term_1, self.lmax)

        second_term_complete = alm_dot_product_JAX(s_cML, alm_central_term_2, self.lmax)

        third_term = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(jnp.linalg.det(red_contribution)) ).sum()

        return -(first_term_complete + second_term_complete + third_term)/2



def single_Metropolis_Hasting_step(random_PRNGKey, old_sample, step_size, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        sample_proposal = dist.Normal(jnp.ravel(old_sample,order='F'), step_size).sample(key_proposal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs))
        new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample,order='F'))

        return new_sample.reshape(old_sample.shape,order='F')


def multivariate_Metropolis_Hasting_step(random_PRNGKey, old_sample, covariance_matrix, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        sample_proposal = dist.MultivariateNormal(jnp.ravel(old_sample,order='F'), covariance_matrix).sample(key_proposal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs))

        new_sample = jnp.where(
            jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, 
            sample_proposal, 
            jnp.ravel(old_sample,order='F'))
        return new_sample.reshape(old_sample.shape,order='F')


def multivariate_Metropolis_Hasting_step_numpyro(state, covariance_matrix, log_proba, **model_kwargs):
        old_sample, random_PRNGKey = state
        random_PRNGKey, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        sample_proposal = dist.MultivariateNormal(jnp.ravel(old_sample,order='F'), covariance_matrix).sample(key_proposal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs))

        new_sample = jnp.where(
            jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, 
            sample_proposal, 
            jnp.ravel(old_sample,order='F'))
        return new_sample.reshape(old_sample.shape,order='F'), random_PRNGKey


def get_log_pdf_lognormal(x, mean, scale):
    return -(jnp.log(x) - mean)**2/(2*scale**2) - jnp.log(x*scale*jnp.sqrt(2*jnp.pi))


def single_lognormal_Metropolis_Hasting_step(random_PRNGKey, old_sample, step_size, log_proba, **model_kwargs):
        rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

        goal_mean = jnp.ravel(old_sample,order='F')
        goal_step_size = step_size

        mean_lognormal = 2*jnp.log(goal_mean) - 0.5*jnp.log(goal_step_size**2 + goal_mean**2)
        std_lognormal = jnp.sqrt(jnp.log(1 + goal_step_size**2/goal_mean**2))
        sample_proposal = dist.LogNormal(loc=mean_lognormal, scale=std_lognormal).sample(key_proposal)

        proposal_mean_lognormal = 2*jnp.log(sample_proposal) - 0.5*jnp.log(goal_step_size**2 + sample_proposal**2)
        proposal_std_lognormal = jnp.sqrt(jnp.log(1 + goal_step_size**2/sample_proposal**2))

        diff_lognormal = get_log_pdf_lognormal(sample_proposal, mean_lognormal, std_lognormal) - get_log_pdf_lognormal(jnp.ravel(old_sample,order='F'), proposal_mean_lognormal, proposal_std_lognormal)

        accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)) - diff_lognormal
        new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample,order='F'))

        # return new_sample.reshape(old_sample.shape,order='F')
        return jnp.reshape(new_sample, old_sample.shape, order='F')

def separate_single_MH_step_index(random_PRNGKey, old_sample, step_size, log_proba, indexes_params, **model_kwargs):
    
    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry[0], 3)

        sample_proposal = dist.Normal(carry[1][index_Bf], step_size[index_Bf]).sample(key_proposal)
        
        proposal_params = jnp.copy(carry[1])
        proposal_params = proposal_params.at[index_Bf].set(sample_proposal)

        accept_prob = -(log_proba(carry[1], **model_kwargs) - log_proba(proposal_params, **model_kwargs))
        new_param = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, carry[1][index_Bf])
        
        proposal_params = proposal_params.at[index_Bf].set(new_param)
        return (rng_key, proposal_params), new_param

    carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F')), indexes_params)
    new_sample = jnp.copy(jnp.ravel(old_sample,order='F'))
    new_sample = new_sample.at[indexes_params].set(new_params)
    latest_PRNGKey = carry[0]
    # return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F')
    return latest_PRNGKey, jnp.reshape(new_sample, old_sample.shape,order='F')

def separate_single_MH_step_index_v2(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, **model_kwargs):
    
    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        sample_proposal = dist.Normal(carry['sample'][index_Bf], step_size[index_Bf]).sample(key_proposal)

        proposal_params = jnp.copy(carry['sample'])
        proposal_params = proposal_params.at[index_Bf].set(sample_proposal)

        proposal_log_proba = log_proba(proposal_params, **model_kwargs)

        # accept_prob = -(log_proba(carry[1], **model_kwargs) - log_proba(proposal_params, **model_kwargs))
        accept_prob = -(carry['log_proba'] - proposal_log_proba)
        
        new_param = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, carry['sample'][index_Bf])
        new_log_proba = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, proposal_log_proba, carry['log_proba'])

        proposal_params = proposal_params.at[index_Bf].set(new_param)
        new_carry = {'PRNGKey':rng_key, 'sample':proposal_params, 'log_proba':new_log_proba}
        return new_carry, new_param
        # return (rng_key, proposal_params), new_param

    initial_carry = {'PRNGKey':random_PRNGKey, 
                     'sample':jnp.ravel(old_sample,order='F'), 
                     'log_proba':log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs)}
    # carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F')), indexes_Bf)
    carry, new_params = jlax.scan(map_func, initial_carry, indexes_Bf)
    new_sample = jnp.copy(jnp.ravel(old_sample,order='F'))
    new_sample = new_sample.at[indexes_Bf].set(new_params)

    # latest_PRNGKey = carry[0]
    latest_PRNGKey = carry['PRNGKey']
    # return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F')
    return latest_PRNGKey, jnp.reshape(new_sample, old_sample.shape,order='F')

def separate_single_MH_step_index_v2b(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, **model_kwargs):
        
    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        sample_proposal = dist.Normal(carry['sample'][index_Bf], step_size[index_Bf]).sample(key_proposal)
        chx.assert_equal_shape((sample_proposal, carry['sample'][index_Bf]))
        chx.assert_size(sample_proposal, 1)

        proposal_params = jnp.copy(carry['sample'], order='K')
        proposal_params = proposal_params.at[index_Bf].set(sample_proposal)

        proposal_log_proba = log_proba(proposal_params, **model_kwargs)

        # accept_prob = -(log_proba(carry[1], **model_kwargs) - log_proba(proposal_params, **model_kwargs))
        accept_prob = -(carry['log_proba'] - proposal_log_proba)
        chx.assert_size(carry['log_proba'], 1)
        chx.assert_size(proposal_log_proba, 1)
        
        log_proba_uniform = jnp.log(dist.Uniform().sample(key_accept))
        new_param = jnp.where(log_proba_uniform < accept_prob, sample_proposal, carry['sample'][index_Bf])
        new_log_proba = jnp.where(log_proba_uniform < accept_prob, proposal_log_proba, carry['log_proba'])

        proposal_params = proposal_params.at[index_Bf].set(new_param)
        new_carry = {'PRNGKey':rng_key, 'sample':proposal_params, 'log_proba':new_log_proba}
        return new_carry, new_param
        # return (rng_key, proposal_params), new_param

    chx.assert_equal_shape((old_sample, step_size))

    initial_carry = {'PRNGKey':random_PRNGKey, 
                     'sample':old_sample, 
                     'log_proba':log_proba(old_sample, **model_kwargs)}
    # carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F')), indexes_Bf)
    last_carry, new_params = jlax.scan(map_func, initial_carry, indexes_Bf)
    # new_sample = jnp.copy(old_sample, order='K')
    # new_sample = new_sample.at[indexes_Bf].set(new_params)
    new_sample = jnp.copy(last_carry['sample'], order='K')

    # latest_PRNGKey = carry[0]
    latest_PRNGKey = last_carry['PRNGKey']
    # return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F')
    return latest_PRNGKey, new_sample

def separate_single_MH_step_index_v3(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, indexes_patches_Bf, size_patches, max_len_patches_Bf, len_indexes_Bf, **model_kwargs):
    """  
        Perform Metroplis-Hasting step for a given set of indexes given by indexes_patches_Bf
        
        Parameters
        ----------
        :param random_PRNGKey: random key for the random number generator ; note it will be split for each sample indexed by indexes B_f
        :param old_sample: old sample to be updated
        :param step_size: step size for all B_f
        :param log_proba: log probability function as log(p(x) (and not -log(p(x)) as in the previous version)
        :param indexes_Bf: all indexes of the parameters to be updated
        :param indexes_patches_Bf: indexes of the parameters to be updated, corresponding to first indexes of params per patch
        :param indexes_patches_B_f: indexes of the patches to be updated
        :param model_kwargs: additional arguments for the log_proba function

        Returns
        -------
        :return: latest_PRNGKey, new_sample
    """

    # max_index_Bf = jnp.max(indexes_Bf)+1
    def map_func(carry, index_Bf):
        indexes_to_consider = (index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32))%len_indexes_Bf
        mask_in_indexes_B_f = jnp.where(jnp.isin(index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32), indexes_Bf), 1, 0)
        mask_indexes_to_consider = jnp.where(jnp.arange(max_len_patches_Bf) < size_patches[index_Bf], mask_in_indexes_B_f, 0)

        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        sample_proposal = dist.Normal(carry['sample'][indexes_to_consider], step_size[indexes_to_consider]*mask_indexes_to_consider).sample(key_proposal)

        proposal_params = jnp.copy(carry['sample'])
        proposal_params = proposal_params.at[indexes_to_consider].set(sample_proposal)

        proposal_log_proba = log_proba(proposal_params, **model_kwargs)

        # accept_prob = -(log_proba(carry[1], **model_kwargs) - log_proba(proposal_params, **model_kwargs))
        accept_prob = -(carry['log_proba'] - proposal_log_proba)

        log_proba_uniform = jnp.log(dist.Uniform().sample(key_accept))
        new_param = jnp.where(log_proba_uniform < accept_prob, sample_proposal, carry['sample'][indexes_to_consider])
        new_log_proba = jnp.where(log_proba_uniform < accept_prob, proposal_log_proba, carry['log_proba'])

        proposal_params = proposal_params.at[indexes_to_consider].set(new_param)
        new_carry = {'PRNGKey':rng_key, 'sample':proposal_params, 'log_proba':new_log_proba}
        return new_carry, new_param
        # return (rng_key, proposal_params), new_param

    # initial_carry = {'PRNGKey':random_PRNGKey, 
    #                  'sample':jnp.ravel(old_sample,order='F'), 
    #                  'log_proba':log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs)}

    initial_carry = {'PRNGKey':random_PRNGKey, 
                     'sample':old_sample,
                     'log_proba':log_proba(old_sample, **model_kwargs)}

    # carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F')), indexes_Bf)
    carry, new_params = jlax.scan(map_func, initial_carry, indexes_patches_Bf)
    # new_sample = jnp.copy(jnp.ravel(old_sample,order='F'))
    # new_sample = jnp.copy(old_sample)
    # new_sample = new_sample.at[indexes_Bf].set(new_params)
    new_sample = carry['sample']

    # latest_PRNGKey = carry[0]
    latest_PRNGKey = carry['PRNGKey']
    # return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F')
    # return latest_PRNGKey, jnp.reshape(new_sample, old_sample.shape,order='F')
    return latest_PRNGKey, new_sample

def separate_single_MH_step_index_accelerated(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, first_guess, **model_kwargs):
    
    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry[0], 3)

        sample_proposal = dist.Normal(carry[1][index_Bf], step_size[index_Bf]).sample(key_proposal)

        old_inverse = carry[2]

        proposal_params = jnp.copy(carry[1])
        proposal_params = proposal_params.at[index_Bf].set(sample_proposal)

        accept_prob_0, inverse_term_0 = log_proba(carry[1], first_guess=old_inverse, **model_kwargs)
        accept_prob_1, inverse_term_1 = log_proba(proposal_params, first_guess=old_inverse, **model_kwargs)
        accept_prob = -(accept_prob_0 - accept_prob_1)
        new_param = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, carry[1][index_Bf])
        new_inverse_term = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, inverse_term_1, inverse_term_0)
        
        proposal_params = proposal_params.at[index_Bf].set(new_param)
        return (rng_key, proposal_params, new_inverse_term), new_param

    carry, new_params = jlax.scan(map_func, (random_PRNGKey, jnp.ravel(old_sample,order='F'), first_guess), indexes_params)
    new_sample = jnp.copy(jnp.ravel(old_sample,order='F'))
    new_sample = new_sample.at[indexes_params].set(new_params)
    latest_PRNGKey = carry[0]
    new_inverse_term = carry[2]
    # return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F'), new_inverse_term
    return latest_PRNGKey, new_sample, new_inverse_term


# def log_proba_proposal(new_sample, old_sample, step_size, grad_proba, **model_kwargs):
#     """ Return log proba proposal distribution MALA
#     """
#     return -1/(4*step_size)*jnp.linalg.norm(new_sample - old_sample - step_size*grad_proba(x, **model_kwargs),ord=2)**2


# def get_MALA_step(random_PRNGKey, old_sample, step_size, log_proba, grad_proba, indexes_params, **model_kwargs):
#     """ Compute Metropolis-adjusted Langevin (MALA) step for a given log-probability function and its gradient.
#     """

#     rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

#     fluctuation = dist.Normal(jnp.ravel(old_sample,order='F'), jnp.ones(jnp.size(old_sample))).sample(key_proposal)*jnp.sqrt(2*step_size)

#     sample_proposal = old_sample + step_size*grad_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) + fluctuation

#     diff_proposal = -(log_proba_proposal(sample_proposal, jnp.ravel(old_sample,order='F'), step_size, grad_proba, **model_kwargs) 
#                         - log_proba_proposal(jnp.ravel(old_sample,order='F'), sample_proposal, step_size, grad_proba, **model_kwargs))

#     accept_prob = -(log_proba(jnp.ravel(old_sample,order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)) + diff_proposal
#     new_sample = jnp.where(jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample,order='F'))

#     return new_sample[indexes_params].reshape(old_sample.shape,order='F')
