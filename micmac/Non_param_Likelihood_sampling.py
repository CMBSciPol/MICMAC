import os, sys, time
import toml
import numpy as np
import healpy as hp
import scipy
import fgbuster
import jax
import jax.lax as jlx
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
from jax_tqdm import scan_tqdm, loop_tqdm
import chex as chx
import numpyro

from .tools import *
from .jax_tools import *
from .algorithm_toolbox import *
from .proba_functions import *
# from .Sampling_toolbox import *
from .mixingmatrix import *
from .templates_spv import *
from .noisecovar import *
from .jax_Sampling_toolbox import *
from .temporary_tools import *

from jax import config
config.update("jax_enable_x64", True)

class MICMAC_Sampler(Sampling_functions):
    def __init__(self, nside, lmax, nstokes, 
                 frequency_array, freq_inverse_noise, pos_special_freqs=[0,-1],
                 freq_noise_c_ell=None,
                 n_components=3, lmin=2,
                 lmin_r=-1, lmax_r=-1,
                 n_iter=8, limit_iter_cg=2000, tolerance_CG=1e-10, atol_CG=1e-8,
                 limit_iter_cg_eta=200,
                 mask=None,
                 use_automatic_step_size=False, num_sample_AM = 100000000, 
                 epsilon_cov = 1e-20, scale_param = 2.38**2,

                 restrict_to_mask=True,
                 bin_ell_distribution=None,
                 use_old_s_c_sampling=False,
                 perturbation_eta_covariance=False,
                 acceptance_posdef=False,

                 use_binning=False,
                 cheap_save=True, very_cheap_save=False,
                 biased_version=False, lognormal_r=False,
                 r_true=0,
                 sample_eta_B_f=True,
                 sample_r_Metropolis=True, sample_C_inv_Wishart=False,
                 step_size_r=1e-4,
                 covariance_B_f=-1,
                 indexes_free_Bf=False,

                 instrument_name='SO_SAT',
                 number_iterations_sampling=100, number_iterations_done=0,
                 seed=0,
                 disable_chex=True,
                 
                 spv_nodes_b=[]):
        """ Non parametric likelihood sampling object
        """

        super(MICMAC_Sampler,self).__init__(nside=nside,lmax=lmax,nstokes=nstokes,lmin=lmin,
                                            lmin_r=lmin_r, lmax_r=lmax_r,
                                            frequency_array=frequency_array,freq_inverse_noise=freq_inverse_noise, 
                                            spv_nodes_b=spv_nodes_b,
                                            pos_special_freqs=pos_special_freqs,n_components=n_components,
                                            n_iter=n_iter, limit_iter_cg=limit_iter_cg, limit_iter_cg_eta=limit_iter_cg_eta, 
                                            tolerance_CG=tolerance_CG, atol_CG=atol_CG, 
                                            mask=mask, restrict_to_mask=restrict_to_mask, bin_ell_distribution=bin_ell_distribution)

        # Quick test parameters
        self.instrument_name = instrument_name
        self.sample_eta_B_f = bool(sample_eta_B_f)
        if self.sample_eta_B_f is True:
            assert self.n_components > 1
            try:
                assert len(pos_special_freqs) == self.n_components-1
            except:
                raise Exception("The number of special frequencies should be equal to the number of components - 1")
        self.biased_version = bool(biased_version)
        self.cheap_save = bool(cheap_save)
        self.very_cheap_save = bool(very_cheap_save)
        self.disable_chex = disable_chex
        if indexes_free_Bf is False:
            # indexes_free_Bf = jnp.arange((self.n_frequencies-len(pos_special_freqs))*(self.n_correlations-1))
            indexes_free_Bf = jnp.arange(self.len_params)
        self.indexes_free_Bf = jnp.array(indexes_free_Bf)
        assert jnp.size(self.indexes_free_Bf) <= self.len_params
        # self.use_old_s_c_sampling = bool(use_old_s_c_sampling)
        # self.fixed_eta_covariance = bool(fixed_eta_covariance)
        self.perturbation_eta_covariance = bool(perturbation_eta_covariance)
        self.use_binning = bool(use_binning)
        self.acceptance_posdef = bool(acceptance_posdef)

        # CMB parameters
        self.r_true = float(r_true)
        assert ((sample_r_Metropolis and sample_C_inv_Wishart) == False) and ((sample_r_Metropolis or not(sample_C_inv_Wishart)) or (not(sample_r_Metropolis) or sample_C_inv_Wishart))
        self.sample_r_Metropolis = bool(sample_r_Metropolis)
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)

        # Noise parameters
        self.freq_noise_c_ell = freq_noise_c_ell 

        # Metropolis-Hastings parameters
        self.lognormal_r = lognormal_r
        # self.step_size_B_f = step_size_B_f
        # if covariance_B_f==-1:
        #     self.covariance_B_f = jnp.diag((jnp.ravel(step_size_B_f,order='F')**2)*jnp.ones((self.n_frequencies-len(pos_special_freqs))*(self.n_correlations-1)))
        # else:
        #     self.covariance_B_f = covariance_B_f
        self.covariance_B_f = covariance_B_f
        self.step_size_r = step_size_r
        # self.number_steps_sampler_B_f = int(number_steps_sampler_B_f) # Maximum number of steps for the Metropolis-Hasting to sample the mixing matrix
        # self.number_steps_sampler_r = int(number_steps_sampler_r)
        self.use_automatic_step_size = use_automatic_step_size
        self.num_sample_AM = num_sample_AM
        self.epsilon_cov = epsilon_cov
        self.scale_param = scale_param

        # Sampling parameters
        self.number_iterations_sampling = int(number_iterations_sampling) # Maximum number of iterations for the sampling
        self.number_iterations_done = int(number_iterations_done) # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed

        # Samples preparation
        self.all_samples_eta = jnp.empty(0)
        self.all_params_mixing_matrix_samples = jnp.empty(0)
        self.all_samples_wiener_filter_maps = jnp.empty(0)
        self.all_samples_fluctuation_maps = jnp.empty(0)
        self.all_samples_r = jnp.empty(0)
        self.all_samples_CMB_c_ell = jnp.empty(0)


    @property
    def all_samples_s_c(self):
        return self.all_samples_wiener_filter_maps +  self.all_samples_fluctuation_maps



    def generate_CMB(self, return_spectra=True):
        """ Returns CMB spectra of scalar modes only and tensor modes only (with r=1)
            Returns either CMB spectra in the usual form,
            or in the red_cov form if return_spectra == False
        """
        if self.nstokes == 2:
            partial_indices_polar = np.array([1,2])
        elif self.nstokes == 1:
            partial_indices_polar = np.array([0])
        else:
            partial_indices_polar = np.arange(4)

        all_spectra_r0 = generate_power_spectra_CAMB(self.nside*2, r=0, typeless_bool=True)
        all_spectra_r1 = generate_power_spectra_CAMB(self.nside*2, r=1, typeless_bool=True)

        camb_cls_r0 = all_spectra_r0['total'][:self.lmax+1,partial_indices_polar]

        tensor_spectra_r1 = all_spectra_r1['tensor'][:self.lmax+1,partial_indices_polar]

        theoretical_r1_tensor = np.zeros((self.n_correlations,self.lmax+1))
        theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

        theoretical_r1_tensor[:self.nstokes,...] = tensor_spectra_r1.T
        theoretical_r0_total[:self.nstokes,...] = camb_cls_r0.T

        if return_spectra:
            return theoretical_r0_total, theoretical_r1_tensor

        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[self.lmin:]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)[self.lmin:]
        return theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor


    def generate_input_freq_maps_from_fgs(self, freq_maps_fgs, return_only_freq_maps=True, return_only_maps=False):
        indices_polar = np.array([1,2,4])

        theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = self.generate_CMB(return_spectra=False)

        c_ell_select_only_Bmodes = np.zeros((6,self.lmax+1))
        c_ell_select_only_Bmodes[2,self.lmin:] = 1
        red_cov_select_Bmodes = get_reduced_matrix_from_c_ell(c_ell_select_only_Bmodes[indices_polar,...])[self.lmin:,...]

        true_cmb_specra = get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total + self.r_true*theoretical_red_cov_r1_tensor)
        true_cmb_specra_extended = np.zeros((6,self.lmax+1))
        true_cmb_specra_extended[indices_polar,self.lmin:] = true_cmb_specra

        input_cmb_maps_alt = hp.synfast(true_cmb_specra_extended, nside=self.nside, new=True, lmax=self.lmax)[1:,...]

        input_cmb_maps = np.repeat(input_cmb_maps_alt.ravel(order='F'), self.n_frequencies).reshape((self.n_frequencies,self.nstokes,self.n_pix),order='F')
        input_freq_maps = input_cmb_maps + freq_maps_fgs

        # true_red_cov_cmb_specra = get_reduced_matrix_from_c_ell(true_cmb_specra)
        if return_only_freq_maps:
            return input_freq_maps
        
        if return_only_maps:
            return input_freq_maps, input_cmb_maps

        return input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor

    def update_variable(self, all_samples, new_samples_to_add):
        if jnp.size(all_samples) == 0:
            return new_samples_to_add
        elif jnp.size(new_samples_to_add.shape) == 1:
            return jnp.hstack([all_samples,new_samples_to_add])
        else:
            return jnp.vstack([all_samples,new_samples_to_add])

    # def old_update_samples(self, all_samples):
    #     indice_s_c = 1
    #     # if self.sample_eta_B_f:
    #     if not(self.cheap_save):
    #         self.all_samples_eta = self.update_variable(self.all_samples_eta, all_samples[0])

    #     if not(self.very_cheap_save):
    #         self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, all_samples[indice_s_c])
    #         self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, all_samples[indice_s_c+1])
            
    #     # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])
    #     if self.sample_C_inv_Wishart:
    #         if all_samples[indice_s_c+2].shape[1] == self.lmax+1-self.lmin:
    #             all_samples_CMB_c_ell = jnp.array([get_c_ells_from_red_covariance_matrix(all_samples[indice_s_c+2][iteration]) for iteration in range(self.number_iterations_sampling-self.number_iterations_done)])
    #         else:
    #             all_samples_CMB_c_ell = all_samples[indice_s_c+2]
    #         self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples_CMB_c_ell)
    #     if self.sample_r_Metropolis:
    #         self.all_samples_r = self.update_variable(self.all_samples_r, all_samples[indice_s_c+3])

    #     self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, all_samples[5])

    # def old_update_one_sample(self, one_sample):
    #     indice_s_c = 1
    #     # if self.sample_eta_B_f:
    #     self.all_samples_eta = self.update_variable(self.all_samples_eta, one_sample[0].reshape([1]+list(one_sample[0].shape)))
    #     self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, one_sample[5].reshape([1]+list(one_sample[5].shape)))
    #     # else:
    #     #     indice_s_c = -1

    #     self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, one_sample[indice_s_c].reshape([1]+list(one_sample[indice_s_c].shape)))
    #     self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, one_sample[indice_s_c+1].reshape([1]+list(one_sample[indice_s_c+1].shape)))
    #     # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])

    #     if one_sample[indice_s_c+2].shape[0] == self.lmax+1-self.lmin:
    #         one_sample_CMB_c_ell = get_c_ells_from_red_covariance_matrix(one_sample[indice_s_c+2])
    #     else:
    #         one_sample_CMB_c_ell = one_sample[indice_s_c+2]
    #     self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, one_sample_CMB_c_ell.reshape([1]+list(one_sample_CMB_c_ell.shape)))
    #     self.all_samples_r = self.update_variable(self.all_samples_r, one_sample[indice_s_c+3])

    def update_samples(self, all_samples):
        if not(self.cheap_save) and self.sample_eta_B_f:
            self.all_samples_eta = self.update_variable(self.all_samples_eta, all_samples['eta_maps'])

        if not(self.very_cheap_save):
            self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, all_samples['wiener_filter_term'])
            self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, all_samples['fluctuation_maps'])
        # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])
        if self.sample_C_inv_Wishart:
            if all_samples['red_cov_matrix_sample'].shape[1] == self.lmax+1-self.lmin:
                all_samples_CMB_c_ell = jnp.array([get_c_ells_from_red_covariance_matrix(all_samples['red_cov_matrix_sample'][iteration]) for iteration in range(self.number_iterations_sampling-self.number_iterations_done)])
            else:
                all_samples_CMB_c_ell = all_samples['red_cov_matrix_sample']
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples_CMB_c_ell)
        if self.sample_r_Metropolis:
            self.all_samples_r = self.update_variable(self.all_samples_r, all_samples['r_sample'])

        self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, all_samples['params_mixing_matrix_sample'])

    def update_one_sample(self, one_sample):

        if not(self.cheap_save):
            self.all_samples_eta = self.update_variable(self.all_samples_eta, jnp.expand_dims(one_sample['eta_maps'],axis=0))

        if not(self.very_cheap_save):
            self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, jnp.expand_dims(one_sample['wiener_filter_term'],axis=0))
            self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, jnp.expand_dims(one_sample['fluctuation_maps'],axis=0))
        # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])

        if self.sample_C_inv_Wishart:
            if one_sample['red_cov_matrix_sample'].shape[0] == self.lmax+1-self.lmin:
                one_sample_CMB_c_ell = get_c_ells_from_red_covariance_matrix(one_sample['red_cov_matrix_sample'])
            else:
                one_sample_CMB_c_ell = one_sample['red_cov_matrix_sample']
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, jnp.expand_dims(one_sample_CMB_c_ell,axis=0))
        if self.sample_r_Metropolis:
            self.all_samples_r = self.update_variable(self.all_samples_r, one_sample['r_sample'])

        self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, jnp.expand_dims(one_sample['params_mixing_matrix_sample'],axis=0))


    # def compute_covariance_1d(self, iteration, all_samples):
    #     """ Give new covariance matrix from the samples of a 1d variable
    #         assuming after iteration the samples are put to 0
    #     """
    #     dimension_sample = 1
    #     mean_samples = all_samples.sum(axis=0)/(iteration+1)

    #     empirical_covariance = (jnp.einsum('t,t',all_samples,all_samples)
    #                             - (iteration+1)*(mean_samples**2))/(iteration)

    #     return (self.scale_param/dimension_sample)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_sample))
    

    # def compute_covariance_nd(self, iteration, all_samples):
    #     """ Give new covariance matrix from the samples of a nd variable
    #         assuming after iteration the samples are put to 0
    #     """
    #     dimension_sample = all_samples.shape[-1]
    #     mean_samples = all_samples.sum(axis=0)/(iteration+1)

    #     empirical_covariance = (jnp.einsum('ti,tj->tij',all_samples,all_samples).sum(axis=0) 
    #                             - (iteration+1)*jnp.einsum('i,j->ij',mean_samples,mean_samples))/(iteration)

    #     return (self.scale_param/dimension_sample)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_sample))

    # def spectral_likelihood_dB(self, new_params_mixing_matrix, full_data, **model_kwargs):
    #     """
    #     Returns a list of the derivatives of -log(L_sp)
    #     per each spectral parameter
    #     """
    # def spectral_likelihood_dB(self, new_params_mixing_matrix, full_data, **model_kwargs):
    #     """
    #     Returns a list of the derivatives of -log(L_sp)
    #     per each spectral parameter
    #     """

    #     # self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)
    #     self.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)

    #     # new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
    #     # new_mixing_matrix_dB = self._fake_mixing_matrix.get_B_db(jax_use=True)
    #     new_mixing_matrix = self.get_B(jax_use=True)
    #     new_mixing_matrix_dB = self.get_B_db(jax_use=True)
    #     # new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
    #     # new_mixing_matrix_dB = self._fake_mixing_matrix.get_B_db(jax_use=True)
    #     new_mixing_matrix = self.get_B(jax_use=True)
    #     new_mixing_matrix_dB = self.get_B_db(jax_use=True)

    #     invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
    #     BtinvN = get_BtinvN(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
    #     invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
    #     BtinvN = get_BtinvN(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
        
    #     BtinvNd = jnp.einsum('cf, fsp -> csp', BtinvN, full_data)
    #     BtinvNd = jnp.einsum('cf, fsp -> csp', BtinvN, full_data)

    #     P = jnp.einsum('ef,fc,cg,lg,lh->eh', self.freq_inverse_noise, new_mixing_matrix, invBtinvNB, new_mixing_matrix, self.freq_inverse_noise)
    #     P = jnp.einsum('ef,fc,cg,lg,lh->eh', self.freq_inverse_noise, new_mixing_matrix, invBtinvNB, new_mixing_matrix, self.freq_inverse_noise)

    #     logL_dB = jnp.einsum('csp,ck,bfk,fg,gsp->b', BtinvNd, invBtinvNB, new_mixing_matrix_dB, self.freq_inverse_noise-P, full_data)
    #     logL_dB = jnp.einsum('csp,ck,bfk,fg,gsp->b', BtinvNd, invBtinvNB, new_mixing_matrix_dB, self.freq_inverse_noise-P, full_data)

    #     return -logL_dB
    #     return -logL_dB

    # def spectral_likelihood_dB_f(self, new_params_mixing_matrix, full_data_without_CMB, **model_kwargs):
    #     """
    #     Returns a list of the derivatives of -log(L_sp)
    #     per each spectral parameter
    #     """
    # def spectral_likelihood_dB_f(self, new_params_mixing_matrix, full_data_without_CMB, **model_kwargs):
    #     """
    #     Returns a list of the derivatives of -log(L_sp)
    #     per each spectral parameter
    #     """

    #     # self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)
    #     self.update_params(new_params_mixing_matrix.reshape((self.n_frequencies-jnp.size(self.pos_special_freqs), self.n_components-1),order='F'),jax_use=True)

    #     # new_mixing_matrix_fg = self._fake_mixing_matrix.get_B(jax_use=True)[:,1:]
    #     # new_mixing_matrix_dB_fg = self._fake_mixing_matrix.get_B_db(jax_use=True)[:,:,1:]
    #     new_mixing_matrix_fg = self.get_B(jax_use=True)[:,1:]
    #     new_mixing_matrix_dB_fg = self.get_B_db(jax_use=True)[:,:,1:]
    #     # new_mixing_matrix_fg = self._fake_mixing_matrix.get_B(jax_use=True)[:,1:]
    #     # new_mixing_matrix_dB_fg = self._fake_mixing_matrix.get_B_db(jax_use=True)[:,:,1:]
    #     new_mixing_matrix_fg = self.get_B(jax_use=True)[:,1:]
    #     new_mixing_matrix_dB_fg = self.get_B_db(jax_use=True)[:,:,1:]

    #     invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix_fg, jax_use=True)
    #     BtinvN = get_BtinvN(self.freq_inverse_noise, new_mixing_matrix_fg, jax_use=True)
    #     invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix_fg, jax_use=True)
    #     BtinvN = get_BtinvN(self.freq_inverse_noise, new_mixing_matrix_fg, jax_use=True)
        
    #     BtinvNd = jnp.einsum('cf, fsp -> csp', BtinvN, full_data_without_CMB)
    #     BtinvNd = jnp.einsum('cf, fsp -> csp', BtinvN, full_data_without_CMB)

    #     P = jnp.einsum('ef,fc,cg,lg,lh->eh', self.freq_inverse_noise, new_mixing_matrix_fg, invBtinvNB, new_mixing_matrix_fg, self.freq_inverse_noise)
    #     P = jnp.einsum('ef,fc,cg,lg,lh->eh', self.freq_inverse_noise, new_mixing_matrix_fg, invBtinvNB, new_mixing_matrix_fg, self.freq_inverse_noise)

    #     logL_dB = jnp.einsum('csp,ck,bfk,fg,gsp->b', BtinvNd, invBtinvNB, new_mixing_matrix_dB_fg, self.freq_inverse_noise-P, full_data_without_CMB)
    #     logL_dB = jnp.einsum('csp,ck,bfk,fg,gsp->b', BtinvNd, invBtinvNB, new_mixing_matrix_dB_fg, self.freq_inverse_noise-P, full_data_without_CMB)

    #     return -logL_dB

    # def parameter_estimate !!!!
    # TO DO !!!!!!!!!!!!!!!!!


    def perform_Gibbs_sampling(self, input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=0, initial_wiener_filter_term=jnp.empty(0), initial_fluctuation_maps=jnp.empty(0),
                         theoretical_r0_total=jnp.empty(0), theoretical_r1_tensor=jnp.empty(0)):
        """ Perform sampling steps with :
                1. A CG on variable eta for (S_approx + mixed_noise) eta = S_approx^(1/2) x + E^t (B^t N^{-1} B)^{-1} E noise^(1/2) y
                2. A CG for the Wiener filter variable s_c : (s_c - s_c,ML)^t (S_c + E^t (B^t N^{-1} B)^{-1} E) (s_c - s_c,ML)
                3. The c_ell sampling assuming inverse Wishart distribution
                4. Mixing matrix B_f sampling with : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t (S_{approx} + E^t (B^T N^{-1} B)^{-1} E) eta

            Parameters
            ----------
            input_freq_maps : data of initial frequency maps, dimensions [frequencies, nstokes, n_pix]
            
            tot_cov_first_guess : total covariance first guess, composed of all c_ell correlations in order (for polarization [EE, BB, EB])

            depth_p: depth for the noise covariance calculation, in uK.arcmin

            c_ell_approx : 

            init_params_mixing_matrix : 
            
            pos_special_freqs :

            Returns
            -------
        """

        # Disabling all chex checks to speed up the code
        # chx acts like an assert, but is JAX compatible
        if self.disable_chex:
            print("Disabling chex !!!", flush=True)
            chx.disable_asserts()

        # Series of tests

        ## Getting only the relevant spectra
        if self.nstokes == 2:
            indices_to_consider = np.array([1,2,4])
            partial_indices_polar = indices_to_consider[:self.nstokes]
        elif self.nstokes == 1:
            indices_to_consider = np.array([0])
        else:
            indices_to_consider = np.arange(6) # All auto- and cross-correlations

        ## Testing the initial WF term, initialize it properly
        if len(initial_wiener_filter_term) == 0:
            wiener_filter_term = jnp.zeros((self.nstokes,self.n_pix))
        else:
            assert len(initial_wiener_filter_term.shape) == 2
            assert initial_wiener_filter_term.shape == (self.nstokes, self.n_pix)
            # assert initial_wiener_filter_term.shape[1] == self.n_pix
            wiener_filter_term = initial_wiener_filter_term

        ## Testing the initial fluctuation term, initialize it properly
        if len(initial_fluctuation_maps) == 0:
            fluctuation_maps = jnp.zeros((self.nstokes,self.n_pix))
        else:
            assert len(initial_fluctuation_maps.shape) == 2
            assert initial_fluctuation_maps.shape == (self.nstokes, self.n_pix)
            # assert initial_fluctuation_maps.shape[1] == self.n_pix
            fluctuation_maps = initial_fluctuation_maps

        ## Testing the initial spectra given in case the sampling is done with r
        if self.sample_r_Metropolis:
            assert len(theoretical_r0_total.shape) == 2
            assert (theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin) #or (theoretical_r0_total.shape[1] == self.lmax + 1)
            assert len(theoretical_r1_tensor.shape) == 2
            assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

            theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
            theoretical_red_cov_r1_tensor  = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)
            assert theoretical_red_cov_r0_total.shape[1] == self.nstokes
        ## Testing the initial CMB spectra given
        if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):
            CMB_c_ell = CMB_c_ell[indices_to_consider,:]
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[indices_to_consider,:]

        ## testing the initial mixing matrix
        if len(init_params_mixing_matrix.shape) == 1:
            assert len(init_params_mixing_matrix) == (self.n_frequencies-len(self.pos_special_freqs))*(self.n_correlations-1)
        else:
            # assert len(init_params_mixing_matrix.shape) == 2
            assert init_params_mixing_matrix.shape[0] == (self.n_frequencies-len(self.pos_special_freqs))
            assert init_params_mixing_matrix.shape[1] == (self.n_correlations-1)

        ## Final set of tests
        assert len(CMB_c_ell.shape) == 2
        assert CMB_c_ell.shape[1] == self.lmax + 1
        assert len(c_ell_approx.shape) == 2
        assert c_ell_approx.shape[1] == self.lmax + 1

        assert len(input_freq_maps.shape) == 3
        assert input_freq_maps.shape == (self.n_frequencies, self.nstokes, self.n_pix)
        # assert input_freq_maps.shape[1] == self.nstokes
        # assert input_freq_maps.shape[2] == self.n_pix

        # Preparing for the full Gibbs sampling
        len_pos_special_freqs = len(self.pos_special_freqs)

        ## Initial guesses
        initial_eta = jnp.zeros((self.nstokes,self.n_pix))
        # params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix).reshape(
        #                                     ((self.n_frequencies-len_pos_special_freqs),self.n_correlations-1), order='F')
        params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix)
        
        ## CMB covariance preparation in the format [lmax,nstokes,nstokes]
        red_cov_approx_matrix = jnp.array(get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin:,...])
        red_cov_matrix = get_reduced_matrix_from_c_ell(CMB_c_ell)[self.lmin:,...]

        ## Preparation of the mixing matrix
        # self.mixing_matrix_obj = MixingMatrix(self.frequency_array, self.n_components, params_mixing_matrix_init_sample, pos_special_freqs=self.pos_special_freqs)


        ## Jitting the sampling function
        # jitted_sample_eta = jax.jit(self.get_sampling_eta_v2, static_argnames=['suppress_low_modes'])

        # func_logproba_eta = self.get_conditional_proba_correction_likelihood_JAX_v2c
        func_logproba_eta = self.get_conditional_proba_correction_likelihood_JAX_v2d

        # jitted_get_fluctuating_term = jax.jit(self.get_fluctuating_term_maps)
        # jitted_solve_wiener_filter_term = jax.jit(self.solve_generalized_wiener_filter_term)

        # jitted_get_fluctuating_term = jax.jit(self.get_fluctuating_term_maps_v2)
        # jitted_solve_wiener_filter_term = jax.jit(self.solve_generalized_wiener_filter_term_v2)

        # sampling_func_WF = self.solve_generalized_wiener_filter_term_v2c
        sampling_func_WF = self.solve_generalized_wiener_filter_term_v2d
        # sampling_func_Fluct = self.get_fluctuating_term_maps_v2c
        sampling_func_Fluct = self.get_fluctuating_term_maps_v2d
        # if self.use_old_s_c_sampling:
        #     sampling_func_WF = self.solve_generalized_wiener_filter_term
        #     sampling_func_Fluct = self.get_fluctuating_term_maps


        # jitted_get_inverse_wishart_sampling_from_c_ells = jax.jit(self.get_inverse_wishart_sampling_from_c_ells, static_argnames=['q_prior', 'option_ell_2', 'tol'])
        # jitted_get_inverse_wishart_sampling_from_c_ells = jax.jit(self.get_inverse_wishart_sampling_from_c_ells)
        # jitted_get_inverse_wishart_sampling_from_c_ells = jax.jit(self.get_inverse_gamma_sampling_from_c_ells) # Use of gamma distribution instead of inverse Wishart
        func_get_inverse_wishart_sampling_from_c_ells = self.get_inverse_wishart_sampling_from_c_ells
        if self.use_binning:
            # func_get_inverse_wishart_sampling_from_c_ells = self.get_binned_inverse_wishart_sampling_from_c_ells_v2
            func_get_inverse_wishart_sampling_from_c_ells = self.get_binned_inverse_wishart_sampling_from_c_ells_v3
        # func_get_inverse_wishart_sampling_from_c_ells = self.get_inverse_gamma_sampling_from_c_ells

        # if self.lognormal_r:
        #     # Using lognormal proposal distribution for r
        #     r_sampling_MH = single_lognormal_Metropolis_Hasting_step
        # else:
        #     # Using normal proposal distribution for r
        #     r_sampling_MH = single_Metropolis_Hasting_step
        r_sampling_MH = single_Metropolis_Hasting_step

        # jitted_single_Metropolis_Hasting_step_r = jax.jit(single_Metropolis_Hasting_step, static_argnames=['log_proba'])

        jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v2b_JAX, static_argnames=['biased_bool'])
        sampling_func = separate_single_MH_step_index_accelerated

        if self.biased_version or self.perturbation_eta_covariance:
            print("Using biased version or perturbation version of mixing matrix sampling !!!", flush=True)
            # jitted_Bf_func_sampling = jax.jit(self.get_biased_conditional_proba_mixing_matrix_v2_JAX)
            jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v3_JAX, static_argnames=['biased_bool'])
            # sampling_func = separate_single_MH_step_index
            sampling_func = separate_single_MH_step_index_v2


        ## Preparing the scalar quantities
        PRNGKey = random.PRNGKey(self.seed)

        actual_number_of_iterations = self.number_iterations_sampling - self.number_iterations_done
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)


        ## Preparing the step-size for Metropolis-within-Gibbs of B_f sampling
        try :
            initial_step_size_Bf = jnp.array(jnp.diag(jsp.linalg.sqrtm(self.covariance_B_f)), dtype=jnp.float64)
        except:
            initial_step_size_Bf = jnp.array(jnp.diag(jnp.sqrt(self.covariance_B_f)), dtype=jnp.float64)
        print('Step-size B_f', initial_step_size_Bf, flush=True)

        ## Few prints to re-check the toml parameters chosen
        if not(self.sample_eta_B_f):
            print("Not sampling for eta and B_f, only for s_c and the covariance !", flush=True)
        if self.sample_r_Metropolis:
            print("Sample for r instead of C !", flush=True)
        else:
            print("Sample for C with inverse Wishart !", flush=True)
        
        use_precond = False
        if self.mask.sum() == self.n_pix and self.freq_noise_c_ell is not None:
            assert len(self.freq_noise_c_ell.shape) == 3
            assert self.freq_noise_c_ell.shape[0] == self.n_frequencies
            assert self.freq_noise_c_ell.shape[1] == self.n_frequencies
            assert (self.freq_noise_c_ell.shape[2] == self.lmax+1) or (self.freq_noise_c_ell.shape[2] == self.lmax+1-self.lmin)
            if self.freq_noise_c_ell.shape[2] == self.lmax+1:
                self.freq_noise_c_ell = self.freq_noise_c_ell[...,self.lmin:]
            self.freq_noise_c_ell = jnp.array(self.freq_noise_c_ell)

            print("Full sky case !", flush=True)
            use_precond = True

        print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done", flush=True)

        @scan_tqdm(actual_number_of_iterations, )
        def all_sampling_steps(carry, iteration):
            """ Gibbs sampling function, performing the following steps :
                - Sampling of eta, for the C_approx term
                - Sampling of s_c, for the constrained CMB map realization
                - Sampling of C, for the CMB covariance matrix
                - Sampling of B_f, for the mixing matrix

                :param carry: tuple containing the following variables at 1 iteration : eta maps, WF maps, fluctuation maps, CMB covariance, r samples, B_f samples, PRNGKey, s_c samples
                :param iteration: current iteration number

                :return: tuple containing the following variables at the last iteration : eta maps, WF maps, fluctuation maps, CMB covariance, r sample, B_f sample, PRNGKey, s_c sample
            """

            # Extracting the variables from the carry
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, _all_r_samples, _all_B_f_samples, PRNGKey, old_s_c_sample, inverse_term = carry
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, _all_r_samples, params_mixing_matrix_sample, PRNGKey, old_s_c_sample, inverse_term = carry
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample, PRNGKey, old_s_c_sample, inverse_term = carry

            # WF_term_maps = carry['WF_maps']
            # fluct_maps = carry['fluct_maps']
            # red_cov_matrix_sample = carry['red_cov_matrix_sample']
            params_mixing_matrix_sample = carry['params_mixing_matrix_sample']

            PRNGKey = carry['PRNGKey']
            inverse_term = carry['inverse_term']

            # Preparing the new carry
            new_carry = dict()
            all_samples = dict()

            # Preparing a new PRNGKey for eta sampling
            PRNGKey, subPRNGKey = random.split(PRNGKey)

            # Extracting the mixing matrix parameters and initializing the new one
            self.update_params(carry['params_mixing_matrix_sample'])
            mixing_matrix_sampled = self.get_B(jax_use=True)

            # Few checks for the mixing matrix
            chx.assert_axis_dimension(mixing_matrix_sampled, 0, self.n_frequencies)
            chx.assert_axis_dimension(mixing_matrix_sampled, 1, self.n_components)
            # chx.assert_shape(mixing_matrix_sampled, (self.n_frequencies, self.n_components))

            # Application of new mixing matrix to the noise covariance and extracted CMB map from data
            invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, mixing_matrix_sampled, jax_use=True)
            BtinvN_sqrt = get_BtinvN(jnp.sqrt(self.freq_inverse_noise), mixing_matrix_sampled, jax_use=True)
            s_cML = get_Wd(self.freq_inverse_noise, mixing_matrix_sampled, input_freq_maps, jax_use=True)[0]
            del mixing_matrix_sampled

            # Sampling step 1 : sampling of Gaussian variable eta
            
            # Initialize the preconditioner for the eta contribution
            precond_func_eta = None

            if self.sample_eta_B_f and not(self.biased_version):
                # Preparing random variables
                map_random_x = None
                map_random_y = None

                # Sampling eta maps
                new_carry['eta_maps'] = self.get_sampling_eta_v2(red_cov_approx_matrix_sqrt, invBtinvNB, BtinvN_sqrt, 
                                                               subPRNGKey, map_random_x=map_random_x, map_random_y=map_random_y, 
                                                               suppress_low_modes=True)
                # eta_maps_sample = new_eta_maps_sample
                
                # Checking shape of the resulting maps
                chx.assert_shape(new_carry['eta_maps'], (self.nstokes, self.n_pix))

                # Preparing the preconditioner
                if use_precond:
                    noise_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled)[0,0]
                    red_inv_noise_c_ell = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(jnp.stack([noise_c_ell, noise_c_ell, jnp.zeros_like(noise_c_ell)])))#[self.lmin:]
                    red_preconditioner_eta = jnp.linalg.pinv(jnp.eye(self.nstokes) 
                                                            + jnp.einsum('lij,ljk,lkm->lim', 
                                                            red_cov_approx_matrix_sqrt, 
                                                            red_inv_noise_c_ell, 
                                                            red_cov_approx_matrix_sqrt))
                    precond_func_eta = lambda x: maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), 
                                                                                red_preconditioner_eta, 
                                                                                nside=self.nside, 
                                                                                lmin=self.lmin, 
                                                                                n_iter=self.n_iter).ravel()

                # Computing the associated log proba term fixed correction covariance for the B_f sampling
                if self.perturbation_eta_covariance:
                    # _, inverse_term = func_logproba_eta(mixing_matrix_sampled, 
                    #                                             new_carry['eta_maps'], 
                    #                                             red_cov_approx_matrix_sqrt, 
                    #                                             previous_inverse=inverse_term, 
                    #                                             return_inverse=True,
                    #                                             precond_func=precond_func_eta)
                    _, inverse_term = func_logproba_eta(invBtinvNB,
                                                        new_carry['eta_maps'], 
                                                        red_cov_approx_matrix_sqrt, 
                                                        previous_inverse=inverse_term, 
                                                        return_inverse=True,
                                                        precond_func=precond_func_eta)
                if not(self.very_cheap_save):
                    all_samples['eta_maps'] = new_carry['eta_maps']

            # Sampling step 2 : sampling of Gaussian variable s_c, contrained CMB map realization

            ## Geting the Wiener filter term, mean of the variable s_c
            red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(carry['red_cov_matrix_sample'])

            # Preparing the preconditioner
            precond_func_s_c = None
            if use_precond:
                noise_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled)[0,0]
                red_inv_noise_c_ell = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(jnp.stack([noise_c_ell, noise_c_ell, jnp.zeros_like(noise_c_ell)])))#[self.lmin:]
                red_preconditioner_s_c = jnp.linalg.pinv(jnp.eye(self.nstokes) 
                                                        + jnp.einsum('lij,ljk,lkm->lim', 
                                                        red_cov_matrix_sqrt, 
                                                        red_inv_noise_c_ell, 
                                                        red_cov_matrix_sqrt))

                precond_func_s_c = lambda x: maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), 
                                                                            red_preconditioner_s_c, 
                                                                            nside=self.nside, 
                                                                            lmin=self.lmin, 
                                                                            n_iter=self.n_iter).ravel()

            initial_guess_WF = maps_x_red_covariance_cell_JAX(carry['wiener_filter_term'], 
                                                              jnp.linalg.pinv(red_cov_matrix_sqrt), 
                                                              nside=self.nside, 
                                                              lmin=self.lmin, 
                                                              n_iter=self.n_iter)
            # initial_guess_WF = jnp.zeros_like(s_cML)
            new_carry['wiener_filter_term'] = sampling_func_WF(s_cML, 
                                                               red_cov_matrix_sqrt, 
                                                               invBtinvNB, 
                                                               initial_guess=initial_guess_WF,
                                                               precond_func=precond_func_s_c)

            ## Preparing the random variables for the fluctuation term
            PRNGKey, new_subPRNGKey = random.split(PRNGKey)
            map_random_realization_xi = None
            map_random_realization_chi = None

            ## Getting the fluctuation maps terms, for the variance of the variable s_c
            initial_guess_Fluct = maps_x_red_covariance_cell_JAX(carry['fluctuation_maps'], 
                                                                jnp.linalg.pinv(red_cov_matrix_sqrt), 
                                                                nside=self.nside, 
                                                                lmin=self.lmin, 
                                                                n_iter=self.n_iter)
            # initial_guess_Fluct = jnp.zeros_like(carry['fluctuation_maps'])
            new_carry['fluctuation_maps'] = sampling_func_Fluct(red_cov_matrix_sqrt, 
                                                                invBtinvNB, 
                                                                BtinvN_sqrt, 
                                                                new_subPRNGKey, 
                                                                map_random_realization_xi=map_random_realization_xi, 
                                                                map_random_realization_chi=map_random_realization_chi, 
                                                                initial_guess=initial_guess_Fluct,
                                                                precond_func=precond_func_s_c)

            s_c_sample = new_carry['fluctuation_maps'] + new_carry['wiener_filter_term']

            if not(self.very_cheap_save):
                all_samples['wiener_filter_term'] = new_carry['wiener_filter_term']
                all_samples['fluctuation_maps'] = new_carry['fluctuation_maps']

            ## Checking the shape of the resulting maps
            chx.assert_shape(new_carry['wiener_filter_term'], (self.nstokes, self.n_pix))
            chx.assert_shape(new_carry['fluctuation_maps'], (self.nstokes, self.n_pix))
            chx.assert_shape(s_c_sample, (self.nstokes, self.n_pix))


            # Sampling step 3 : sampling of CMB covariance C
            
            ## Preparing the c_ell which will be used for the sampling
            c_ells_Wishart_ = get_cell_from_map_jax(s_c_sample, lmax=self.lmax, n_iter=self.n_iter)
            
            ### Getting them in the format [lmax,nstokes,nstokes]
            red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_*(2*jnp.arange(self.lmax+1) + 1))[self.lmin:]

            ## Preparing the new PRNGkey
            PRNGKey, new_subPRNGKey_2 = random.split(PRNGKey)

            ## Performing the sampling
            if self.sample_C_inv_Wishart:
                # Sampling C with inverse Wishart
                new_carry['red_cov_matrix_sample'] = func_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_[:,self.lmin:], 
                                                                    PRNGKey=new_subPRNGKey_2, 
                                                                    old_sample=carry['red_cov_matrix_sample'], 
                                                                    acceptance_posdef=self.acceptance_posdef)
                all_samples['red_cov_matrix_sample'] = new_carry['red_cov_matrix_sample']

            elif self.sample_r_Metropolis:
                # Sampling r which will parametrize C(r) = C_scalar + r*C_tensor
                new_carry['r_sample'] = r_sampling_MH(random_PRNGKey=new_subPRNGKey_2, old_sample=carry['r_sample'],
                                                          step_size=self.step_size_r, log_proba=self.get_conditional_proba_C_from_r, 
                                                          red_sigma_ell=red_c_ells_Wishart_modified, 
                                                          theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, 
                                                          theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)

                new_carry['red_cov_matrix_sample'] = theoretical_red_cov_r0_total + new_carry['r_sample']*theoretical_red_cov_r1_tensor
                all_samples['r_sample'] = new_carry['r_sample']
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')

            ## Checking the shape of the resulting covariance matrix, and correcting it if needed
            if new_carry['red_cov_matrix_sample'].shape[0] == self.lmax + 1:
                new_carry['red_cov_matrix_sample'] = new_carry['red_cov_matrix_sample'][self.lmin:]

            chx.assert_shape(new_carry['red_cov_matrix_sample'], (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))


            # Sampling step 4 : sampling of mixing matrix B_f

            ## Preparation of sampling step 4

            full_data_without_CMB = input_freq_maps - jnp.einsum('f,sp->fsp', jnp.ones(self.n_frequencies), s_c_sample)
            chx.assert_shape(full_data_without_CMB, (self.n_frequencies, self.nstokes, self.n_pix))

            ## Preparing the new PRNGKey
            PRNGKey, new_subPRNGKey_3 = random.split(PRNGKey)
            
            ## Performing the sampling
            if self.sample_eta_B_f:
                # Preparing the step-size
                step_size_Bf = initial_step_size_Bf

                # Sampling B_f
                if self.perturbation_eta_covariance or self.biased_version:
                    inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.n_pix), 
                                                                                 red_cov_approx_matrix_sqrt, 
                                                                                 nside=self.nside, 
                                                                                 lmin=self.lmin, 
                                                                                 n_iter=self.n_iter
                                                                                 ).ravel()
                    new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'] = sampling_func(random_PRNGKey=new_subPRNGKey_3, old_sample=carry['params_mixing_matrix_sample'], 
                                                            step_size=step_size_Bf, indexes_Bf=self.indexes_free_Bf,
                                                            log_proba=jitted_Bf_func_sampling,
                                                            full_data_without_CMB=full_data_without_CMB, component_eta_maps=new_carry['eta_maps'], 
                                                            red_cov_approx_matrix_sqrt=red_cov_approx_matrix_sqrt, previous_inverse=inverse_term,
                                                            previous_inverse_x_Capprox_root=inverse_term_x_Capprox_root,
                                                            old_params_mixing_matrix=carry['params_mixing_matrix_sample'],
                                                            biased_bool=self.biased_version)
                else:
                    new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'], inverse_term = sampling_func(random_PRNGKey=new_subPRNGKey_3, old_sample=carry['params_mixing_matrix_sample'], 
                                                            step_size=step_size_Bf, indexes_Bf=self.indexes_free_Bf,
                                                            log_proba=jitted_Bf_func_sampling,
                                                            full_data_without_CMB=full_data_without_CMB, component_eta_maps=new_carry['eta_maps'], 
                                                            red_cov_approx_matrix_sqrt=red_cov_approx_matrix_sqrt,
                                                            previous_inverse=inverse_term,
                                                            biased_bool=self.biased_version,
                                                            precond_func=precond_func_eta)
                
                all_samples['params_mixing_matrix_sample'] = new_carry['params_mixing_matrix_sample']

                # Checking the shape of the resulting mixing matrix
                chx.assert_axis_dimension(new_carry['params_mixing_matrix_sample'], 0, self.n_frequencies-len_pos_special_freqs)
                chx.assert_axis_dimension(new_carry['params_mixing_matrix_sample'], 1, self.n_correlations-1)
                # params_mixing_matrix_sample = params_mixing_matrix_sample.reshape((self.n_frequencies-len_pos_special_freqs,n_correlations-1),order='F')
            else:
                new_carry['params_mixing_matrix_sample'] = carry['params_mixing_matrix_sample']
                all_samples['params_mixing_matrix_sample'] = new_carry['params_mixing_matrix_sample']

            new_carry['inverse_term'] = inverse_term
            new_carry['PRNGKey'] = PRNGKey

            # new_carry = (new_eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample, PRNGKey, s_c_sample, inverse_term)
            # all_samples = (new_eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample)

            return new_carry, all_samples

        # Initializing r and B_f samples

        # initial_carry = (initial_eta, 
        #                  wiener_filter_term, fluctuation_maps, 
        #                  red_cov_matrix,
        #                  initial_guess_r,
        #                  params_mixing_matrix_init_sample,
        #                  PRNGKey,
        #                  wiener_filter_term+fluctuation_maps,
        #                  jnp.zeros_like(initial_eta))
        # initial_carry_0 = (initial_eta, 
        #                  wiener_filter_term, fluctuation_maps, 
        #                  red_cov_matrix,
        #                  initial_guess_r,
        #                  params_mixing_matrix_init_sample,
        #                  PRNGKey)

        initial_carry = {'wiener_filter_term': wiener_filter_term,
                            'fluctuation_maps': fluctuation_maps,
                            'red_cov_matrix_sample': red_cov_matrix,
                            'params_mixing_matrix_sample': params_mixing_matrix_init_sample,
                            'PRNGKey': PRNGKey,
                            'inverse_term': jnp.zeros_like(initial_eta)}

        if self.sample_eta_B_f:
            initial_carry['eta_maps'] = initial_eta
        if self.sample_r_Metropolis:
            initial_carry['r_sample'] = initial_guess_r
        
        self.update_one_sample(initial_carry)

        # jitted_all_sampling_steps = jax.jit(all_sampling_steps)

        time_start_sampling = time.time()
        # Start sampling !!!
        # last_sample, all_samples = jlx.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        
        # last_sample, all_samples = jlx.scan(jitted_all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        last_sample, all_samples = jlx.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))

        # last_sample, all_samples = my_scan(all_sampling_steps, initial_carry, None, length=actual_number_of_iterations)
        time_full_chain = (time.time()-time_start_sampling)/60      
        print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

        # Saving the samples as attributes of the Sampler object
        self.update_samples(all_samples)
        self.number_iterations_done = self.number_iterations_sampling


    def continue_sampling(self,input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=0, initial_wiener_filter_term=jnp.empty(0), initial_fluctuation_maps=jnp.empty(0),
                         theoretical_r0_total=jnp.empty(0), theoretical_r1_tensor=jnp.empty(0)):

        if self.sample_r_Metropolis:
            CMB_c_ell = theoretical_r0_total + initial_guess_r*theoretical_r1_tensor

        self.perform_Gibbs_sampling(self, input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=initial_guess_r, initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
                         theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)


def create_MICMAC_sampler_from_toml_file(path_toml_file, path_file_spv=''):
    """ Create a MICMAC_Sampler object from:
    * the path of a toml file: params for the sims and for the sampling
    * the path of a spv file: params for addressing spatial variability
    """
    ### Sims and samplig params
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    if dictionary_parameters['instrument_name'] != 'customized_instrument':
        instrument = fgbuster.get_instrument(dictionary_parameters['instrument_name'])
        dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
        dictionary_parameters['freq_inverse_noise'] = get_noise_covar(instrument['depth_p'], dictionary_parameters['nside'])

    ### Spatial variability params
    n_fgs_comp = dictionary_parameters['n_components']-1
    # total number of params in the mixing matrix for a specific pixel
    n_betas = (np.shape(dictionary_parameters['frequency_array'])[0]-len(dictionary_parameters['pos_special_freqs']))*(n_fgs_comp)
    # Read or create spv config
    root_tree = tree_spv_config(path_file_spv, n_betas, n_fgs_comp, print_tree=True)
    dictionary_parameters['spv_nodes_b'] = get_nodes_b(root_tree)
    
    # del dictionary_parameters['instrument_name']
    return MICMAC_Sampler(**dictionary_parameters)
