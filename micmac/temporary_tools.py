import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import healpy as hp
import astropy.io.fits as fits
import camb

def generate_power_spectra_CAMB(Nside,  r=0, Alens=1, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, ns=0.965, As=2e-9, lens_potential_accuracy=1, nt=0, ntrun=0, type_power='total', typeless_bool=False):
    """
    Return [Cl^TT, Cl^EE, Cl^BB, Cl^TE]
    """
    lmax = 2*Nside
    # pars = camb.CAMBparams(max_l_tensor=lmax, parameterization='tensor_param_indeptilt')
    pars = camb.CAMBparams(max_l_tensor=lmax)
    pars.WantTensors = True

    pars.Accuracy.AccurateBB = True
    pars.Accuracy.AccuratePolarization = True
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, Alens=Alens)
    pars.InitPower.set_params(As=As, ns=ns, r=r, parameterization='tensor_param_indeptilt', nt=nt, ntrun=ntrun)
    pars.max_eta_k_tensor = lmax + 100  # 15000  # 100

    # pars.set_cosmology(H0=H0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)

    print("Calculating spectra from CAMB !")
    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lmax=lmax)    
    if typeless_bool:
        return powers
    return powers[type_power]

# def read_par_file(path_file):
#     file_flux = np.load(path_file)
#     dict_obj = dict()
#     for key in file_flux.__dict__['files']:
#         dict_obj[key] = file_flux[key]
#     return dict_obj

def get_MCMC_batch_error(sample_single_chain, batch_size):
    # number_iterations = np.size(sample_single_chain, axis=0)
    number_iterations = sample_single_chain.shape[0]
    assert number_iterations%batch_size == 0

    overall_mean = np.average(sample_single_chain, axis=0)
    standard_error = np.sqrt((batch_size/number_iterations)*((sample_single_chain-overall_mean)**2).sum())
    return standard_error


def get_empirical_covariance_JAX(samples):
    """ Compute empirical covariance from samples
    """
    number_samples = jnp.size(samples, axis=0)
    mean_samples = jnp.mean(samples, axis=0)

    return (jnp.einsum('ti,tj->tij',samples,samples).sum(axis=0) - number_samples*jnp.einsum('i,j->ij',mean_samples,mean_samples))/(number_samples-1)
