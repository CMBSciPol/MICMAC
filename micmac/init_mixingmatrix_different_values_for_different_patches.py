"""
Module to initialize the Mixing Matrix 
with parameter values coming from the SEDs.
"""

import numpy as np
import healpy as hp
import sympy
from pysm3 import Sky
from scipy import constants
from astropy.cosmology import Planck15
from collections.abc import Iterable

from .templates_spv import create_one_template, get_n_patches_b


h_over_k = constants.h * 1e9 / constants.k


def get_d1s1_spectral_params_values(nside):
    """For model d1s1 get the spectral parameter values 
    one per pixel of the input frequency maps

    nside: nside of input freq maps
    """
    # Dust
    sky = Sky(nside=nside, preset_strings=["d1"])
    dust = sky.components[0]
    beta_mbb = dust.mbb_index
    beta_mbb_fin = hp.ud_grade(beta_mbb, nside)   # TODO: replace ud_grade?
    temp_mbb = dust.mbb_temperature
    temp_mbb_fin = hp.ud_grade(temp_mbb, nside)   # TODO: replace ud_grade?

    # Synchrotron
    sky = Sky(nside=nside, preset_strings=["s1"])
    synch = sky.components[0]
    beta_pl = synch.pl_index
    beta_pl_fin = hp.ud_grade(beta_pl, nside)   # TODO: replace ud_grade?

    return beta_mbb_fin, temp_mbb_fin, beta_pl_fin


class InitMixingMatrix:
    def __init__(self, freqs, ncomp, pos_special_freqs, spv_nodes_b, 
                 nside=None, beta_mbb=[1.54], temp_mbb=[20.0], beta_pl=[-3.0]):
        """
        Notes: 
        * units are K_CMB.
        * beta_mbb, temp_mbb, beta_pl: 
            - iterable with one entry --> same value on the whole sky
            - iterable with one value per pixel
        """
        self.freqs = freqs  # all input freq bands
        self.ncomp = ncomp  # all comps (also cmb)
        self.pos_special_freqs = pos_special_freqs
        self.spv_nodes_b = spv_nodes_b   # tree containing info to build spv_templates
        self.nside = nside
        self.beta_mbb = np.array(beta_mbb)
        self.temp_mbb = np.array(temp_mbb)
        self.beta_pl = np.array(beta_pl)

    def K_rj2K_cmb(self, nu):
        """Conversion factor from K_rj to K_CMB at frequency nu
        """
        Tcmb = Planck15.Tcmb(0).value
        # Conversion factor at frequency nu
        K_rj2K_cmb = np.expm1(h_over_k * nu / Tcmb) ** 2 / (
            np.exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb) ** 2
        )

        return K_rj2K_cmb

    def K_rj2K_cmb_nu0(self, nu, nu0):
        """Conversion factor at frequency nu divided by the one at frequency nu0
        """
        K_rj2K_cmb_nu0 = self.K_rj2K_cmb(nu) / self.K_rj2K_cmb(nu0)

        return K_rj2K_cmb_nu0

    def powerlaw(self, nu0, nu, beta_pl):
        """Scaling lawing law for pl (in K_CMB)

        beta_pl (arr): spectral idx of the pl
        """
        analytic_expr = (nu / nu0) ** (beta_pl)
        # conversion to K_CMB units
        analytic_expr *= self.K_rj2K_cmb_nu0(nu, nu0)

        return analytic_expr
    
    def modifiedblackbody(self, nu0, nu, temp_mbb, beta_mbb):
        """Scaling law for mbb (in K_CMB)

        temp_mbb (arr): temperature of the mbb
        beta_mbb (arr): spectral idx of the mbb
        """
        analytic_expr = (
            (np.exp(nu0 / temp_mbb * h_over_k) - 1)
            / (np.exp(nu / temp_mbb * h_over_k) - 1)
            * (nu / nu0) ** (1 + beta_mbb)
        )
        # conversion to K_CMB units
        analytic_expr *= self.K_rj2K_cmb_nu0(nu, nu0)

        return analytic_expr

    def init_params_fullsky_or_all_pix(self, beta_pl, temp_mbb, beta_mbb):
        # Get array of known and unknown frequencies
        unknown_freqs = np.delete(self.freqs, self.pos_special_freqs)
        known_freqs = np.array(
            [
                self.freqs[self.pos_special_freqs[0]],
                self.freqs[self.pos_special_freqs[1]],
            ]
        )
        assert list(known_freqs) == [f for f in self.freqs if f not in unknown_freqs]
        # Get reference frequency nu0
        nu0_dust = self.freqs[self.pos_special_freqs[0]]
        nu0_synch = self.freqs[self.pos_special_freqs[1]]

        # Define the params w true fgs SEDs (true A entries)
        if len(beta_pl)==1 and len(beta_mbb)==1 and len(temp_mbb)==1:
            params_ = np.zeros((len(unknown_freqs), self.ncomp - 1, 1))
            A_f1 = np.zeros((self.ncomp - 1, self.ncomp - 1, 1))
        else:
            assert self.nside
            params_ = np.zeros((len(unknown_freqs), self.ncomp - 1, 12*self.nside**2))
            A_f1 = np.zeros((self.ncomp - 1, self.ncomp - 1, 12*self.nside**2))
        for f, val_f in enumerate(unknown_freqs):
            params_[f, 0, :] = self.powerlaw(nu0_synch, val_f, beta_pl)
            params_[f, 1, :] = self.modifiedblackbody(nu0_dust, val_f, temp_mbb, beta_mbb)
        # Get A_f1 (SEDs of the fgs at the unknown freqs)
        for j, val_j in enumerate(known_freqs):
            A_f1[j, 0, :] = self.powerlaw(nu0_synch, val_j, beta_pl)
            A_f1[j, 1, :] = self.modifiedblackbody(nu0_dust, val_j, temp_mbb, beta_mbb)
        inv_A_f1 = np.linalg.inv(A_f1.swapaxes(0,-1)).swapaxes(0,-1)
        # Get params w SEDs for redefined fgs (true B entries)
        # B = A invM
        params_no_spv = np.einsum("fc...,cg...->fg...", params_, inv_A_f1)
        print('params_no_spv', params_no_spv)
        
        return unknown_freqs, params_no_spv
    
    def init_params_synch_dust(self):
        """Give values for the parameters of the redefined mixing matrix B.
        Values from the parametric mixing matrix with 1st_fgs=synch(PL) and 2nd_fgs=dust(MBB),
        the spectral parameter values are by default Bs=-3.0, Bd=1.54, Td=20K,
        otherwise can be passed by the user a single value on the full sky
        or a value per pixel.
        """
        assert self.ncomp - 1 == 2
        
        # Get params of B pixel indep or with pixel dimension
        unknown_freqs, params_ = self.init_params_fullsky_or_all_pix(self.beta_pl, self.temp_mbb, self.beta_mbb)

        # Modify to have a value per patch
        params_s = []
        params_d = []
        for ind_f, val_f in enumerate(unknown_freqs):
            # TODO: extend the counting of the patches to adaptive multires
            # (maybe easier to extend it by looking at the spv_templates
            # or add a function in the templates_spv.py to get the number of patches for each b)
            n_patches_b_s = get_n_patches_b(self.spv_nodes_b[ind_f])
            n_patches_b_d = get_n_patches_b(self.spv_nodes_b[ind_f+len(unknown_freqs)])
            
            if len(self.beta_pl)==1 and len(self.beta_mbb)==1 and len(self.temp_mbb)==1:
                for patch in range(n_patches_b_s):
                    # UPgrade to params spv (one parameter per patch)
                    params_s.append(params_[ind_f, 0])
            else:
                # DOWNgrade to params spv (one parameter per patch)
                print('params_[ind_f, 0, :]', params_[ind_f, 0, :].shape)
                nside_b_s = int(np.sqrt(n_patches_b_s/12))
                if nside_b_s == 0: params_s.append(np.mean(params_[ind_f, 0, :]))
                else: [params_s.append(item) for item in hp.ud_grade(params_[ind_f, 0, :], nside_b_s)]   # TODO: decide if keeping ud_grade

            if len(self.beta_pl)==1 and len(self.beta_mbb)==1 and len(self.temp_mbb)==1:
                for patch in range(n_patches_b_d):
                    # UPgrade to params spv (one parameter per patch)
                    params_d.append(params_[ind_f, 1])
            else:
                # DOWNgrade to params spv (one parameter per patch)
                nside_b_d = int(np.sqrt(n_patches_b_d/12))
                if nside_b_d == 0: params_d.append(np.mean(params_[ind_f, 1, :]))
                else: [params_d.append(item) for item in hp.ud_grade(params_[ind_f, 1, :], nside_b_d)]  # TODO: decide if keeping ud_grade
        
        params = np.concatenate((params_s, params_d))

        return params
