"""
Module to initialize the Mixing Matrix 
with parameter values coming from the SEDs.
"""

import numpy as np
import sympy
from scipy import constants
from astropy.cosmology import Planck15


h_over_k = constants.h * 1e9 / constants.k



class InitMixingMatrix():
    def __init__(self, freqs, ncomp, pos_special_freqs):
        """
        Note: units are K_CMB.
        """
        self.freqs = freqs    # all input freq bands
        self.ncomp = ncomp    # all comps (also cmb)
        self.pos_special_freqs = pos_special_freqs


    def K_rj2K_cmb(self, nu):
        Tcmb=Planck15.Tcmb(0).value
        # Conversion factor at frequency nu
        K_rj2K_cmb = ((np.expm1(h_over_k * nu / Tcmb)**2
                     / (np.exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb)**2)))
        return K_rj2K_cmb
    

    def K_rj2K_cmb_nu0(self, nu, nu0):
        # Conversion factor at frequency nu divided by the one at frequency nu0
        K_rj2K_cmb_nu0 = self.K_rj2K_cmb(nu) / self.K_rj2K_cmb(nu0)
        return K_rj2K_cmb_nu0
    

    def modifiedblackbody(self, nu0, nu):
        """
        in K_CMB
        """
        beta_mbb = 1.54
        temp = 20.

        analytic_expr = ((np.exp(nu0 / temp * h_over_k) -1)
                         / (np.exp(nu / temp * h_over_k) - 1)
                         * (nu / nu0)**(1 + beta_mbb))
        # conversion to K_CMB units
        analytic_expr *= self.K_rj2K_cmb_nu0(nu, nu0)
        
        return analytic_expr


    def powerlaw(self, nu0, nu):
        """
        in K_CMB
        """
        beta_pl = -3.

        analytic_expr = (nu / nu0)**(beta_pl)
        # conversion to K_CMB units
        analytic_expr *= self.K_rj2K_cmb_nu0(nu, nu0)
        
        return analytic_expr
    

    def init_params(self):
        # implemented only the case 1st_fgs=dust(MBB), 2nd_fgs=synch(PL)
        assert self.ncomp-1 == 2
        # remove the special freqs
        unknown_freqs = np.delete(self.freqs, self.pos_special_freqs)
        # get nu0
        nu0_dust = self.freqs[self.pos_special_freqs[0]]
        nu0_synch = self.freqs[self.pos_special_freqs[1]]
        # define the params
        params = np.zeros((len(unknown_freqs),self.ncomp-1))
        for f, val_f in enumerate(unknown_freqs):
            params[f, 0] = self.modifiedblackbody(nu0_dust, val_f)
            params[f, 1] = self.powerlaw(nu0_synch, val_f)

        return params
