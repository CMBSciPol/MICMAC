# This file is part of MICMAC.
# Copyright (C) 2024 CNRS / SciPol developers
#
# MICMAC is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MICMAC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MICMAC. If not, see <https://www.gnu.org/licenses/>.
"""
Module to initialize the Mixing Matrix
with parameter values coming from the SEDs.
"""

from collections.abc import Iterable

import healpy as hp
import numpy as np
from astropy.cosmology import Planck15
from scipy import constants

from micmac.foregrounds.templates import get_n_patches_b

__all__ = ['InitMixingMatrix']

h_over_k = constants.h * 1e9 / constants.k


class InitMixingMatrix:
    def __init__(
        self,
        freqs,
        ncomp,
        pos_special_freqs,
        spv_nodes_b,
        nside=None,
        non_param_fgs_mixing_matrix=None,
        beta_pl=-3.0,
        beta_mbb=1.54,
        temp_mbb=20.0,
    ):
        """

        The main goal of this class is create the initial parameter values,
        with init_params, which returns the params as a flattened array of values ordered
        by components, freqs, patches.

        Note:
        * units are K_CMB.
        * you must be always coherent with the order of fgs:
          e.g. if you put first synchrotron and then dust in the mixing matrix (A)
               that you pass to this class, then also in the pos_special_freqs, etc.

        Parameters
        ----------
        freqs: array
            all input freq bands
        ncomp: int
            all comps (also cmb)
        pos_special_freqs: array
            indices of the freqs that are known
        spv_nodes_b: array
            tree containing info to build spv_templates
        nside: int
            nside of the map
        non_param_fgs_mixing_matrix: array
            only the fgs part of mixing matrix (A)
        beta_pl: float or array[float]
            spectral idx of the synchrotron power law (pl), if given as array it is expected to correspond to different patches
        beta_mbb: float or array[float]
            spectral idx of the dust modified black body (mbb), if given as array it is expected to correspond to different patches
        temp_mbb: float or array[float]
            temperature of the dust  modified black body (mbb), if given as array it is expected to correspond to different patches
        """
        self.freqs = freqs  # all input freq bands
        self.ncomp = ncomp  # all comps (also cmb)
        # make pos_special_freqs only positive
        for i, val_i in enumerate(pos_special_freqs):
            if val_i < 0:
                pos_special_freqs[i] = len(self.freqs) + pos_special_freqs[i]
        self.pos_special_freqs = pos_special_freqs
        self.spv_nodes_b = spv_nodes_b  # tree containing info to build spv_templates
        self.nside = nside  # nside of the map
        self.non_param_fgs_mixing_matrix = non_param_fgs_mixing_matrix  # only the fgs part of mixing matrix
        self.beta_mbb = np.array(beta_mbb)
        self.temp_mbb = np.array(temp_mbb)
        self.beta_pl = np.array(beta_pl)

        self.known_freqs = np.array(
            [
                self.freqs[self.pos_special_freqs[0]],
                self.freqs[self.pos_special_freqs[1]],
            ]
        )
        self.unknown_freqs = np.delete(self.freqs, self.pos_special_freqs)

    def init_params(self):
        """
        Main function to initialize the mixing matrix B.
        * if non_param_mixing_matrix (A) is passed that is used to build the init values
        * otherwise the init values are build from beta_mbb, temp_mbb, beta_pl

        Returns
        -------
        init_param_values: array
            initial parameter values as a flattened array of values ordered as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]
        """
        # Get params of B pixel indep or with pixel dimension
        if isinstance(self.non_param_fgs_mixing_matrix, Iterable):
            # Get params from fgs mixing matrix (A) passed by the user
            print('>>> init params built from fgs mixing matrix (A) passed by the user:')
            fgs_SEDs = self.non_param_fgs_mixing_matrix
        else:
            print('>>> init params built with spectral params:', self.beta_mbb, self.temp_mbb, self.beta_pl)
            fgs_SEDs = self.fgs_SEDs_from_spectral_params()
        init_param_values_ = self.init_params_from_fgs_SEDs(fgs_SEDs)
        # Make one parameter per patch
        init_param_values = self.ud_grade_init_params_to_npatches(init_param_values_)

        return init_param_values

    def fgs_SEDs_from_spectral_params(self):
        """
        Get the SEDs of the foregrounds (fgs) from the spectral parameters.
        Note: In Mixing Matrix we consider first synch then dust

        Returns
        -------
        fgs_SEDs: array
            SEDs of the fgs
        """
        fgs_SEDs_pl = self.powerlaw(self.known_freqs[0], self.freqs[:, None], self.beta_pl)
        fgs_SEDs_mbb = self.modifiedblackbody(self.known_freqs[1], self.freqs[:, None], self.temp_mbb, self.beta_mbb)
        fgs_SEDs = np.stack((fgs_SEDs_pl, fgs_SEDs_mbb), axis=1)

        return fgs_SEDs

    def init_params_from_fgs_SEDs(self, fgs_SEDs):
        """
        Get params w SEDs for redefined fgs (true B entries)
        B = A invM
        in A only fgs entries are needed (no CMB column)
        in M only the fgs entries are needed (only A_f1)
        build M

        Parameters
        ----------
        fgs_SEDs: array
            SEDs of the fgs
        """
        # Get A_f1
        A_f1 = fgs_SEDs[[self.pos_special_freqs[0], self.pos_special_freqs[-1]], :, :]
        # Get inverse of A_f1
        inv_A_f1 = np.linalg.inv(A_f1.swapaxes(0, -1)).swapaxes(0, -1)

        return np.einsum('fc...,cg...->fg...', fgs_SEDs, inv_A_f1)

    def ud_grade_init_params_to_npatches(self, params_):
        """
        Take one mixing matrix value on the full sky or one per pixel
        and returns one mixing matrix value per patch.

        Parameters
        ----------
        params_: array
            mixing matrix values
        """
        assert self.ncomp - 1 == 2
        # Modify to have a value per patch
        params_s = []
        params_d = []
        ind_unknown_f = 0
        for ind_f, val_f in enumerate(self.freqs):
            if ind_f in self.pos_special_freqs:
                pass
            else:
                # TODO: extend the counting of the patches to adaptive multires
                # (maybe easier to extend it by looking at the spv_templates
                # or add a function in the templates_spv.py to get the number of patches for each b)
                n_patches_b_s = get_n_patches_b(self.spv_nodes_b[ind_unknown_f])
                n_patches_b_d = get_n_patches_b(self.spv_nodes_b[ind_unknown_f + len(self.unknown_freqs)])
                if params_.shape[2] == 1:
                    ### Synchrotron
                    for patch in range(n_patches_b_s):
                        # UPgrade to params spv (one parameter per patch)
                        params_s.append(params_[ind_f, 0])
                    ### Dust
                    for patch in range(n_patches_b_d):
                        # UPgrade to params spv (one parameter per patch)
                        params_d.append(params_[ind_f, 1])
                else:
                    # DOWNgrade to params spv (one parameter per patch)
                    nside_b_s = int(np.sqrt(n_patches_b_s / 12))
                    if nside_b_s == 0:
                        params_s.append(np.mean(params_[ind_f, 0, :]))
                    else:
                        [params_s.append(item) for item in hp.ud_grade(params_[ind_f, 0, :], nside_b_s)]
                    nside_b_d = int(np.sqrt(n_patches_b_d / 12))
                    if nside_b_d == 0:
                        params_d.append(np.mean(params_[ind_f, 1, :]))
                    else:
                        [params_d.append(item) for item in hp.ud_grade(params_[ind_f, 1, :], nside_b_d)]
                ind_unknown_f += 1
        params = np.concatenate((params_s, params_d)).flatten()

        return params

    ### Functions to get the parametric scaling laws
    def K_rj2K_cmb(self, nu):
        """
        Conversion factor from K_rj to K_CMB at frequency nu

        Parameters
        ----------
        nu: float or array[float]
            frequency in GHz

        Returns
        -------
        K_rj2K_cmb: float or array[float]
            conversion factor from K_rj to K_CMB at frequency nu
        """
        Tcmb = Planck15.Tcmb(0).value
        # Conversion factor at frequency nu
        K_rj2K_cmb = np.expm1(h_over_k * nu / Tcmb) ** 2 / (np.exp(h_over_k * nu / Tcmb) * (h_over_k * nu / Tcmb) ** 2)

        return K_rj2K_cmb

    def K_rj2K_cmb_nu0(self, nu, nu0):
        """
        Conversion factor at frequency nu divided by the one at frequency nu0

        Parameters
        ----------
        nu: float or array[float]
            frequency in GHz
        nu0: float or array[float]
            reference frequency in GHz

        Returns
        -------
        K_rj2K_cmb_nu0: float or array[float]
            conversion factor at frequency nu divided by the one at frequency nu0
        """
        K_rj2K_cmb_nu0 = self.K_rj2K_cmb(nu) / self.K_rj2K_cmb(nu0)

        return K_rj2K_cmb_nu0

    def powerlaw(self, nu0, nu, beta_pl):
        """
        Scaling lawing law for synchrotron power law (pl) in K_CMB

        Parameters
        ----------
        nu0: float or array[float]
            reference frequency in GHz
        nu: float or array[float]
            frequency in GHz
        beta_pl: array
            spectral idx of the pl

        Returns
        -------
        analytic_expr: array[float]
            scaling law for pl (in K_CMB)
        """
        analytic_expr = (nu / nu0) ** (beta_pl)
        # conversion to K_CMB units
        analytic_expr *= self.K_rj2K_cmb_nu0(nu, nu0)

        return analytic_expr

    def modifiedblackbody(self, nu0, nu, temp_mbb, beta_mbb):
        """
        Scaling law for modified balck body (mbb) in K_CMB

        Parameters
        ----------
        nu0: float or array[float]
            reference frequency in GHz
        nu: float or array[float]
            frequency in GHz
        temp_mbb: array
            temperature of the mbb
        beta_mbb: array
            spectral idx of the mbb

        Returns
        -------
        analytic_expr: array[float]
            scaling law for dust mbb (in K_CMB)
        """
        analytic_expr = (
            (np.exp(nu0 / temp_mbb * h_over_k) - 1)
            / (np.exp(nu / temp_mbb * h_over_k) - 1)
            * (nu / nu0) ** (1 + beta_mbb)
        )
        # conversion to K_CMB units
        analytic_expr *= self.K_rj2K_cmb_nu0(nu, nu0)

        return analytic_expr
