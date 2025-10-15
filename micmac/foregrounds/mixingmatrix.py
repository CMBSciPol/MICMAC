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

import copy

import jax
import jax.numpy as jnp
import numpy as np

from micmac.foregrounds.templates import create_one_template

__all__ = ['get_indexes_b', 'MixingMatrix']

# Note:
# the mixing matrix is supposed to be the same for Q and U Stokes params
# (also we suppose that I is not used)
# Mixing matrix dimensions: n_frequencies*n_components*number_pixels


def get_indexes_b(templates):
    """
    Return indexes of params for all frequencies and components

    Parameters
    ----------
    n_frequencies: int
        Number of frequencies
    n_components: int
        Number of components
    templates: list
        List of nodes for b containing info patches to build spv_templates

    Returns
    -------
    indexes: array
        Indexes of params for all frequencies and components
    """

    return templates.min(axis=-1)


class MixingMatrix:
    def __init__(self, frequency_array, n_components, templates, nside, params=None, pos_special_freqs=[0, -1]):
        """
        Note: units are K_CMB.

        Parameters
        ----------
        frequency_array: array
            Array of frequencies
        n_components: int
            Number of components
        templates: array[int]
            Array maps with patch ids ([freq, comp, pix])
        nside: int
            Healpix nside of the expected input maps
        params: array (optional)
            Initial values of the parameters of the mixing matrix, default None (then initialized with zeros)
        pos_special_freqs: list (optional)
            List of indexes of special frequencies (e.g. 0 for synchrotron, -1 for dust)
        """
        self.nside = nside  # nside of the expected input maps
        self.frequency_array = np.array(frequency_array, dtype=int)  # all input freq bands
        self.n_frequencies = np.size(frequency_array)  # all input freq bands
        self.n_components = n_components  # all comps (also cmb)

        if templates is None:
            templates = np.zeros((self.n_frequencies, self.n_components, 12 * nside**2), dtype=int)
            for f in range(self.n_frequencies):
                for c in range(self.n_components - 1):
                    templates[f, c] = create_one_template(self.nside) + f * (self.n_components - 1) + c
        else:
            msg_error = f'templates must be of dimensions {(self.n_frequencies - self.n_components + 1, self.n_components - 1, 12 * nside**2)}'
            assert templates.shape == (
                self.n_frequencies - self.n_components + 1,
                self.n_components - 1,
                12 * nside**2,
            ), msg_error

            assert issubclass(templates.dtype.type, np.integer) or issubclass(
                templates.dtype.type, jnp.integer
            ), 'templates must be of integer type'

            for j in range(n_components - 1):
                j_idx = j

                for i in range(self.n_frequencies - n_components + 1):
                    if i == 0 and j == 0:
                        assert templates[i, j].min() == 0, 'templates values must start at 0'
                        continue
                    i_idx = i
                    if j != 0 and i == 0:
                        j_idx = 0
                        i_idx = -1
                    assert (
                        templates[i_idx, j_idx].min() > templates[i_idx - 1, j_idx].max()
                    ), f'templates values must be unique and increasing'
                    unique_template_sorted = np.sort(np.unique(templates[i, j]))
                    assert np.all(
                        unique_template_sorted[1:] - unique_template_sorted[:-1] == 1
                    ), f'templates values must be contiguous without lacking indices'

        self.templates = templates  # templates for all frequencies and components
        self.len_params = np.unique(
            self.templates
        ).size  # total number of free parameters (summed for frequency, component, patch)

        if params is None:
            params = np.zeros(self.len_params)
        else:
            assert params.size == self.len_params, f'params must be of dimensions {self.len_params}'

        self.params = params

        # Indexes frequency array without the special frequencies
        self.indexes_frequency_array_no_special = np.delete(np.arange(self.n_frequencies), pos_special_freqs)

        ### checks on pos_special_freqs

        # check no duplicates
        assert len(pos_special_freqs) == len(set(pos_special_freqs))

        # make pos_special_freqs only positive
        for i, val_i in enumerate(pos_special_freqs):
            if val_i < 0:
                pos_special_freqs[i] = self.n_frequencies + pos_special_freqs[i]
        self.pos_special_freqs = pos_special_freqs

        if self.n_components != 1:
            # Values of the first index of each Bf parameter in params
            self.indexes_b = jnp.array(get_indexes_b(self.templates))
            n_patches_array = np.zeros_like(self.indexes_b).ravel(order='F')
            n_patches_array[:-1] = self.indexes_b.ravel(order='F')[1:] - self.indexes_b.ravel(order='F')[:-1]
            n_patches_array[-1] = self.len_params - self.indexes_b[-1, -1]

            self.n_patches = jnp.array(n_patches_array)

            self.max_len_patches_Bf = int(self.n_patches.max())
            n_unknown_freqs = self.n_frequencies - self.n_components + 1
            n_comp_fgs = self.n_components - 1
            self.multipatch_bool = not (
                (self.n_patches == 1).all() and (self.len_params == n_comp_fgs * n_unknown_freqs)
            )
        else:
            self.indexes_b = jnp.array([[0]])  # Values of the first index of each Bf parameter in params
            self.n_patches = None  # Number of patches for each node
            # self.sum_n_patches_indexed_freq_comp = None  # Cumulative sum of the number of patches for each node
            self.max_len_patches_Bf = None  # Maximum number of patches for each node
            self.multipatch_bool = False

    @property
    def n_pix(self):
        """
        Number of pixels of one input freq map
        """
        return 12 * self.nside**2

    # def get_params_long(self, jax_use=False):
    #     """
    #     From the params to all the entries of the mixing matrix

    #     Parameters
    #     ----------
    #     jax_use: bool (optional)
    #         If True, params are expected as JAX Array, default False

    #     Returns
    #     -------
    #     params_long: array[float] of dimensions [n_frequencies - n_components + 1, n_components - 1, n_pix]
    #         Reshaped free parameters of the mixing matrix
    #     """

    #     if jax_use:
    #         templates_to_fill = self.get_all_templates()

    #         ## Filling the templates with parameters values
    #         return self.params.at[templates_to_fill].get()

    #     return self.get_params_long_python(self.params)

    # def get_B_fgs(self, jax_use=False):
    #     """
    #     Foreground part of the mixing matrix.

    #     Parameters
    #     ----------
    #     jax_use: bool (optional)
    #         If True, params are expected as JAX Array, default False

    #     Returns
    #     -------
    #     B_fgs: array[float] of dimensions [n_frequencies, n_components - 1, n_pix]
    #         Foreground part of the mixing matrix (including special frequencies)
    #     """
    #     ncomp_fgs = self.n_components - 1
    #     params_long = self.get_params_long(jax_use=jax_use)

    #     if jax_use:
    #         B_fgs = jnp.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
    #         # insert all the ones given by the pos_special_freqs
    #         B_fgs = B_fgs.at[jnp.array(self.pos_special_freqs), ...].set(
    #             jnp.broadcast_to(jnp.eye(ncomp_fgs), (self.n_pix, ncomp_fgs, ncomp_fgs)).T
    #         )
    #         # insert all the parameters values
    #         B_fgs = B_fgs.at[self.indexes_frequency_array_no_special, ...].set(params_long)
    #         return B_fgs

    #     if ncomp_fgs != 0:
    #         assert params_long.shape == ((self.n_frequencies - len(self.pos_special_freqs)), ncomp_fgs, self.n_pix)
    #         assert len(self.pos_special_freqs) <= ncomp_fgs

    #     B_fgs = np.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
    #     if len(self.pos_special_freqs) != 0:
    #         # insert all the ones given by the pos_special_freqs
    #         for c in range(len(self.pos_special_freqs)):
    #             B_fgs[self.pos_special_freqs[c]][c] = 1
    #     # insert all the parameters values
    #     f = 0
    #     for i in range(self.n_frequencies):
    #         if i not in self.pos_special_freqs:
    #             B_fgs[i, :] = params_long[f, :, :]
    #             f += 1

    #     return B_fgs

    def get_B_cmb(self, jax_use=True):
        """
        CMB column of the mixing matrix.

        Parameters
        ----------
        jax_use: bool (optional)
            If True, returned as JAX Array, default False

        Returns
        -------
        B_cmb: array[float] of dimensions [n_frequencies, 1, n_pix]
            CMB column of the mixing matrix, filled with ones
        """
        if jax_use:
            B_cmb = jnp.ones((self.n_frequencies, self.n_pix))
            return B_cmb[:, np.newaxis, :]

        B_cmb = np.ones((self.n_frequencies, self.n_pix))
        B_cmb = B_cmb[:, np.newaxis, :]

        return B_cmb

    # def get_B(self, jax_use=False):
    #     """
    #     Full mixing matrix, (n_frequencies*n_components).
    #     CMB is given as the first component.

    #     Parameters
    #     ----------
    #     jax_use: bool (optional)
    #         If True, returned as JAX Array, default False

    #     Returns
    #     -------
    #     B_mat: array[float] of dimensions [n_frequencies, n_components, n_pix]
    #         Full mixing matrix
    #     """
    #     if jax_use:
    #         if self.n_components != 1:
    #             return jnp.concatenate((self.get_B_cmb(jax_use=jax_use), self.get_B_fgs(jax_use=jax_use)), axis=1)
    #         else:
    #             return self.get_B_cmb(jax_use=jax_use)
    #     if self.n_components != 1:
    #         B_mat = np.concatenate((self.get_B_cmb(), self.get_B_fgs()), axis=1)
    #     else:
    #         B_mat = self.get_B_cmb()
    #     return B_mat

    def get_B_fgs_from_params(self, params, jax_use=True):
        """
        Foreground part of the mixing matrix obtained from the parameters.

        Parameters
        ----------
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [Bf1_comp1_patch1, Bf1_comp1_patch2, ..., Bf2_comp1_patch1, ..., Bf1_comp2_patch1, ..., Bfn_comp2_patchn, ...]
        jax_use: bool (optional)
            If True, params are expected as JAX Array and B_fgs will be returned as JAX Array, default False

        Returns
        -------
        B_fgs: array[float] of dimensions [n_frequencies, n_components - 1, n_pix]
            Foreground part of the mixing matrix (including special frequencies)
        """
        ncomp_fgs = self.n_components - 1

        if jax_use:
            # Get all templates
            templates = self.templates

            B_fgs = jnp.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
            # insert all the ones given by the pos_special_freqs
            B_fgs = B_fgs.at[jnp.array(self.pos_special_freqs), ...].set(
                jnp.broadcast_to(jnp.eye(ncomp_fgs), (self.n_pix, ncomp_fgs, ncomp_fgs)).T
            )
            # insert all the parameters values
            B_fgs = B_fgs.at[self.indexes_frequency_array_no_special, ...].set(params.at[templates].get())

            return B_fgs

        params_long = self.get_params_long_python(params)
        if ncomp_fgs != 0:
            assert params_long.shape == ((self.n_frequencies - len(self.pos_special_freqs)), ncomp_fgs, self.n_pix)
            assert len(self.pos_special_freqs) <= ncomp_fgs

        B_fgs = np.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
        if len(self.pos_special_freqs) != 0:
            # insert all the ones given by the pos_special_freqs
            for c in range(len(self.pos_special_freqs)):
                B_fgs[self.pos_special_freqs[c]][c] = 1
        # insert all the parameters values
        f = 0
        for i in range(self.n_frequencies):
            if i not in self.pos_special_freqs:
                B_fgs[i, :] = params_long[f, :, :]
                f += 1

        return B_fgs

    def get_B_from_params(self, params, jax_use=True):
        """
        Full mixing matrix, (n_frequencies*n_components), obtained from the parameters.
        CMB is given as the first component.

        Parameters
        ----------
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [Bf1_comp1_patch1, Bf1_comp1_patch2, ..., Bf2_comp1_patch1, ..., Bf1_comp2_patch1, ..., Bfn_comp2_patchn, ...]
        jax_use: bool (optional)
            If True, params are expected as JAX Array and B_mat will be returned as JAX Array, default False

        Returns
        -------
        B_mat: array[float] of dimensions [n_frequencies, n_components, n_pix]
            Full mixing matrix
        """
        if jax_use:
            if self.n_components != 1:
                return jnp.concatenate(
                    (self.get_B_cmb(jax_use=jax_use), self.get_B_fgs_from_params(params, jax_use=jax_use)), axis=1
                )
            else:
                return self.get_B_cmb(jax_use=jax_use)

        B_mat = np.concatenate((self.get_B_cmb(), self.get_B_fgs_from_params(params)), axis=1)
        return B_mat

    def get_template_B_fgs_from_params(
        self, freq, component, params, jax_use=True
    ):  ## TODO: take as input freq, component instead of nside patch
        """
        Foreground (fgs) part of the mixing matrix and one patch distribution template
        obtained from nside_patch expected lower than nside of the input maps.

        Parameters
        ----------
        nside_patch: int
            Healpix nside of one patch distribution, expected lower than nside of the input maps
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [Bf1_comp1_patch1, Bf1_comp1_patch2, ..., Bf2_comp1_patch1, ..., Bf1_comp2_patch1, ..., Bfn_comp2_patchn, ...]
        jax_use: bool (optional)
            If True, params are expected as JAX Array and the results will be returned as JAX Array, default False

        Returns
        -------
        B_fgs: array[float] of dimensions [n_frequencies, n_components - 1, n_pix]
            Foreground part of the mixing matrix (including special frequencies)
        template: array[int] of dimensions [12*nside_patch**2]
            One template indexes map whose values correspond to the indices of params
            for one patch distribution according to Healpix pixelization
        """
        ncomp_fgs = self.n_components - 1

        if jax_use:
            B_fgs = jnp.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
            # insert all the ones given by the pos_special_freqs
            B_fgs = B_fgs.at[jnp.array(self.pos_special_freqs), ...].set(
                jnp.broadcast_to(jnp.eye(ncomp_fgs), (self.n_pix, ncomp_fgs, ncomp_fgs)).T
            )
            # insert all the parameters values
            B_fgs = B_fgs.at[self.indexes_frequency_array_no_special, ...].set(params[self.templates])

            # Retrieving freq and comp indices corresponding to idx_template
            # freq_idx_template, comp_idx_template = jnp.argwhere(self.indexes_b==idx_template)

            return B_fgs, self.templates[freq, component]

    def get_patch_B_from_params(self, freq, component, params, jax_use=True):  ## TODO: fix doc
        """
        Full mixing matrix, (n_frequencies*n_components) from params and one patch distribution template.
        cmb is given as the first component.

        Parameters
        ----------
        nside_patch: int
            Healpix nside of one patch distribution, expected lower than nside of the input maps
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [Bf1_comp1_patch1, Bf1_comp1_patch2, ..., Bf2_comp1_patch1, ..., Bf1_comp2_patch1, ..., Bfn_comp2_patchn, ...]
        jax_use: bool (optional)
            If True, params are expected as JAX Array and the results will be returned as JAX Array, default False

        Returns
        -------
        B_mat: array[float] of dimensions [n_frequencies, n_components, n_pix]
            Full mixing matrix
        template: array[int] of dimensions [12*nside_patch**2]
            One template indexes map whose values correspond to the indices of params
            for one patch distribution according to Healpix pixelization
        """
        if jax_use:
            B_fgs, template = self.get_template_B_fgs_from_params(freq, component, params, jax_use=jax_use)
            return jnp.concatenate((self.get_B_cmb(jax_use=jax_use), B_fgs), axis=1), template

        B_fgs, template = self.get_template_B_fgs_from_params(freq, component, params, jax_use=jax_use)
        B_mat = np.concatenate((self.get_B_cmb(), B_fgs), axis=1)
        return B_mat, template

    def get_params_db(self, jax_use=True):
        # TODO: adjust with spv
        """
        STATUS: Not used currently, to be adjusted with spv

        Derivatives of the part of the Mixing Matrix w params
        (wrt to each entry of first comp and then each entry of second comp)
        Note: w/o pixel dimension
        """
        nrows = self.n_frequencies - self.n_components + 1
        ncols = self.n_components - 1
        if jax_use:

            def set_1(i):
                params_dBi = jnp.zeros((nrows, ncols))
                index_i = i // 2
                index_j = i % 2
                return params_dBi.at[index_i, index_j].set(1).ravel(order='C').reshape((nrows, ncols), order='F')

            return jax.vmap(set_1)(jnp.arange(nrows * ncols))

        params_dBi = np.zeros((nrows, ncols))
        params_dB = []
        for j in range(ncols):
            for i in range(nrows):
                params_dBi_copy = copy.deepcopy(params_dBi)
                params_dBi_copy[i, j] = 1
                params_dB.append(params_dBi_copy)

        return params_dB

    def get_B_db(self, jax_use=True):
        """
        STATUS: Not used currently, to be adjusted with spv

        List of derivatives of the Mixing Matrix
        (wrt to each entry of first comp and then each entry of second comp)
        Note: w/o pixel dimension
        """
        params_db = self.get_params_db(jax_use=jax_use)
        if jax_use:
            B_db = jnp.zeros((self.n_frequencies, self.n_frequencies, self.n_components))
            relevant_indexes = jnp.arange(self.pos_special_freqs[0] + 1, self.pos_special_freqs[-1])
            B_db = B_db.at[:, relevant_indexes, 1:].set(params_db)
            return B_db

        B_db = []
        for B_db_i in params_db:
            # add the zeros of special positions
            for i in self.pos_special_freqs:
                B_db_i = np.insert(B_db_i, i, np.zeros(self.n_components - 1), axis=0)
            # add the zeros of CMB
            B_db_i = np.column_stack((np.zeros(self.n_frequencies), B_db_i))
            B_db.append(B_db_i)
        return B_db
