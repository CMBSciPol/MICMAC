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

import chex as chx
import jax
import jax.numpy as jnp
import numpy as np

from micmac.foregrounds.templates import (
    create_one_template,
    create_one_template_from_bdefaultvalue,
    get_n_patches_b,
    get_values_b,
)

__all__ = ['get_indexes_b', 'get_indexes_patches', 'get_len_params', 'MixingMatrix']

# Note:
# the mixing matrix is supposed to be the same for Q and U Stokes params
# (also we suppose that I is not used)
# Mixing matrix dimensions: n_frequencies*n_components*number_pixels


def get_indexes_b(n_frequencies, n_components, spv_nodes_b):
    """
    Return indexes of params for all frequencies and components

    Parameters
    ----------
    n_frequencies: int
        Number of frequencies
    n_components: int
        Number of components
    spv_nodes_b: list
        List of nodes for b containing info patches to build spv_templates

    Returns
    -------
    indexes: array
        Indexes of params for all frequencies and components
    """
    indexes = np.zeros((n_frequencies, n_components), dtype=int)
    for freq in range(n_frequencies):
        for comp in range(n_components):
            indexes[freq, comp] = get_n_patches_b(spv_nodes_b[freq + comp * n_frequencies - 1])
    indexes[0, 0] = 0
    return indexes.ravel(order='F').cumsum().reshape((n_frequencies, n_components), order='F')


def get_indexes_patches(indexes_b, len_params):
    """
    Return indexes of params for all patches of a b

    Parameters
    ----------
    indexes_b: array
        Indexes of params for all frequencies and components
    len_params: int
        Total number of free parameters (per frequency, component, patch)

    Returns
    -------
    indexes_patches_list: list
        List of indexes of params for all patches of a b
    """
    indexes_patches_list = []
    for i in range(indexes_b.shape[0] - 1):
        indexes_patches_list.append(jnp.arange(indexes_b[i], indexes_b[i + 1]))
    indexes_patches_list.append(jnp.arange(indexes_b[-1], len_params))
    return indexes_patches_list


def get_len_params(spv_nodes_b):
    """
    Return total number of free parameters (per frequency, component, patch)

    Parameters
    ----------
    spv_nodes_b: list
        List of nodes for b containing info patches to build spv_templates

    Returns
    -------
    len_params: int
        Total number of free parameters (per frequency, component, patch)
    """
    len_params = 0
    for node in spv_nodes_b:
        len_params += get_n_patches_b(node)
    return len_params


class MixingMatrix:
    def __init__(self, frequency_array, n_components, spv_nodes_b, nside, params=None, pos_special_freqs=[0, -1]):
        """
        Note: units are K_CMB.

        Parameters
        ----------
        frequency_array: array
            Array of frequencies
        n_components: int
            Number of components
        spv_nodes_b: list
            List of nodes for b containing info patches to build spv_templates
        nside: int
            Healpix nside of the expected input maps
        params: array (optional)
            Initial values of the parameters of the mixing matrix, default None (then initialized with zeros)
        pos_special_freqs: list (optional)
            List of indexes of special frequencies (e.g. 0 for synchrotron, -1 for dust)
        """
        self.frequency_array = np.array(frequency_array, dtype=int)  # all input freq bands
        self.n_frequencies = np.size(frequency_array)  # all input freq bands
        self.n_components = n_components  # all comps (also cmb)
        self.spv_nodes_b = spv_nodes_b  # nodes for b containing info patches to build spv_templates
        self.nside = nside  # nside of the expected input maps
        self.len_params = get_len_params(
            self.spv_nodes_b
        )  # total number of free parameters (per frequency, component, patch)
        if params is None:
            params = np.zeros(self.len_params)
        else:
            try:
                assert np.shape(params)[0] == self.len_params
            except:
                raise Exception('params must be of dimensions', self.len_params, flush=True)

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
            # Values of the patch nsides corresponding to each node
            self.values_b = jnp.array(
                get_values_b(self.spv_nodes_b, self.n_frequencies - len(self.pos_special_freqs), self.n_components - 1)
            )
            # Values of the first index of each B_f parameter in params
            self.indexes_b = jnp.array(
                get_indexes_b(self.n_frequencies - len(self.pos_special_freqs), self.n_components - 1, self.spv_nodes_b)
            )
            self.size_patches = jnp.array([get_n_patches_b(node) for node in self.spv_nodes_b])
            self.sum_size_patches_indexed_freq_comp = (
                self.size_patches.cumsum().reshape(
                    (self.n_frequencies - len(self.pos_special_freqs), self.n_components - 1), order='F'
                )
                - self.size_patches[0]
            )
            self.max_len_patches_Bf = int(self.size_patches.max())
        else:
            self.values_b = None  # Values of the patch nsides corresponding to each node
            self.indexes_b = jnp.array([[0]])  # Values of the first index of each B_f parameter in params
            self.size_patches = None  # Number of patches for each node
            self.sum_size_patches_indexed_freq_comp = None  # Cumulative sum of the number of patches for each node
            self.max_len_patches_Bf = None  # Maximum number of patches for each node

    @property
    def n_pix(self):
        """
        Number of pixels of one input freq map
        """
        return 12 * self.nside**2

    def update_params(self, new_params, jax_use=False):
        """
        Update values of the params in the mixing matrix.

        Parameters
        ----------
        new_params: array
            New values of the parameters of the mixing matrix

        jax_use: bool (optional)
            If True, use JAX to update it as JAX Array, default False
        """
        if jax_use:
            chx.assert_shape(new_params, (self.len_params,))
            self.params = jnp.array(new_params)
            return
        assert np.shape(new_params)[0] == self.len_params
        self.params = new_params

        return

    def get_params_long_python(self, params, print_bool=False):
        # only python version
        """
        From the params (B_f) to all the entries of the mixing matrix
        only with Python (numpy) without JAX

        Parameters
        ----------
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]

        print_bool: bool (optional)
            If True, print the node names

        Returns
        -------
        params_long: array[float] of dimensions [n_frequencies - n_components + 1, n_components - 1, n_pix]
            Reshaped free parameters of the mixing matrix
        """
        n_unknown_freqs = self.n_frequencies - self.n_components + 1
        n_comp_fgs = self.n_components - 1
        params_long = np.zeros((n_unknown_freqs, n_comp_fgs, self.n_pix))
        ind_params = 0
        for ind_node_b, node_b in enumerate(self.spv_nodes_b):
            if print_bool:
                print('node: ', node_b.parent.name, node_b.name)
            # template of all the patches for this b
            spv_template_b = np.array(
                create_one_template(node_b, nside=self.nside, all_nsides=None, spv_templates=None)
            )
            # hp.mollview(spv_template_b)
            # plt.show()
            # loop over the patches of this b
            params_long_b = np.zeros(self.n_pix)
            for b in range(get_n_patches_b(node_b)):
                params_long_b += np.where(spv_template_b == b, 1, 0) * params[ind_params]
                ind_params += 1
            # hp.mollview(params_long_b)
            # plt.show()
            ind_freq = np.where(ind_node_b < n_unknown_freqs, ind_node_b, ind_node_b - n_unknown_freqs)
            ind_comp = np.where(ind_node_b < n_unknown_freqs, 0, 1)

            params_long[ind_freq, ind_comp, :] = params_long_b

        return params_long

    # def pure_call_ud_get_params_long_python(self, params):
    #     """
    #         JAX Pure call to get_params_long_python

    #         Parameters
    #         ----------
    #         params: compressed version of the parameters of the mixing matrix

    #         Returns
    #         -------
    #         Full parameters of the mixing matrix
    #     """
    #     shape_output = (self.n_frequencies-self.n_components+1,self.n_components-1,12*self.nside**2,)
    #     return jax.pure_callback(self.get_params_long_python, jax.ShapeDtypeStruct(shape_output, np.float64),params,)

    def get_idx_template_params_long_python(self, idx_template, params, print_bool=False):
        # only python version
        """
        From the params to all the entries of the mixing matrix
        For a given template index, retrieve the corresponding template

        Parameters
        ----------
        idx_template: array[int]
            index of params of the corresponding template which will be saved
        params: array[float]
            flatttened compressed array of the free params of the mixing matrix
            expected to be stored as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]

        Returns
        -------
        idx_template: array[int]
            Full parameters of the mixing matrix re-flattened as [(n_frequencies - n_components + 1)*(n_components - 1), n_pix]
            stacked with one patch distribution template indicated by idx_template
        """
        n_unknown_freqs = self.n_frequencies - self.n_components + 1
        n_comp_fgs = self.n_components - 1
        params_long = np.zeros((n_unknown_freqs, n_comp_fgs, self.n_pix))
        all_templates = []
        ind_params = 0
        for ind_node_b, node_b in enumerate(self.spv_nodes_b):
            if print_bool:
                print('node: ', node_b.parent.name, node_b.name)
            # template of all the patches for this b
            spv_template_b = np.array(
                create_one_template(node_b, nside=self.nside, all_nsides=None, spv_templates=None)
            )

            # loop over the patches of this b
            params_long_b = np.zeros(self.n_pix)

            patch_arange = np.arange(get_n_patches_b(node_b))
            arange_ind_params = ind_params + patch_arange
            if np.isin(arange_ind_params, idx_template).any():
                all_templates.append(spv_template_b)
            for b in patch_arange:
                params_long_b += np.where(spv_template_b == b, 1, 0) * params[ind_params]
                ind_params += 1
            # hp.mollview(params_long_b)
            # plt.show()
            ind_freq = np.where(ind_node_b < n_unknown_freqs, ind_node_b, ind_node_b - n_unknown_freqs)
            ind_comp = np.where(ind_node_b < n_unknown_freqs, 0, 1)

            params_long[ind_freq, ind_comp, :] = params_long_b

        all_templates = np.array(all_templates)
        return np.vstack([params_long.reshape((n_unknown_freqs * n_comp_fgs, self.n_pix)), all_templates.squeeze()])

    # def pure_call_ud_get_idx_template_params_long_python(self, idx_template, params):
    #     """
    #         JAX Pure call to get_params_long_python

    #         Parameters
    #         ----------
    #         idx_template
    #         params: compressed version of the parameters of the mixing matrix

    #         Returns
    #         -------
    #         Full parameters of the mixing matrix
    #     """
    #     n_unknown_freqs = self.n_frequencies-self.n_components+1
    #     n_comp_fgs = self.n_components-1
    #     shape_output = (((n_unknown_freqs*n_comp_fgs+1),self.n_pix))
    #     output_pure_call_back = jax.pure_callback(self.get_idx_template_params_long_python, jax.ShapeDtypeStruct(shape_output, np.float64),idx_template,params,)
    #     return output_pure_call_back[:-1].reshape((n_unknown_freqs,n_comp_fgs,self.n_pix,)), output_pure_call_back[-1]

    def get_all_templates(self):
        """
        Retrieve all templates maps whose values correspond to the indices of params,
        and indexed per frequency and component

        Returns
        -------
        all_templates: array[int] of dimensions [n_frequencies - n_components + 1, n_components - 1, 12*nside**2]
            All templates indexes maps whose values correspond to the indices of params
            for all the patches distributions per frequency and component
        """
        n_unknown_freqs = self.n_frequencies - self.n_components + 1
        n_comp_fgs = self.n_components - 1

        ## Creating all the templates
        def create_all_templates_indexed_freq(idx_freq):
            def create_all_templates_indexed_comp(idx_comp):
                template_idx_comp = create_one_template_from_bdefaultvalue(
                    jnp.expand_dims(self.values_b[idx_freq, idx_comp], axis=0),
                    self.nside,
                    all_nsides=None,
                    spv_templates=None,
                    use_jax=True,
                    print_bool=False,
                )
                return template_idx_comp + self.sum_size_patches_indexed_freq_comp[idx_freq, idx_comp]

            template_idx_freq_comp = jax.vmap(create_all_templates_indexed_comp)(jnp.arange(n_comp_fgs))
            return template_idx_freq_comp

        ## Maping over the functions to create the templates
        return jax.vmap(create_all_templates_indexed_freq)(jnp.arange(n_unknown_freqs))

    def get_one_template(self, nside_patch):
        """
        Retrieve all templates maps whose values correspond to the indices of params,
        and indexed per frequency and component

        Parameters
        ----------
        nside_patch: int
            Healpix nside of one patch distribution

        Returns
        -------
        template: array[int] of dimensions [12*nside_patch**2]
            One template indexes map whose values correspond to the indices of params
            for one patch distribution according to Healpix pixelization
        """
        return create_one_template_from_bdefaultvalue(
            jnp.expand_dims(nside_patch, axis=0),
            self.nside,
            all_nsides=None,
            spv_templates=None,
            use_jax=True,
            print_bool=False,
        )

    def get_params_long(self, jax_use=False):
        """
        From the params to all the entries of the mixing matrix

        Parameters
        ----------
        jax_use: bool (optional)
            If True, params are expected as JAX Array, default False

        Returns
        -------
        params_long: array[float] of dimensions [n_frequencies - n_components + 1, n_components - 1, n_pix]
            Reshaped free parameters of the mixing matrix
        """

        if jax_use:
            templates_to_fill = self.get_all_templates()

            ## Filling the templates with parameters values
            return self.params.at[templates_to_fill].get()

        return self.get_params_long_python(self.params)

    def get_B_fgs(self, jax_use=False):
        """
        Foreground part of the mixing matrix.

        Parameters
        ----------
        jax_use: bool (optional)
            If True, params are expected as JAX Array, default False

        Returns
        -------
        B_fgs: array[float] of dimensions [n_frequencies, n_components - 1, n_pix]
            Foreground part of the mixing matrix (including special frequencies)
        """
        ncomp_fgs = self.n_components - 1
        params_long = self.get_params_long(jax_use=jax_use)

        if jax_use:
            B_fgs = jnp.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
            # insert all the ones given by the pos_special_freqs
            B_fgs = B_fgs.at[jnp.array(self.pos_special_freqs), ...].set(
                jnp.broadcast_to(jnp.eye(ncomp_fgs), (self.n_pix, ncomp_fgs, ncomp_fgs)).T
            )
            # insert all the parameters values
            B_fgs = B_fgs.at[self.indexes_frequency_array_no_special, ...].set(params_long)
            return B_fgs

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

    def get_B_cmb(self, jax_use=False):
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

    def get_B(self, jax_use=False):
        """
        Full mixing matrix, (n_frequencies*n_components).
        CMB is given as the first component.

        Parameters
        ----------
        jax_use: bool (optional)
            If True, returned as JAX Array, default False

        Returns
        -------
        B_mat: array[float] of dimensions [n_frequencies, n_components, n_pix]
            Full mixing matrix
        """
        if jax_use:
            if self.n_components != 1:
                return jnp.concatenate((self.get_B_cmb(jax_use=jax_use), self.get_B_fgs(jax_use=jax_use)), axis=1)
            else:
                return self.get_B_cmb(jax_use=jax_use)
        if self.n_components != 1:
            B_mat = np.concatenate((self.get_B_cmb(), self.get_B_fgs()), axis=1)
        else:
            B_mat = self.get_B_cmb()
        return B_mat

    def get_B_fgs_from_params(self, params, jax_use=False):
        """
        Foreground part of the mixing matrix obtained from the parameters.

        Parameters
        ----------
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]
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
            templates = self.get_all_templates()

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

    def get_B_from_params(self, params, jax_use=False):
        """
        Full mixing matrix, (n_frequencies*n_components), obtained from the parameters.
        CMB is given as the first component.

        Parameters
        ----------
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]
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

    def get_template_B_fgs_from_params(self, nside_patch, params, jax_use=False):
        """
        Foreground (fgs) part of the mixing matrix and one patch distribution template
        obtained from nside_patch expected lower than nside of the input maps.

        Parameters
        ----------
        nside_patch: int
            Healpix nside of one patch distribution, expected lower than nside of the input maps
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]
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
            # Get all templates
            templates = self.get_all_templates()

            B_fgs = jnp.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
            # insert all the ones given by the pos_special_freqs
            B_fgs = B_fgs.at[jnp.array(self.pos_special_freqs), ...].set(
                jnp.broadcast_to(jnp.eye(ncomp_fgs), (self.n_pix, ncomp_fgs, ncomp_fgs)).T
            )
            # insert all the parameters values
            B_fgs = B_fgs.at[self.indexes_frequency_array_no_special, ...].set(params[templates])

            # Retrieving freq and comp indices corresponding to idx_template
            # freq_idx_template, comp_idx_template = jnp.argwhere(self.indexes_b==idx_template)

            return B_fgs, self.get_one_template(nside_patch)

        assert nside_patch <= self.nside

        if ncomp_fgs != 0:
            assert params_long.shape == ((self.n_frequencies - len(self.pos_special_freqs)), ncomp_fgs, self.n_pix)
            assert len(self.pos_special_freqs) <= ncomp_fgs

        params_long, template = self.get_idx_template_params_long_python(idx_template, params)
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

        return B_fgs, template

    def get_patch_B_from_params(self, nside_patch, params, jax_use=False):
        """
        Full mixing matrix, (n_frequencies*n_components) from params and one patch distribution template.
        cmb is given as the first component.

        Parameters
        ----------
        nside_patch: int
            Healpix nside of one patch distribution, expected lower than nside of the input maps
        params: array[float]
            Flattened version of all free parameters of the mixing matrix per patch
            expected to be stored as [B_f1_comp1_patch1, B_f1_comp1_patch2, ..., B_f2_comp1_patch1, ..., B_f1_comp2_patch1, ..., B_fn_comp2_patchn, ...]
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
            B_fgs, template = self.get_template_B_fgs_from_params(nside_patch, params, jax_use=jax_use)
            return jnp.concatenate((self.get_B_cmb(jax_use=jax_use), B_fgs), axis=1), template

        B_fgs, template = self.get_template_B_fgs_from_params(nside_patch, params, jax_use=jax_use)
        B_mat = np.concatenate((self.get_B_cmb(), B_fgs), axis=1)
        return B_mat, template

    def get_params_db(self, jax_use=False):
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

    def get_B_db(self, jax_use=False):
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
