import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import copy
import jax
import jax.numpy as jnp
import chex as chx
from functools import partial
from jax.tree_util import tree_flatten, tree_unflatten

from .templates_spv import get_n_patches_b, get_nodes_b, create_one_template, get_values_b, create_one_template_from_bdefaultvalue

# Note: 
# the mixing matrix is supposed to be the same for Q and U Stokes params
# (also we suppose that I is not used)
# Mixing matrix dimensions: n_frequencies*n_components*number_pixels



def get_indexes_b(n_frequencies, n_components, spv_nodes_b):
    """ Return indexes of params for all frequencies and components
    """
    indexes = np.zeros((n_frequencies, n_components))
    for freq in range(n_frequencies):
        for comp in range(n_components):
            indexes[freq, comp] = get_n_patches_b(spv_nodes_b[freq + comp*n_frequencies - 1])
    indexes[0,0] = 0
    return indexes.ravel(order='F').cumsum().reshape((n_frequencies, n_components),order='F')

def get_indexes_patches(indexes_b, len_params, jax_use=False):
    """ Return indexes of params for all patches of a b
    """
    indexes_patches_list = []
    for i in range(indexes_b.shape[0]-1):
        indexes_patches_list.append(jnp.arange(indexes_b[i],indexes_b[i+1]))
    indexes_patches_list.append(jnp.arange(indexes_b[-1],len_params))
    return indexes_patches_list

def get_len_params(spv_nodes_b):
    len_params = 0
    for node in spv_nodes_b:
        len_params += get_n_patches_b(node)
    return len_params


class MixingMatrix():
    def __init__(self, frequency_array, n_components, spv_nodes_b, nside, params=None, pos_special_freqs=[0,-1]):
        """
        Note: units are K_CMB.
        """
        self.frequency_array = frequency_array
        self.n_frequencies = np.size(frequency_array)    # all input freq bands
        self.n_components = n_components         # all comps (also cmb)
        self.spv_nodes_b = spv_nodes_b   # nodes for b containing info patches to build spv_templates
        self.nside = nside
        self.len_params = get_len_params(self.spv_nodes_b)
        if params is None:
            params = np.zeros((self.len_params))
        else:
            try:
                assert np.shape(params)[0] == self.len_params
            except:
                raise Exception("params must be of dimensions", self.len_params, flush=True)
            
        self.params = params

        ### checks on pos_special_freqs
        # check no duplicates
        assert len(pos_special_freqs) == len(set(pos_special_freqs))
        # make pos_special_freqs only positive
        for i, val_i in enumerate(pos_special_freqs):
            if val_i < 0:
                pos_special_freqs[i] = self.n_frequencies + pos_special_freqs[i]
        self.pos_special_freqs = pos_special_freqs

        self.values_b = get_values_b(self.spv_nodes_b, self.n_frequencies-len(self.pos_special_freqs), self.n_components-1)        
        self.indexes_b = get_indexes_b(self.n_frequencies-len(self.pos_special_freqs), self.n_components-1, self.spv_nodes_b)
        self.size_patches = jnp.array([get_n_patches_b(node) for node in self.spv_nodes_b])
        self.max_len_patches_Bf = int(self.size_patches.max())
        
        
    @property
    def n_pix(self):
        """ Number of pixels of one input freq map
        """
        return 12*self.nside**2


    def update_params(self, new_params, jax_use=False):
        """
        Update values of the params in the mixing matrix.
        """
        if jax_use:
            chx.assert_shape(new_params,(self.len_params,))
            self.params = new_params
            return
        assert np.shape(new_params)[0] == self.len_params
        self.params = new_params

        return
    

    def get_params_long_python(self, params, print_bool=False):
        # only python version
        """From the params to all the entries of the mixing matrix"""
        n_unknown_freqs = self.n_frequencies-self.n_components+1
        n_comp_fgs = self.n_components-1
        params_long = np.zeros((n_unknown_freqs, n_comp_fgs, self.n_pix))
        ind_params = 0
        for ind_node_b, node_b in enumerate(self.spv_nodes_b):
            if print_bool:
                print("node: ", node_b.parent.name, node_b.name)
            # template of all the patches for this b
            spv_template_b = np.array(create_one_template(node_b, nside=self.nside, all_nsides=[], spv_templates=[]))
            # hp.mollview(spv_template_b)
            # plt.show()
            # loop over the patches of this b
            params_long_b = np.zeros((self.n_pix))
            for b in range(get_n_patches_b(node_b)):
                params_long_b += np.where(spv_template_b == b, 1, 0)*params[ind_params]
                ind_params += 1
            # hp.mollview(params_long_b)
            # plt.show()
            ind_freq = np.where(ind_node_b<n_unknown_freqs, ind_node_b, ind_node_b-n_unknown_freqs)
            ind_comp = np.where(ind_node_b<n_unknown_freqs, 0, 1)

            params_long[ind_freq, ind_comp, :] = params_long_b

        return params_long
    

    def get_params_long(self, jax_use=False):
        """From the params to all the entries of the mixing matrix"""

        if jax_use:
            # # TODO: to finish in jax instead of purecallback
            # # spv_jnodes_b = tree_flatten(self.spv_nodes_b)
            # def get_paramslong_patch(carry, indx_freq):
            #     indx_comp = indx_freq%n_unknown_freqs
            #     value_b = jnp.array(self.values_b)[indx_freq + indx_comp*self.n_frequencies - 1]   # TODO: generalize for adaptive comp sep
            #     # value_b = value_b_copy[indx_freq + indx_comp*self.n_frequencies - 1]   # TODO: generalize for adaptive comp sep
            #     spv_template_b = jnp.array(create_one_template_from_bdefaultvalue(value_b, nside=self.nside, all_nsides=[], spv_templates=[], use_jax=True))

            #     def fill_patch(b):
            #         return jnp.where(spv_template_b == b+1, 1, 0)*self.params[self.indexes_b[indx_freq,indx_comp]]

            #     params_long_b = jax.vmap(fill_patch)(jnp.arange(get_n_patches_b(value_b, jax_use=True))).sum(axis=0)
            #     return carry, params_long_b
            
            # # params_long = jax.vmap(get_paramslong_patch, out_axes=(0,None,None))(jnp.arange(n_unknown_freqs), jnp.arange(n_comp_fgs))
            # # params_long = jax.vmap(get_paramslong_patch)(jnp.arange(n_unknown_freqs*n_comp_fgs))
            # _, params_long = jax.lax.scan(get_paramslong_patch, None, jnp.arange(n_unknown_freqs*n_comp_fgs))
            # return params_long.reshape((n_unknown_freqs,n_comp_fgs))
            def pure_call_ud_get_params_long_python(params):
                shape_output = (self.n_frequencies-self.n_components+1,self.n_components-1,12*self.nside**2,)
                return jax.pure_callback(self.get_params_long_python, jax.ShapeDtypeStruct(shape_output, np.float64),params,)
            
            return pure_call_ud_get_params_long_python(self.params)

        return self.get_params_long_python(self.params)
    

    def get_B_fgs(self, jax_use=False):
        """
        fgs part of the mixing matrix.
        """
        ncomp_fgs = self.n_components - 1
        params_long = self.get_params_long(jax_use=jax_use)

        if jax_use:
            B_fgs = jnp.zeros((self.n_frequencies, ncomp_fgs, self.n_pix))
            # insert all the ones given by the pos_special_freqs
            for c in range(ncomp_fgs):
                # B_fgs[self.pos_special_freqs[c]][c] = 1
                B_fgs = B_fgs.at[self.pos_special_freqs[c],c].set(1)
            # insert all the parameters values
            f = 0 
            for i in range(self.n_frequencies):
                if i not in self.pos_special_freqs:
                    # B_fgs[i, :] = self.params[f, :]
                    B_fgs = B_fgs.at[i, :].set(params_long[f, :, :])
                    f += 1
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
        cmb is given as the first component.
        """
        if jax_use:
            return jnp.concatenate((self.get_B_cmb(jax_use=jax_use), self.get_B_fgs(jax_use=jax_use)), axis=1)
        B_mat = np.concatenate((self.get_B_cmb(), self.get_B_fgs()), axis=1)
        
        return B_mat
    

    # def get_params_db(self, jax_use=False):
    #     # TODO: adjust with spv
    #     """
    #     Derivatives of the part of the Mixing Matrix w params
    #     (wrt to each entry of first comp and then each entry of second comp)
    #     Note: w/o pixel dimension
    #     """
    #     nrows = self.n_frequencies-self.n_components+1
    #     ncols = self.n_components-1
    #     if jax_use:
    #         def set_1(i):
    #             params_dBi = jnp.zeros((nrows,ncols))
    #             index_i = i//2
    #             index_j = i%2
    #             return params_dBi.at[index_i,index_j].set(1).ravel(order='C').reshape((nrows,ncols),order='F')
    #         return jax.vmap(set_1)(jnp.arange(nrows*ncols))

    #     params_dBi = np.zeros((nrows, ncols))
    #     params_dB = []
    #     for j in range(ncols):
    #         for i in range(nrows):
    #             params_dBi_copy = copy.deepcopy(params_dBi)
    #             params_dBi_copy[i,j] = 1
    #             params_dB.append(params_dBi_copy)
            
    #     return params_dB


    # def get_B_db(self, jax_use=False):
    #     """
    #     List of derivatives of the Mixing Matrix
    #     (wrt to each entry of first comp and then each entry of second comp)
    #     Note: w/o pixel dimension
    #     """
    #     params_db = self.get_params_db(jax_use=jax_use)
    #     if jax_use:
    #         B_db = jnp.zeros((params_db.shape[0],self.n_frequencies,self.n_components))
    #         relevant_indexes = jnp.arange(self.pos_special_freqs[0]+1,self.pos_special_freqs[-1])
    #         B_db = B_db.at[:,relevant_indexes,1:].set(params_db)
    #         return B_db

    #     B_db = []
    #     for B_db_i in params_db:
    #         # add the zeros of special positions
    #         for i in self.pos_special_freqs:
    #             B_db_i = np.insert(B_db_i, i, np.zeros(self.n_components-1), axis=0)
    #         # add the zeros of CMB
    #         B_db_i = np.column_stack((np.zeros(self.n_frequencies), B_db_i))
    #         B_db.append(B_db_i)
        
    #     return B_db
