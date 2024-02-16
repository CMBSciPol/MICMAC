import numpy as np
import copy
import jax
import jax.numpy as jnp
import chex as chx
from functools import partial

from .templates_spv import get_n_patches_b, get_nodes_b, create_one_template

# Note: 
# the mixing matrix is supposed to be the same for Q and U Stokes params
# (also we suppose that I is not used)
# Mixing matrix dimensions: number_frequencies*number_components*number_pixels



class MixingMatrix():
    def __init__(self, frequency_array, number_components, spv_nodes_b, nside_out, params=None, pos_special_freqs=[0,-1]):
        """
        Note: units are K_CMB.
        """
        self.frequency_array = frequency_array
        self.number_frequencies = jnp.size(frequency_array)    # all input freq bands
        self.number_components = number_components         # all comps (also cmb)
        self.spv_nodes_b = spv_nodes_b   # nodes for b containing info patches to build spv_templates
        self.nside_out = nside_out
        self.n_pixels = 12*self.nside_out**2
        if params is None:
            params = np.zeros((self.get_len_params()))
        else:
            assert len(params) == self.get_len_params()
            self.params = params

        ### checks on pos_special_freqs
        # check no duplicates
        assert len(pos_special_freqs) == len(set(pos_special_freqs))
        # make pos_special_freqs only positive
        for i, val_i in enumerate(pos_special_freqs):
            if val_i < 0:
                pos_special_freqs[i] = self.number_frequencies + pos_special_freqs[i]
        self.pos_special_freqs = pos_special_freqs


    def get_len_params(self):
        len_params = 0
        for node in self.spv_nodes_b:
            len_params += get_n_patches_b(node)
        return len_params


    def update_params(self, new_params, jax_use=False):
        """
        Update values of the params in the mixing matrix.
        """
        if jax_use:
            chx.assert_shape(new_params,self.get_len_params())
            self.params = new_params
            return
        assert np.shape(new_params)[0] == self.get_len_params()
        self.params = new_params

        return
    

    def get_params_long(self):
        """From the params to all the entries of the mixing matrix"""
        n_unknown_freqs = self.number_frequencies-self.number_components+1
        n_comp_fgs = self.number_components-1
        
        params_long = np.zeros((n_unknown_freqs, n_comp_fgs, self.n_pixels))
        ind_params = 0
        for ind_node_b, node_b in enumerate(self.spv_nodes_b):
            print("node: ", node_b.parent.name, node_b.name)
            # template of all the patches for this b
            spv_template_b = np.array(create_one_template(node_b, nside_out=self.nside_out, all_nsides=[], spv_templates=[]))
            # loop over the patches of this b
            params_long_b = np.zeros((self.n_pixels))
            for b in range(get_n_patches_b(node_b)):
                params_long_b += np.where(spv_template_b == b, 1, 0)*self.params[ind_params]
                ind_params += 1
            if ind_node_b < n_unknown_freqs:
                ind_freq = ind_node_b
                ind_comp = 0
            else:
                ind_freq = ind_node_b - n_unknown_freqs
                ind_comp = 1
            params_long[ind_freq, ind_comp, :] = params_long_b
        
        return params_long
    

    def get_B_fgs(self, jax_use=False):
        """
        fgs part of the mixing matrix.
        """
        ncomp_fgs = self.number_components - 1
        params_long = self.get_params_long()

        if jax_use:
            # TODO: jax part adjust with spv
            B_fgs = jnp.zeros((self.number_frequencies, ncomp_fgs, self.n_pixels))
            # insert all the ones given by the pos_special_freqs
            for c in range(ncomp_fgs):
                # B_fgs[self.pos_special_freqs[c]][c] = 1
                B_fgs = B_fgs.at[self.pos_special_freqs[c],c].set(1)
            # insert all the parameters values
            f = 0 
            for i in range(self.number_frequencies):
                if i not in self.pos_special_freqs:
                    # B_fgs[i, :] = self.params[f, :]
                    B_fgs = B_fgs.at[i, :].set(self.params[f, :])
                    f += 1
            return B_fgs

        if ncomp_fgs != 0:
            assert params_long.shape == ((self.number_frequencies - len(self.pos_special_freqs)), ncomp_fgs, self.n_pixels)
            assert len(self.pos_special_freqs) <= ncomp_fgs

        B_fgs = np.zeros((self.number_frequencies, ncomp_fgs, self.n_pixels))
        if len(self.pos_special_freqs) != 0:
            # insert all the ones given by the pos_special_freqs
            for c in range(len(self.pos_special_freqs)):
                B_fgs[self.pos_special_freqs[c]][c] = 1
        # insert all the parameters values
        f = 0
        for i in range(self.number_frequencies):
            if i not in self.pos_special_freqs:
                B_fgs[i, :] = params_long[f, :, :]
                f += 1
        
        return B_fgs


    def get_B_cmb(self, jax_use=False):
        """
        CMB column of the mixing matrix.
        """
        if jax_use:
            B_cmb = jnp.ones((self.number_frequencies, self.n_pixels))
            return B_cmb[:, np.newaxis, :]

        B_cmb = np.ones((self.number_frequencies, self.n_pixels))
        B_cmb = B_cmb[:, np.newaxis, :]
        
        return B_cmb


    def get_B(self, jax_use=False):
        """
        Full mixing matrix, (number_frequencies*number_components).
        cmb is given as the first component.
        """
        if jax_use:
            return jnp.concatenate((self.get_B_cmb(jax_use=jax_use), self.get_B_fgs(jax_use=jax_use)), axis=1)
        B_mat = np.concatenate((self.get_B_cmb(), self.get_B_fgs()), axis=1)
        
        return B_mat
    

    def get_params_db(self, jax_use=False):
        """
        Derivatives of the part of the Mixing Matrix w params
        (wrt to each entry of first comp and then each entry of second comp)
        Note: w/o pixel dimension
        """
        nrows = self.number_frequencies-self.number_components+1
        ncols = self.number_components-1
        if jax_use:
            def set_1(i):
                params_dBi = jnp.zeros((nrows,ncols))
                index_i = i//2
                index_j = i%2
                return params_dBi.at[index_i,index_j].set(1).ravel(order='C').reshape((nrows,ncols),order='F')
            return jax.vmap(set_1)(jnp.arange(nrows*ncols))

        params_dBi = np.zeros((nrows, ncols))
        params_dB = []
        for j in range(ncols):
            for i in range(nrows):
                params_dBi_copy = copy.deepcopy(params_dBi)
                params_dBi_copy[i,j] = 1
                params_dB.append(params_dBi_copy)
            
        return params_dB


    def get_B_db(self, jax_use=False):
        """
        List of derivatives of the Mixing Matrix
        (wrt to each entry of first comp and then each entry of second comp)
        Note: w/o pixel dimension
        """
        params_db = self.get_params_db(jax_use=jax_use)
        if jax_use:
            B_db = jnp.zeros((params_db.shape[0],self.number_frequencies,self.number_components))
            relevant_indexes = jnp.arange(self.pos_special_freqs[0]+1,self.pos_special_freqs[-1])
            B_db = B_db.at[:,relevant_indexes,1:].set(params_db)
            return B_db

        B_db = []
        for B_db_i in params_db:
            # add the zeros of special positions
            for i in self.pos_special_freqs:
                B_db_i = np.insert(B_db_i, i, np.zeros(self.number_components-1), axis=0)
            # add the zeros of CMB
            B_db_i = np.column_stack((np.zeros(self.number_frequencies), B_db_i))
            B_db.append(B_db_i)
        
        return B_db
