import numpy as np
import copy
import jax
import jax.numpy as jnp
import chex as chx
from functools import partial

# Note: the mixing matrix is supposed to be the same 
# for Q and U Stokes params and all pixels.
# (also we supposed that I is never used)
# Thus it has dimensions (nfreq*ncomp).



class MixingMatrix():
    def __init__(self, freqs, ncomp, params, pos_special_freqs):
        """
        Note: units are K_CMB.
        """
        self.freqs = freqs
        self.nfreq = len(freqs)    # all input freq bands
        self.ncomp = ncomp         # all comps (also cmb)
        assert np.shape(params) == (self.nfreq-self.ncomp+1, self.ncomp-1)
        self.params = params

        ### checks on pos_special_freqs
        # check no duplicates
        assert len(pos_special_freqs) == len(set(pos_special_freqs))
        # make pos_special_freqs only positive
        for i, val_i in enumerate(pos_special_freqs):
            if val_i < 0:
                pos_special_freqs[i] = self.nfreq + pos_special_freqs[i]
        self.pos_special_freqs = pos_special_freqs


    def update_params(self, new_params, jax_use=False):
        """
        Update values of the params in the mixing matrix.
        """
        if jax_use:
            chx.assert_shape(new_params,(self.nfreq-self.ncomp+1, self.ncomp-1))
            self.params = new_params
            return
        assert np.shape(new_params) == (self.nfreq-self.ncomp+1, self.ncomp-1)
        self.params = new_params

        return
    

    def get_B_fgs(self, jax_use=False):
        """
        fgs part of the mixing matrix.
        """
        ncomp_fgs = self.ncomp - 1
        if jax_use:
            B_fgs = jnp.zeros((self.nfreq, ncomp_fgs))
            # insert all the ones given by the pos_special_freqs
            for c in range(ncomp_fgs):
                # B_fgs[self.pos_special_freqs[c]][c] = 1
                B_fgs = B_fgs.at[self.pos_special_freqs[c],c].set(1)
            # insert all the parameters values
            f = 0 
            for i in range(self.nfreq):
                if i not in self.pos_special_freqs:
                    # B_fgs[i, :] = self.params[f, :]
                    B_fgs = B_fgs.at[i, :].set(self.params[f, :])
                    f += 1
            return B_fgs

        if ncomp_fgs != 0:
            assert self.params.shape == ((self.nfreq - ncomp_fgs),ncomp_fgs)
            assert len(self.pos_special_freqs) == ncomp_fgs
        
        B_fgs = np.zeros((self.nfreq, ncomp_fgs))
        # insert all the ones given by the pos_special_freqs
        for c in range(ncomp_fgs):
            B_fgs[self.pos_special_freqs[c]][c] = 1
        # insert all the parameters values
        f = 0 
        for i in range(self.nfreq):
            if i not in self.pos_special_freqs:
                B_fgs[i, :] = self.params[f, :]
                f += 1
        
        return B_fgs


    def get_B_cmb(self, jax_use=False):
        """
        CMB column of the mixing matrix.
        """
        if jax_use:
            B_cmb = jnp.ones((self.nfreq))
            return B_cmb[:, np.newaxis]

        B_cmb = np.ones((self.nfreq))
        B_cmb = B_cmb[:, np.newaxis]
        
        return B_cmb


    def get_B(self, jax_use=False):
        """
        Full mixing matrix, (nfreq*ncomp).
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
        nrows = self.nfreq-self.ncomp+1
        ncols = self.ncomp-1
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
            B_db = jnp.zeros((params_db.shape[0],self.nfreq,self.ncomp))
            relevant_indexes = jnp.arange(self.pos_special_freqs[0]+1,self.pos_special_freqs[-1])
            B_db = B_db.at[:,relevant_indexes,1:].set(params_db)
            return B_db

        B_db = []
        for B_db_i in params_db:
            # add the zeros of special positions
            for i in self.pos_special_freqs:
                B_db_i = np.insert(B_db_i, i, np.zeros(self.ncomp-1), axis=0)
            # add the zeros of CMB
            B_db_i = np.column_stack((np.zeros(self.nfreq), B_db_i))
            B_db.append(B_db_i)
        
        return B_db



# # @partial(jax.jit, static_argnames=['number_components', 'number_frequencies'])
# def create_mixing_matrix_jax(params_mixing_matrix, number_components, number_frequencies, pos_special_freqs=[0,-1]):
#     # number_frequencies = params_mixing_matrix.shape[0] + 2
#     # number_components = params_mixing_matrix.shape[1] + 1

#     new_mixing_matrix = jnp.zeros((number_frequencies,number_components))
#     new_mixing_matrix = new_mixing_matrix.at[:,0].set(1)
#     # new_mixing_matrix[0,1] = 0
#     # new_mixing_matrix[-1,-1] = 0
#     # new_mixing_matrix[1:,1:-1] = jnp.array(params_mixing_matrix.reshape((param_dict['number_frequencies']-2,param_dict['number_components']-1)))
    
#     for c in range(1,number_components):
#         # print("Test :", pos_special_freqs[c-1],c)
#         new_mixing_matrix = new_mixing_matrix.at[pos_special_freqs[c-1],c].set(1)
#     # new_mixing_matrix = new_mixing_matrix.at[0,1].set(0)
#     # new_mixing_matrix = new_mixing_matrix.at[-1,-1].set(0)

#     all_indexes_bool = jnp.ones(number_frequencies, dtype=bool)
#     # all_indexes_bool[pos_special_freqs] = False

#     all_indexes_bool = all_indexes_bool.at[jnp.array(pos_special_freqs)].set(False)
    
#     # TODO: put special frequencies not hardcoded
#     new_mixing_matrix = new_mixing_matrix.at[1:-1,1:].set(jnp.array(params_mixing_matrix.reshape((number_frequencies-number_components+1,number_components-1),order='F')))
#     # new_mixing_matrix = new_mixing_matrix.at[all_indexes_bool,1:].set(jnp.array(params_mixing_matrix.reshape((number_frequencies-2,number_components-1),order='F')))
#     return new_mixing_matrix
