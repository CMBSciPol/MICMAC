import numpy as np
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

# @partial(jax.jit, static_argnames=['number_components', 'number_frequencies'])
def create_mixing_matrix_jax(params_mixing_matrix, number_components, number_frequencies, pos_special_freqs=[0,-1]):
    # number_frequencies = params_mixing_matrix.shape[0] + 2
    # number_components = params_mixing_matrix.shape[1] + 1

    new_mixing_matrix = jnp.zeros((number_frequencies,number_components))
    new_mixing_matrix = new_mixing_matrix.at[:,0].set(1)
    # new_mixing_matrix[0,1] = 0
    # new_mixing_matrix[-1,-1] = 0
    # new_mixing_matrix[1:,1:-1] = jnp.array(params_mixing_matrix.reshape((param_dict['number_frequencies']-2,param_dict['number_components']-1)))
    
    for c in range(1,number_components):
        # print("Test :", pos_special_freqs[c-1],c)
        new_mixing_matrix = new_mixing_matrix.at[pos_special_freqs[c-1],c].set(1)
    # new_mixing_matrix = new_mixing_matrix.at[0,1].set(0)
    # new_mixing_matrix = new_mixing_matrix.at[-1,-1].set(0)

    all_indexes_bool = jnp.ones(number_frequencies, dtype=bool)
    # all_indexes_bool[pos_special_freqs] = False

    all_indexes_bool = all_indexes_bool.at[jnp.array(pos_special_freqs)].set(False)
    
    # TODO: put special frequencies not hardcoded
    new_mixing_matrix = new_mixing_matrix.at[1:-1,1:].set(jnp.array(params_mixing_matrix.reshape((number_frequencies-number_components+1,number_components-1),order='F')))
    # new_mixing_matrix = new_mixing_matrix.at[all_indexes_bool,1:].set(jnp.array(params_mixing_matrix.reshape((number_frequencies-2,number_components-1),order='F')))
    return new_mixing_matrix
