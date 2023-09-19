import numpy as np

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


    def update_params(self, new_params):
        """
        Update values of the params in the mixing matrix.
        """
        assert np.shape(new_params) == (self.nfreq-self.ncomp+1, self.ncomp-1)
        self.params = new_params

        return
    

    def get_B_fgs(self):
        """
        fgs part of the mixing matrix.
        """
        ncomp_fgs = self.ncomp - 1
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


    def get_B_cmb(self):
        """
        CMB column of the mixing matrix.
        """
        B_cmb = np.ones((self.nfreq))
        B_cmb = B_cmb[:, np.newaxis]
        
        return B_cmb


    def get_B(self):
        """
        Full mixing matrix, (nfreq*ncomp).
        cmb is given as the first component.
        """
        B_mat = np.concatenate((self.get_B_cmb(), self.get_B_fgs()), axis=1)
        
        return B_mat
