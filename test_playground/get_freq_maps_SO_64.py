import numpy as np
from fgbuster.observation_helpers import *
from non_parametric_ML_compsep import *


# General parameters
NSIDE = 64
model = 'c1d0s0'
noise = True
noise_seed = 42
instr_name = 'SO_SAT'

# get instrument from public database
instrument = get_instrument(instr_name)

# get input freq maps
np.random.seed(noise_seed)
freq_maps = get_observation(instrument, model, nside=NSIDE, noise=noise)[:, 1:, :]   # keep only Q and U
print(freq_maps.shape)

