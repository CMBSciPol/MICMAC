import numpy as np
from fgbuster.observation_helpers import *
from non_parametric_ML_compsep import *


# General parameters
NSIDE = 128
model = 'd0s0'
noise = True
noise_seed = 42
instr_name = 'LiteBIRD'

# get instrument from public database
instrument = get_instrument('LiteBIRD')

# get input freq maps
np.random.seed(42)
freq_maps = get_observation(instrument, model, nside=NSIDE, noise=noise)[:, 1:, :]   # keep only Q and U
print(freq_maps.shape)

