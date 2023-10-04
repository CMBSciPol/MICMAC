from collections import namedtuple
from jax import random, dtypes
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC

# Copy-pasted from https://num.pyro.ai/en/0.5.0/mcmc.html#numpyro.infer.mcmc.MCMCKernel


MHState = namedtuple("MHState", ["u", "rng_key"])

class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "u"

    def __init__(self, potential_fn, step_size=0.1):
        self.potential_fn = potential_fn
        self.step_size = step_size

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        accept_prob = jnp.exp(self.potential_fn(u, model_kwargs) - self.potential_fn(u_proposal, model_kwargs))
        u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
        return MHState(u_new, rng_key)

def f(x, model_kwargs):
    print('Test !', model_kwargs, flush=True)
    y = model_kwargs['y']
    return ((x + y - 2) ** 2).sum()

# def f(x):
#     return -.5*((x-1)/2)**2 



kernel = MetropolisHastings(f)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
mcmc.run(random.PRNGKey(0), init_params=jnp.array([1., 2.]), y=jnp.array(2), z=jnp.array(0))
samples = mcmc.get_samples()
mcmc.print_summary() 
