import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro.distributions import constraints, HalfCauchy, Laplace
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    validate_sample,
)

class LaplaceRandomWalk(Distribution):
    arg_constraints = {"scale": constraints.positive}
    support = constraints.real_vector
    reparametrized_params = ["scale"]

    def __init__(self, scale=1.0, num_steps=1, validate_args=None):
        assert (
            isinstance(num_steps, int) and num_steps > 0
        ), "`num_steps` argument should be an positive integer."
        self.scale = scale
        self.num_steps = num_steps
        batch_shape, event_shape = jnp.shape(scale), (num_steps,)
        super(LaplaceRandomWalk, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape
        walks = random.laplace(key, shape=shape)
        return jnp.cumsum(walks, axis=-1) * jnp.expand_dims(self.scale, axis=-1)


    @validate_sample
    def log_prob(self, value):
        init_prob = Laplace(0.0, self.scale).log_prob(value[..., 0])
        scale = jnp.expand_dims(self.scale, -1)
        step_probs = Laplace(value[..., :-1], scale).log_prob(value[..., 1:])
        return init_prob + jnp.sum(step_probs, axis=-1)

    @property
    def mean(self):
        return jnp.zeros(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return jnp.broadcast_to(
            jnp.expand_dims(self.scale, -1) ** 2 * jnp.arange(1, self.num_steps + 1),
            self.batch_shape + self.event_shape,
        )

    def tree_flatten(self):
        return (self.scale,), self.num_steps

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls(*params, num_steps=aux_data)

def LAS_Laplace(beta_name, k):
    gam = numpyro.sample("gam", HalfCauchy(0.5))
    beta = numpyro.sample(beta_name, LaplaceRandomWalk(scale=gam, num_steps=k))
    return beta 