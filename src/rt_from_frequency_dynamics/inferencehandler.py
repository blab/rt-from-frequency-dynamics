from jax import random, lax
from jax import jit
import jax.example_libraries.optimizers as optimizers
import jax.numpy as jnp

from numpyro.infer import SVI, NUTS, MCMC, Predictive, Trace_ELBO
from numpyro.infer.svi import SVIState

import pickle


class SVIHandler:
    def __init__(
        self, rng_key=1, loss=Trace_ELBO(num_particles=2), optimizer=None
    ):
        self.rng_key = random.PRNGKey(rng_key)
        self.loss = loss
        self.optimizer = optimizer
        self.svi = None
        self.svi_state = None

    def init_svi(self, model, guide, data):
        self.svi = SVI(model, guide, self.optimizer, self.loss)
        svi_state = self.svi.init(self.rng_key, **data)
        if self.svi_state is None:
            self.svi_state = svi_state

    def fit(self, model, guide, data, n_epochs):
        self.init_svi(model, guide, data)
        self.svi_result = (
            self.svi.run(self.rng_key, n_epochs, **data, progress_bar=False, stable_update=True)
        )
        self.svi_state = self.svi_result.state

    @property
    def params(self):
        if self.svi and self.svi_state:
            return self.svi.get_params(self.svi_state)

    @property
    def losses(self):
        if self.svi_result:
            return self.svi_result.losses

    def predict(self, model, guide, data, **kwargs):
        if self.svi is None:
            self.init_svi(model, guide, data)

        original_pred = Predictive(guide, params=self.params, **kwargs)
        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples = original_pred(rng_key_, **data)
        predictive = Predictive(model, samples)

        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples_pred = predictive(rng_key_, pred=True, **data)
        return {**samples, **samples_pred}

    def reset_state(self):
        return SVIHandler(self.rng_key, self.loss, self.optimizer)

    # Optim state contains number of iterations and current state
    @property
    def optim_state(self):
        if self.svi_state is not None:
            return self.svi_state.optim_state

    def save_state(self, fp):
        with open(fp, "wb") as f:
            pickle.dump(
                optimizers.unpack_optimizer_state(self.optim_state[1]), f
            )

    def load_state(self, fp):
        with open(fp, "rb") as f:
            optim_state = (0, optimizers.pack_optimizer_state(pickle.load(f)))
        self.svi_state = SVIState(optim_state, None, self.rng_key)


class MCMCHandler:
    def __init__(self, rng_key=1, kernel=None):
        if kernel is None:
            kernel = NUTS
        self.rng_key = random.PRNGKey(rng_key)
        self.kernel = kernel
        self.mcmc = None

    def fit(self, model, data, num_warmup, num_samples, **kwargs):
        self.mcmc = MCMC(
            self.kernel(model),
            num_warmup=num_warmup,
            num_samples=num_samples,
            **kwargs,
        )
        self.mcmc.run(self.rng_key, **data)
        self.samples = self.mcmc.get_samples()

    @property
    def params(self):
        if self.samples is not None:
            return self.samples
        return None

    def save_state(self, fp):
        if self.samples is None:
            return None
        with open(fp, "wb") as f:
            jnp.save(f, self.samples)

    def load_state(self, fp):
        with open(fp, "rb") as f:
            self.samples = jnp.load(f)

    def predict(self, model, data, **kwargs):
        predictive = Predictive(model, self.params)
        self.rng_key, rng_key_ = random.split(self.rng_key)
        samples_pred = predictive(rng_key_, pred=True, **data)
        return {**self.params, **samples_pred}
