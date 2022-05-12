from jax import random, lax
from jax import jit
import jax.example_libraries.optimizers as optimizers
import jax.numpy as jnp

from numpyro.infer import SVI, NUTS, MCMC, Predictive, Trace_ELBO
from numpyro.infer.svi import SVIState

import pickle


class SVIHandler:
    def __init__(self, rng_key=1, loss=Trace_ELBO(num_particles=2), optimizer=None):
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
        return self

    def _fit(self, data, n_epochs):
        @jit
        def train(svi_state, n_epochs):
            def _train_single(_, val):
                loss, svi_state = val
                svi_state, loss = self.svi.stable_update(svi_state, **data)
                return loss, svi_state

            return lax.fori_loop(0, n_epochs, _train_single, (0.0, svi_state))

        loss, self.svi_state = train(self.svi_state, n_epochs)
        return loss

    def fit(self, model, guide, data, n_epochs, log_each=10000):
        self.init_svi(model, guide, data)
        if log_each == 0:
            self._fit(data, n_epochs)
        else:
            this_loss = self.svi.evaluate(self.svi_state, **data)

            # Can this be done in a while loop?
            this_epoch = 0
            print(f"Epoch: {this_epoch}. Loss: {this_loss}")
            for i in range(n_epochs // log_each):
                this_epoch += log_each
                this_loss = self._fit(data, n_epochs)
                print(f"Epoch: {this_epoch}. Loss: {this_loss}")
            if n_epochs % log_each:
                this_epoch += n_epochs % log_each
                this_loss = self._fit(data, n_epochs % log_each)
        loss = self.svi.evaluate(self.svi_state, **data)
        self.rng_key = self.svi_state.rng_key
        return loss

    @property
    def params(self):
        if self.svi_state is not None:
            return self.svi.get_params(self.svi_state)
        else:
            return None

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
            pickle.dump(optimizers.unpack_optimizer_state(self.optim_state[1]), f)

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
            self.kernel(model), num_warmup=num_warmup, num_samples=num_samples, **kwargs
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
