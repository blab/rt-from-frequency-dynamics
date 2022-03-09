import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from .datahelpers import prep_dates, prep_sequence_counts

class MLRData():
    def __init__(self, raw_seqs):
        self.dates, date_to_index = prep_dates(raw_seqs.date)
        self.seq_names, self.seq_counts = prep_sequence_counts(raw_seqs, date_to_index=date_to_index)

    def make_numpyro_input(self):
        data = dict()
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        return data

def MLR_numpyro(seq_counts, N, X, tau):
    _, N_variants = seq_counts.shape
    _, N_features = X.shape

    # Sampling parameters
    raw_beta = numpyro.sample("beta", 
                   dist.Normal(0.0, 3.0),
                   sample_shape = (N_features, N_variants-1))
    beta = jnp.column_stack((raw_beta, jnp.zeros(N_features))) # Use zeros so that last column is pivot

    logits = jnp.dot(X, beta) # Logit frequencies by variant

    # Evaluate likelihood
    numpyro.sample("obs", 
                   dist.MultinomialLogits(
                       logits= logits,
                       total_count=jnp.nan_to_num(N)),
                   obs=seq_counts)

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Compute growth advantage from model
    numpyro.deterministic("ga", jnp.exp(beta[-1, :] * tau)) # Last row corresponds to coefficient for linear predictor


class MultinomialLogisticRegression():
    def __init__(self, tau) -> None:
        self.tau = tau # Fixed generation time
        self.make_model()
    
    def make_model(self):
        self.model = MLR_numpyro

    def augment_data(self, data):
        T = len(data["N"])
        data["tau"] = self.tau
        data["X"] = jnp.column_stack((jnp.ones(T), jnp.arange(T))) # Use intercept and time as predictors

