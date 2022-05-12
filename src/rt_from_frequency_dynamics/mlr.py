import jax.numpy as jnp
import numpy as np
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from .datahelpers import prep_dates, prep_sequence_counts


class MLRData:
    def __init__(self, raw_seqs):
        self.dates, date_to_index = prep_dates(raw_seqs.date)
        self.seq_names, self.seq_counts = prep_sequence_counts(
            raw_seqs, date_to_index=date_to_index
        )

    def make_numpyro_input(self):
        data = dict()
        data["seq_counts"] = self.seq_counts
        data["N"] = self.seq_counts.sum(axis=1)
        return data


def MLR_numpyro(seq_counts, N, X, tau=None, pred=False):
    _, N_variants = seq_counts.shape
    _, N_features = X.shape

    # Sampling parameters
    raw_beta = numpyro.sample(
        "raw_beta", dist.Normal(0.0, 3.0), sample_shape=(N_features, N_variants - 1)
    )

    beta = numpyro.deterministic(
        "beta",
        jnp.column_stack(
            (raw_beta, jnp.zeros(N_features))
        ),  # All parameters are relative to last column / variant
    )

    logits = jnp.dot(X, beta)  # Logit frequencies by variant

    # Evaluate likelihood
    obs = None if pred else np.nan_to_num(seq_counts)
    numpyro.sample(
        "obs",
        dist.MultinomialLogits(logits=logits, total_count=np.nan_to_num(N)),
        obs=obs,
    )

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(beta[-1, :] * tau)
        )  # Last row corresponds to linar predictor / growth advantage


class MultinomialLogisticRegression:
    def __init__(self, tau) -> None:
        self.tau = tau  # Fixed generation time
        self.make_model()

    def make_model(self):
        self.model = MLR_numpyro

    @staticmethod
    def make_ols_feature(start, stop):
        """
        Given start and stop time, return feature matrix for MLR.
        One column is the bias term (all ones), the second is date as integer.
        """
        t = jnp.arange(start=start, stop=stop)
        return jnp.column_stack((jnp.ones_like(t), t))

    def augment_data(self, data):
        T = len(data["N"])
        data["tau"] = self.tau
        data["X"] = self.make_ols_feature(0, T)  # Use intercept and time as predictors


class MLR_General:
    def __init__(self, X) -> None:
        self.X = X  # Feature matrix. shape: N_time x N_strain
        self.make_model()

    def make_model(self):
        self.model = MLR_numpyro

    def augment_data(self, data):
        data["X"] = self.X
