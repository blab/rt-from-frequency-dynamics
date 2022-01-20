import jax.numpy as jnp
from jax import ops
import numpy as np
import numpyro
import numpyro.distributions as dist

from .LAS import LaplaceRandomWalk
from .modelhelpers import is_obs_idx, pad_to_obs


class FixedGA():
    def __init__(self, gam_prior=0.5):
        self.gam_prior = gam_prior

    def model(self, N_variant, X):
        _, k = X.shape

        # Locally adaptive smoothing
        gam = numpyro.sample("gam", dist.HalfCauchy(self.gam_prior))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0., 1.))
        beta_rw = numpyro.sample("beta_rw", 
                                 LaplaceRandomWalk(scale=gam, num_steps=k))
        beta = beta_0 + beta_rw

        # Getting log variant growth advantages
        with numpyro.plate("N_variant_m1", N_variant-1):
            v = numpyro.sample("v", dist.Normal(0.0, 1.0))
        
        numpyro.deterministic("ga", jnp.exp(v)) # Transform to growth advantage
        
        # Computing R
        R = numpyro.deterministic("R", 
                                   jnp.exp(
                                       (X@beta + jnp.append(v, 0.0)[:, None])
                                   ).T)
        return R

class FreeGrowth():
    def __init__(self, gam_prior=0.5):
        self.gam_prior = gam_prior

    def model(self, N_variant, X):
        _, k = X.shape

        # Locally adaptive smoothing on all R trajectories
        gam = numpyro.sample("gam", dist.HalfCauchy(self.gam_prior))
        with numpyro.plate("variant_beta", N_variant):
            beta_0 = numpyro.sample("beta_0", dist.Normal(0., 1.))
            beta_rw = numpyro.sample("beta_rw", 
                                     LaplaceRandomWalk(scale=gam, num_steps=k))
            beta = beta_0 + beta_rw.T

        # Computing R
        R = numpyro.deterministic("R", jnp.exp(X@beta))
        return R

class GARW():
    def __init__(self, gam_prior=0.5, gam_delta_prior=10):
        self.gam_prior = gam_prior
        self.gam_delta_prior = gam_delta_prior

    def model(self, N_variant, X):
        _, k = X.shape

        # Time varying base trajectory
        gam = numpyro.sample("gam", dist.HalfCauchy(self.gam_prior))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0))
        beta_rw = numpyro.sample("beta_rw", 
                                 LaplaceRandomWalk(scale=gam, num_steps=k))
        beta = beta_0 + beta_rw

        # Time varying growth advantage as random walk
        # Regularizes changes in growth advantage of variants
        with numpyro.plate("N_variant_m1", N_variant-1):
            delta_0 = numpyro.sample("delta_0", dist.Normal(0.0, 0.5))
            gam_delta = numpyro.sample("gam_delta", dist.Exponential(rate=self.gam_delta_prior))
            delta_rw = numpyro.sample("delta_rw", 
                                      dist.GaussianRandomWalk(
                                          scale=gam_delta, 
                                          num_steps=k)
                                      )
            delta = delta_0 + delta_rw.T
        
        # Transform to growth advatnage
        numpyro.deterministic("ga", jnp.exp(X@delta))

        # Construct beta matrix
        beta_mat = beta[:, None] + jnp.hstack((delta, jnp.zeros((k,1))))
        R = numpyro.deterministic("R", jnp.exp((X@beta_mat)))
        return R

class PoisCases():
    def __init__(self):
        pass

    def model(self, cases, EC):
        is_obs = is_obs_idx(cases) # Find unoberserved case counts
        numpyro.sample("cases",
                       dist.Poisson(pad_to_obs(EC, is_obs)),
                       obs=jnp.nan_to_num(cases))
class ZIPoisCases():
    def __init__(self):
        pass

    def model(self, cases, EC):

        # Finding zero locations and making only parameters for zero observation
        is_zero = (cases == 0)
        zero_idx = np.nonzero(is_zero)[0]

        gate = jnp.zeros_like(cases) + 1e-12
        with numpyro.plate("n_zero", zero_idx.shape[0]):
            zp = numpyro.sample("zp", dist.Beta(0.1, 0.1))
        gate = ops.index_update(gate, zero_idx, zp)

        numpyro.sample("cases",
                       dist.ZeroInflatedPoisson(rate=EC, gate=gate),
                       obs=jnp.nan_to_num(cases))

class NegBinomCases():
    def __init__(self, raw_alpha_sd=0.01):
        self.raw_alpha_sd = raw_alpha_sd

    def model(self, cases, EC):
        is_obs = is_obs_idx(cases) # Find unoberserved case counts
        # NegativeBinomial sampling
        raw_alpha = numpyro.sample("raw_alpha", 
                                   dist.HalfNormal(self.raw_alpha_sd))
        numpyro.sample("cases",
                       dist.NegativeBinomial2(
                           mean=pad_to_obs(EC, is_obs), 
                           concentration=jnp.power(raw_alpha, -2)),
                       obs=jnp.nan_to_num(cases))

class ZINegBinomCases():
    def __init__(self, raw_alpha_sd=0.01):
        self.raw_alpha_sd = raw_alpha_sd

    def model(self, cases, EC):
        # NegativeBinomial sampling
        T = cases.shape[0]
        raw_alpha = numpyro.sample("raw_alpha", 
                                   dist.HalfNormal(self.raw_alpha_sd))

        # Finding zero locations and making only parameters for zero observation
        is_zero = (cases == 0)
        zero_idx = np.nonzero(is_zero)[0]

        gate = jnp.zeros_like(cases) + 1e-12
        with numpyro.plate("n_zero", zero_idx.shape[0]):
            zp = numpyro.sample("zp", dist.Beta(0.1, 0.1))
        gate = ops.index_update(gate, zero_idx, zp)

        numpyro.sample("cases",
                       dist.ZeroInflatedNegativeBinomial2(
                           mean=EC, 
                           gate=gate,
                           concentration=jnp.power(raw_alpha, -2)),
                       obs=jnp.nan_to_num(cases))

class MultinomialSeq():
    def __init__(self):
        pass
    
    def model(self, seq_counts, N, freq):
        # Sample with Multinomial
        numpyro.sample("Y",
                       dist.Multinomial(
                           total_count=jnp.nan_to_num(N), 
                           probs=freq),
                       obs=jnp.nan_to_num(seq_counts))

class DirMultinomialSeq():
    def __init__(self, xi_prior=99):
        self.xi_prior = xi_prior

    def model(self, seq_counts, N, freq):
        # Overdispersion in sequence counts
        xi = numpyro.sample("xi", dist.Beta(1, self.xi_prior))
        trans_xi = jnp.reciprocal(xi) - 1

        # Sample with DirichletMultinomial
        numpyro.sample("Y",
                        dist.DirichletMultinomial(
                            total_count=jnp.nan_to_num(N), 
                            concentration= 1e-8 + trans_xi*freq),
                        obs= jnp.nan_to_num(seq_counts))
