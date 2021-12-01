import jax.numpy as jnp
import numpy
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.reparam import TransformReparam
from numpyro.primitives import param
from numpyro.util import identity
from .modelfunctions import v_fs_I, reporting_to_vec
from .LAS import LAS_Laplace


def _fixed_lineage_model_factory(g_rev, delays, seed_L):
    def _lineage_model(cases, seq_counts, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape

        # This is where priors would come in 
        # beta_priors(self.priors)
        # with numpyro.plate("k", k):
        #     beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

        beta = LAS_Laplace("beta", k)

        with numpyro.plate("N_variant_m1", N_variant-1):
            v = numpyro.sample("v", dist.Normal(0.0, 1.0))
        ga = numpyro.deterministic("ga", jnp.exp(v))

        R = numpyro.deterministic("R", jnp.exp((X@beta + jnp.append(v, 0.0)[:, None])).T)
        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 30000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = v_fs_I(I0, R, g_rev, delays, seed_L)
        I_smooth = numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * I_prev[seed_L:,:])

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        total_smooth_prev = numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * I_prev.sum(axis=1)[seed_L:])
        EC = numpyro.deterministic("EC", total_prev[seed_L:] * rho_vec)

        # NegativeBinomial sampling per region
        sqrt_inv_phi = numpyro.sample("raw_phi", dist.HalfNormal(1.))
        numpyro.sample("cases",
                        dist.NegativeBinomial2(mean=EC, concentration=1/sqrt_inv_phi**2),
                        obs=cases)

        # Compute frequency
        freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
        R_ave = numpyro.deterministic("R_ave", (R * freq).sum(axis=1))

        numpyro.sample("Y",
                        dist.Multinomial(total_count=N, probs=freq),
                        obs=seq_counts)               
    return _lineage_model

def _fixed_lineage_guide_factory(g_rev, delays, seed_L):
    def _lineage_guide(cases, seq_counts, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape
        
        # upper = cases[0:14].mean()
        # freqs = seq_counts[0:14].mean(axis=0)

        gam = numpyro.sample(
            "gam", 
            dist.Exponential(
                rate=numpyro.param("gam_sigma", 1. * jnp.ones(1), constraint=dist.constraints.positive)
            )
        )
            
        beta = numpyro.sample(
            "beta",
            dist.Normal(
                loc = numpyro.param("beta_mu", 0.01 * jnp.ones((k,))),
                scale = numpyro.param("beta_sigma", 0.2 * jnp.ones((k,)) , constraint=dist.constraints.positive)
            )
        )

        with numpyro.plate("N_variant_m1", N_variant-1):
            v = numpyro.sample(
                "v",
                dist.Normal(
                    loc = numpyro.param("v_mu", jnp.zeros(1)),
                    scale = numpyro.param("v_sigma", 0.1 * jnp.ones(1), constraint=dist.constraints.positive)
                    )
                )

        with numpyro.plate("N_variant", N_variant):
                I0 = numpyro.sample(
                    "I0",
                        dist.TruncatedNormal(
                            low=jnp.zeros(1) + 0.001,
                            loc=numpyro.param("I0_mu", 300.0 * jnp.ones(1)),
                            scale=numpyro.param("I0_sigma", 20. * jnp.ones(1), constraint=dist.constraints.positive)
                            )
                )
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample(
                    "rho", 
                    dist.BetaProportion(
                        mean=numpyro.param("rho_mu", 0.5 * jnp.ones(1), constraint=dist.constraints.unit_interval),
                        concentration=numpyro.param("rho_conc", 10.0 * jnp.ones(1), constraint=dist.constraints.positive)
                    ))
                
        sqrt_inv_phi = numpyro.sample(
                "raw_phi",
                dist.TruncatedNormal(
                    low=jnp.zeros(1) + 0.001,
                    loc= numpyro.param("raw_phi_mu", 10.0 * jnp.ones(1)),
                    scale = numpyro.param("raw_phi_sigma", 0.2 * jnp.ones(1), constraint=dist.constraints.positive)
                 ),
                ) 
    return _lineage_guide

def _free_lineage_model_factory(g_rev, delays, seed_L):
    def _lineage_model(cases, seq_counts, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape
        with numpyro.plate("variant_beta", N_variant):
            beta = LAS_Laplace("beta", k)
        
        R = numpyro.deterministic("R", jnp.exp(X@beta.T))
        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 30000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = v_fs_I(I0, R, g_rev, delays, seed_L)
        I_smooth = numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * I_prev[seed_L:,:])

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        total_smooth_prev = numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * I_prev.sum(axis=1)[seed_L:])
        EC = numpyro.deterministic("EC", total_prev[seed_L:] * rho_vec)

        # NegativeBinomial sampling per region
        sqrt_inv_phi = numpyro.sample("raw_phi", dist.HalfNormal(1.))
        numpyro.sample("cases",
                        dist.NegativeBinomial2(mean=EC, concentration=1/sqrt_inv_phi**2),
                        obs=cases)

        # Compute frequency
        freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
        numpyro.sample("Y",
                        dist.Multinomial(total_count=N, probs=freq),
                        obs=seq_counts)   
    return _lineage_model