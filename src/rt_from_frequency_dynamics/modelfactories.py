import jax.numpy as jnp
from jax.scipy.special import logit, expit
import numpyro
import numpyro.distributions as dist
from .modelfunctions import v_fs_I, reporting_to_vec
from .LAS import LaplaceRandomWalk


def _fixed_lineage_model_factory(g_rev, delays, seed_L):
    def _lineage_model(cases, seq_counts, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape

        # Locally adaptive smoothing on base R trajectories
        gam = numpyro.sample("gam", dist.HalfCauchy(0.5))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0., 1.))
        beta_rw = numpyro.sample("beta_rw", LaplaceRandomWalk(scale=gam, num_steps=k))
        beta = beta_0 + beta_rw

        # Getting growth variant growth advantages
        with numpyro.plate("N_variant_m1", N_variant-1):
            v = numpyro.sample("v", dist.Normal(0.0, 1.0))
        ga = numpyro.deterministic("ga", jnp.exp(v))

        R = numpyro.deterministic("R", jnp.exp((X@beta + jnp.append(v, 0.0)[:, None])).T)
        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 300_000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = jnp.clip(v_fs_I(I0, R, g_rev, delays, seed_L),  a_min=0., a_max=1e25)
        I_smooth = numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * I_prev[seed_L:,:])

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        total_smooth_prev = numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * total_prev[seed_L:])
        EC = numpyro.deterministic("EC", total_prev[seed_L:] * rho_vec)

        # NegativeBinomial sampling per region
        sqrt_inv_phi = numpyro.sample("raw_phi", dist.HalfNormal(10.))
        numpyro.sample("cases",
                        dist.NegativeBinomial2(mean=EC, concentration=1/sqrt_inv_phi**2),
                        obs=cases)

        # Compute frequency
        freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
        R_ave = numpyro.deterministic("R_ave", (R * freq).sum(axis=1))
        
        # Over-dispersion parameter for multinomial
        xi = numpyro.sample("xi", dist.Beta(1, 99))
        trans_xi = 1 / xi - 1

        numpyro.sample("Y",
                        dist.DirichletMultinomial(total_count=N, concentration= 1e-8 + trans_xi*freq),
                        obs=seq_counts)               
    return _lineage_model

# def _fixed_lineage_guide_factory(g_rev, delays, seed_L):
#     def _lineage_guide(cases, seq_counts, N, X):
#         L, N_variant = seq_counts.shape
#         T, k = X.shape
        
#         gam = numpyro.sample(
#             "gam", 
#             dist.Gamma(
#                 concentration=numpyro.param("gam_con", 1 * jnp.ones(1), constraints=dist.constraints.positive),
#                 rate=numpyro.param("gam_rate", 1. * jnp.ones(1), constraint=dist.constraints.positive)
#             )
#         )
            
#         beta_loc = numpyro.param("b_loc", jnp.zeros(k))
#         beta_scale_tril = numpyro.param("b_scale_tril", jnp.identity(k), constraint=dist.constraints.scaled_unit_lower_cholesky)

#         beta = numpyro.sample(
#             "beta",
#             dist.TransformedDistribution(
#                 dist.Normal(loc=jnp.zeros(k)),
#                 transforms=dist.transforms.LowerCholeskyAffine(beta_loc, beta_scale_tril)
#             )
#         )

#         # beta = numpyro.sample(
#         #     "beta",
#         #     dist.Normal(
#         #         loc = numpyro.param("beta_mu", 0.01 * jnp.ones((k,))),
#         #         scale = numpyro.param("beta_sigma", 0.2 * jnp.ones((k,)) , constraint=dist.constraints.positive)
#         #     )
#         # )

#         v_loc = numpyro.param("v_loc", jnp.zeros((N_variant-1,)))
#         v_scale_tril = numpyro.param("v_scale_tril", jnp.identity(N_variant-1), constraint=dist.constraints.scaled_unit_lower_cholesky)

#         v = numpyro.sample(
#             "v",
#             dist.TransformedDistribution(
#                 dist.Normal(loc=jnp.zeros(N_variant-1)),
#                 transforms=dist.transforms.LowerCholeskyAffine(v_loc, v_scale_tril)
#             )
#         )
#         # with numpyro.plate("N_variant_m1", N_variant-1):
#         #     v = numpyro.sample(
#         #         "v",
#         #         dist.Normal(
#         #             loc = numpyro.param("v_mu", 1e-4 * jnp.ones(1)),
#         #             scale = numpyro.param("v_sigma", 0.1 * jnp.ones(1), constraint=dist.constraints.positive)
#         #             )
#         #         )

#         with numpyro.plate("N_variant", N_variant):
#                 I0 = numpyro.sample(
#                     "I0",
#                         dist.TruncatedNormal(
#                             low=jnp.ones(1) * 0.001,
#                             loc=numpyro.param("I0_mu", 300.0 * jnp.ones(1)),
#                             scale=numpyro.param("I0_sigma", 20. * jnp.ones(1), constraint=dist.constraints.positive)
#                             )
#                 )
            
#         with numpyro.plate("rho_parms", 7):
#             rho = numpyro.sample(
#                     "rho", 
#                     dist.Beta(
#                         concentration0=numpyro.param("rho_c1", 5.0 * jnp.ones(1), constraint=dist.constraints.positive),
#                         concentration1=numpyro.param("rho_c2", 5.0 * jnp.ones(1), constraint=dist.constraints.positive)
#                     ))
                
#         sqrt_inv_phi = numpyro.sample(
#                 "raw_phi",
#                 dist.TruncatedNormal(
#                     low=jnp.zeros(1) + 0.001,
#                     loc= numpyro.param("raw_phi_mu", 2 * jnp.ones(1)),
#                     scale = numpyro.param("raw_phi_sigma", 1. * jnp.ones(1), constraint=dist.constraints.positive)
#                  ),
#                 ) 
#     return _lineage_guide

def _free_lineage_model_factory(g_rev, delays, seed_L):
    def _lineage_model(cases, seq_counts, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape

        # Locally adaptive smoothing on all R trajectories
        gam = numpyro.sample("gam", dist.HalfCauchy(0.5))
        with numpyro.plate("variant_beta", N_variant):
            beta_0 = numpyro.sample("beta_0", dist.Normal(0., 1.))
            beta_rw = numpyro.sample("beta_rw", LaplaceRandomWalk(scale=gam, num_steps=k))
            beta = beta_0 + beta_rw.T
        
        R = numpyro.deterministic("R", jnp.exp(X@beta))
        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 300_000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = v_fs_I(I0, R, g_rev, delays, seed_L)
        I_smooth = numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * I_prev[seed_L:,:])

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        total_smooth_prev = numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * total_prev[seed_L:])
        EC = numpyro.deterministic("EC", total_prev[seed_L:] * rho_vec)

        # NegativeBinomial sampling per region
        sqrt_inv_phi = numpyro.sample("raw_phi", dist.HalfNormal(10.))
        numpyro.sample("cases",
                        dist.NegativeBinomial2(mean=EC, concentration=1/sqrt_inv_phi**2),
                        obs=cases)

        # Compute frequency
        freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
        R_ave = numpyro.deterministic("R_ave", (R * freq).sum(axis=1))

        # Over-dispersion parameter for multinomial
        xi = numpyro.sample("xi", dist.Beta(1, 99))
        trans_xi = numpyro.deterministic("trans_xi", 1 / xi - 1)

        numpyro.sample("Y",
                        dist.DirichletMultinomial(total_count=N, concentration=1e-8+trans_xi*freq),
                        obs=seq_counts)               
    return _lineage_model



def _decomp_lineage_model_factory(g_rev, delays, phi_0, seed_L):
    N_pop = 7_000_000
    def _lineage_model(cases, seq_counts, vacc, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape

        # This is where priors would come in 
        # beta_priors(self.priors)
        # with numpyro.plate("k", k):
        #     beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

        beta = LAS_Laplace("beta", k)

        with numpyro.plate("N_variant_m1", N_variant-1):
            eta_T = numpyro.sample("eta_T", dist.Normal(0.0, 0.1))
            eta_E = numpyro.sample("eta_E", dist.HalfNormal(0.01))

        # Parameterize immune fraction
        phi_delta_0 = numpyro.sample("phi_delta_init", dist.Normal(loc=0, scale=1))
        with numpyro.plate("k_phi", k-1):
            phi_delta = numpyro.sample("phi_delta", dist.HalfNormal(0.5))
        beta_phi = jnp.cumsum(jnp.concatenate([jnp.array([0.]), phi_delta])) + phi_delta_0
        phi = numpyro.deterministic("phi", expit(X@beta_phi))

        # R_advantage = eta_T + (eta_E - eta_T)*phi
        R_base = numpyro.deterministic("R_base", jnp.exp(X@beta) * (1-phi))
        R_adv = numpyro.deterministic("R_adv", jnp.append(eta_T,0) + jnp.append(eta_E - eta_T, 0)*phi[:,None])
        R = numpyro.deterministic("R", jnp.clip(R_base[:, None] + R_adv, a_min=0))

        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 300_000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = v_fs_I(I0, R, g_rev, delays, seed_L)
        I_smooth = numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * I_prev[seed_L:,:])

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        total_smooth_prev = numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * I_prev.sum(axis=1)[seed_L:])
        EC = numpyro.deterministic("EC", total_prev[seed_L:] * rho_vec)
        infect_phi = numpyro.deterministic(
            "infect_phi", 
            jnp.clip(phi_0 + jnp.cumsum(total_prev[seed_L:])/ N_pop, a_min=0, a_max=1))
        # infect_phi = numpyro.deterministic("infect_phi", phi_0_dist)

        # Evaluate immune likelihood
        phi_conc = numpyro.sample("phi_conc", dist.HalfNormal(scale=0.1))
        numpyro.sample(
            "phi_obs", 
            dist.Normal(loc=phi, scale=phi_conc),
            obs = vacc + infect_phi - vacc * infect_phi)

        # NegativeBinomial sampling per region
        sqrt_inv_phi = numpyro.sample("raw_phi", dist.HalfNormal(10.))
        numpyro.sample("cases",
                        dist.NegativeBinomial2(mean=EC, concentration=1/sqrt_inv_phi**2),
                        obs=cases)

        # Compute frequency
        freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
        R_ave = numpyro.deterministic("R_ave", (R * freq).sum(axis=1))
        
        # Over-dispersion parameter for multinomial
        xi = numpyro.sample("xi", dist.Beta(concentration0=1, concentration1=99))
        trans_xi = 1 / xi - 1

        numpyro.sample("Y",
                        dist.DirichletMultinomial(total_count=N, concentration=0.001+trans_xi*freq),
                        obs=seq_counts)               
    return _lineage_model

def _GARW_model_factory(g_rev, delays, seed_L):
    def _lineage_model(cases, seq_counts, N, X):
        L, N_variant = seq_counts.shape
        T, k = X.shape

        # Time varying base trajectory
        gam = numpyro.sample("gam", dist.HalfCauchy(0.5))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 1.0))
        beta_rw = numpyro.sample("beta_rw", LaplaceRandomWalk(scale=gam, num_steps=k))
        beta = beta_0 + beta_rw

        # Time varying growth advantage as random walk
        # Regularizes changes in growth advantage of variants
        gam_delta = numpyro.sample("gam_delta", dist.Exponential(rate=10))
        with numpyro.plate("N_variant_m1", N_variant-1):
            delta_0 = numpyro.sample("delta_0", dist.Normal(0.0, 1.0))
            delta_rw = numpyro.sample("delta_rw", dist.GaussianRandomWalk(scale=gam_delta, num_steps=k))
            delta = delta_0 + delta_rw.T
        
        beta_mat = beta[:, None] + jnp.hstack((delta, jnp.zeros((k,1))))
        R = numpyro.deterministic("R", jnp.exp((X@beta_mat)))

        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 300_000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = jnp.clip(v_fs_I(I0, R, g_rev, delays, seed_L),  a_min=0., a_max=1e25)
        I_smooth = numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * I_prev[seed_L:,:])

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        total_smooth_prev = numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * total_prev[seed_L:])
        EC = numpyro.deterministic("EC", total_prev[seed_L:] * rho_vec)

        # NegativeBinomial sampling per region
        sqrt_inv_phi = numpyro.sample("raw_phi", dist.HalfNormal(10.))
        numpyro.sample("cases",
                        dist.NegativeBinomial2(mean=EC, concentration=1/sqrt_inv_phi**2),
                        obs=cases)

        # Compute frequency
        freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
        R_ave = numpyro.deterministic("R_ave", (R * freq).sum(axis=1))
        
        # Over-dispersion parameter for multinomial
        xi = numpyro.sample("xi", dist.Beta(1, 99))
        trans_xi = 1 / xi - 1

        numpyro.sample("Y",
                        dist.DirichletMultinomial(total_count=N, concentration= 1e-8 + trans_xi*freq),
                        obs=seq_counts)               

    return _lineage_model
