import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from .modelfunctions import v_fs_I, reporting_to_vec
from .modeloptions import GARW, NegBinomCases, DirMultinomialSeq

def _model_factory(g_rev, 
                   delays, 
                   seed_L, 
                   forecast_L, 
                   RLik=None, 
                   CaseLik=None, 
                   SeqLik=None):
    if RLik is None:
        RLik = GARW() 
    if CaseLik is None:
        CaseLik = NegBinomCases()
    if SeqLik is None:
        SeqLik = DirMultinomialSeq()

    def _lineage_model(cases, seq_counts, N, X):
        T, N_variant = seq_counts.shape
        obs_range = jnp.arange(seed_L, seed_L+T, 1)

        _R = RLik.model(N_variant, X)

        # Add forecasted R
        if forecast_L > 0:
            R_forecast = numpyro.deterministic(
                "R_forecast",
                jnp.vstack((_R[-1,:],)*forecast_L)
                 )
            R = jnp.vstack((_R, R_forecast))
        else:
            R = _R

        # Getting initial conditions
        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 300_000.0))
            
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5., 5.))
        rho_vec = reporting_to_vec(rho, T)

        I_prev = jnp.clip(
            v_fs_I(I0, R, g_rev, delays, seed_L),  
            a_min=0., 
            a_max=1e25)
        
        # Smooth trajectory for plotting
        numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * jnp.take(I_prev, obs_range, axis=0))

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        numpyro.deterministic("total_smooth_prev", jnp.mean(rho_vec) * jnp.take(total_prev, obs_range))
        EC = numpyro.deterministic("EC", 
                                   jnp.take(total_prev, obs_range) * rho_vec)

        # Evaluate case likelihood
        CaseLik.model(cases, EC)

        # Compute frequency
        _freq = jnp.divide(I_prev, total_prev[:, None])
        freq = numpyro.deterministic("freq", jnp.take(_freq, obs_range, axis=0))

        # Evaluate frequency likelihood
        SeqLik.model( seq_counts, N, freq)

        ## Misc
        # Getting average R
        numpyro.deterministic("R_ave", (_R * freq).sum(axis=1))

        if forecast_L > 0:
            numpyro.deterministic("freq_forecast", _freq[(seed_L+T):, :])
            numpyro.deterministic("I_forecast", 
                                  jnp.mean(rho_vec) * I_prev[(seed_L+T):,:]
                                  )
    return _lineage_model
