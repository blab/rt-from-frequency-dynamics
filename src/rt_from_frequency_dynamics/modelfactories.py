import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
import numpyro
import numpyro.distributions as dist
from .modelfunctions import forward_simulate_I, reporting_to_vec
from .modeloptions import GARW, NegBinomCases, DirMultinomialSeq


def _renewal_model_factory(
    g_rev,
    delays,
    seed_L,
    forecast_L,
    RLik=None,
    CaseLik=None,
    SeqLik=None,
    v_names=None,
):
    if RLik is None:
        RLik = GARW()
    if CaseLik is None:
        CaseLik = NegBinomCases()
    if SeqLik is None:
        SeqLik = DirMultinomialSeq()

    # If single generation time
    if g_rev.ndim == 1:
        gmap_dim = None  # Use same generation time
    else:
        gmap_dim = 0  # Use each row
        # Specifying variant name map to column
        if v_names is not None:
            which_col = {s: i for i, s in enumerate(v_names)}

    v_fs_I = jit(
        vmap(
            forward_simulate_I,
            in_axes=(-1, -1, gmap_dim, None, None),
            out_axes=-1,
        ),
        static_argnums=4,
    )

    def _variant_model(cases, seq_counts, N, X, seq_names, pred=False):
        T, N_variant = seq_counts.shape
        obs_range = jnp.arange(seed_L, seed_L + T, 1)

        # Computing first introduction dates
        first_obs = (np.ma.masked_invalid(np.array(seq_counts)) != 0).argmax(
            axis=0
        )
        intro_dates = np.concatenate(
            [first_obs + d for d in np.arange(0, seed_L)]
        )

        intro_idx = (
            intro_dates,
            np.tile(np.arange(N_variant), seed_L),
        )  # Multiple introductions per variant

        _R = RLik.model(
            N_variant, X
        )  # likelihood on effective reproduction number

        # Add forecasted R
        if forecast_L > 0:
            R_forecast = numpyro.deterministic(
                "R_forecast", jnp.vstack((_R[-1, :],) * forecast_L)
            )
            R = jnp.vstack((_R, R_forecast))
        else:
            R = _R

        # Getting initial conditions
        intros = jnp.zeros((T + seed_L + forecast_L, N_variant))
        with numpyro.plate("N_variant", N_variant):
            I0 = numpyro.sample("I0", dist.Uniform(0.0, 300_000.0))
        intros = intros.at[intro_idx].set(jnp.tile(I0, seed_L))

        # Get day of the week parameters
        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5.0, 5.0))
        rho_vec = reporting_to_vec(rho, T)

        _g_rev = g_rev  # Assume we're using original g_rev
        if gmap_dim is not None:  # Match variants to correct generation time
            v_idx = [
                which_col[s] for s in seq_names
            ]  # Find which variants present
            _g_rev = _g_rev[v_idx, :]

        I_prev = jnp.clip(
            v_fs_I(intros, R, _g_rev, delays, seed_L), a_min=0.0, a_max=1e25
        )

        # Smooth trajectory for plotting
        numpyro.deterministic(
            "I_smooth", jnp.mean(rho_vec) * jnp.take(I_prev, obs_range, axis=0)
        )

        # Compute growth rate assuming I_{t+1} = I_{t} \exp(r_{t})
        numpyro.deterministic(
            "r",
            jnp.diff(
                jnp.log(jnp.take(I_prev, obs_range, axis=0)),
                prepend=jnp.nan,
                axis=0,
            ),
        )

        # Compute expected cases
        total_prev = I_prev.sum(axis=1)
        numpyro.deterministic(
            "total_smooth_prev",
            jnp.mean(rho_vec) * jnp.take(total_prev, obs_range),
        )
        EC = numpyro.deterministic(
            "EC", jnp.take(total_prev, obs_range) * rho_vec
        )

        # Evaluate case likelihood
        CaseLik.model(cases, EC, pred=pred)

        # Compute frequency
        _freq = jnp.divide(I_prev, total_prev[:, None])
        freq = numpyro.deterministic(
            "freq", jnp.take(_freq, obs_range, axis=0)
        )

        SeqLik.model(
            seq_counts, N, freq, pred
        )  # Evaluate frequency likelihood

        numpyro.deterministic(
            "R_ave", (_R * freq).sum(axis=1)
        )  # Getting average R

        if forecast_L > 0:
            numpyro.deterministic("freq_forecast", _freq[(seed_L + T):, :])
            I_forecast = numpyro.deterministic(
                "I_smooth_forecast",
                jnp.mean(rho_vec) * I_prev[(seed_L + T):, :],
            )
            numpyro.deterministic(
                "r_forecast",
                jnp.diff(jnp.log(I_forecast), prepend=jnp.nan, axis=0),
            )

    return _variant_model


def _exp_model_factory(
    g_rev,
    delays,  # Should be encoded into data tbh, so X, Xprime, Xdelay
    CaseLik=None,
    SeqLik=None,
):
    if CaseLik is None:
        CaseLik = NegBinomCases()
    if SeqLik is None:
        SeqLik = DirMultinomialSeq()

    def _variant_model(cases, seq_counts, N, X, X_prime, pred=False):
        _, N_variant = seq_counts.shape
        T, k = X.shape

        # Need some way of making the R parameter formations a bit more usable
        # Time varying base trajectory
        gam = numpyro.sample("gam", dist.HalfCauchy(0.1))

        beta_rw = numpyro.sample(
            "beta_rw", dist.GaussianRandomWalk(scale=gam, num_steps=k - 1)
        )
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 10.0))
        beta = numpyro.deterministic(
            "beta", beta_0 + jnp.concatenate([jnp.array([0.0]), beta_rw])
        )

        # Time varying growth advantage as random walk
        # Regularizes changes in growth advantage of variants
        gam_delta = numpyro.sample("gam_delta", dist.Exponential(rate=50))
        with numpyro.plate("N_variant_m1", N_variant - 1):
            delta_rw = numpyro.sample(
                "delta_rw",
                dist.GaussianRandomWalk(scale=gam_delta, num_steps=k),
            )

        delta = delta_rw.T
        beta_mat = beta[:, None] + jnp.hstack((delta, jnp.zeros((k, 1))))

        incidence = jnp.exp(jnp.dot(X, beta_mat))  # Variant-specific incidence
        r = numpyro.deterministic("r", X_prime @ beta_mat)

        with numpyro.plate("rho_parms", 7):
            rho = numpyro.sample("rho", dist.Beta(5.0, 5.0))
        rho_vec = reporting_to_vec(rho, T)
        numpyro.deterministic("I_smooth", jnp.mean(rho_vec) * incidence)

        # Evaluate case likelihood
        CaseLik.model(cases, rho_vec * incidence.sum(axis=1), pred=pred)

        # Compute frequency
        freq = numpyro.deterministic(
            "freq", jnp.divide(incidence, incidence.sum(axis=1)[:, None])
        )

        # Evaluate frequency likelihood
        SeqLik.model(seq_counts, N, freq)

        # Getting average R
        numpyro.deterministic("r_ave", (r * freq).sum(axis=1), pred=pred)

    return _variant_model
