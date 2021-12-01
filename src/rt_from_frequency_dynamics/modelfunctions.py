from functools import partial
from jax import lax, jit, vmap
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


@partial(jit, static_argnums=3)
def get_infections(I0, R, g_rev, seed_L):
    l = len(g_rev)

    I0_vec = jnp.repeat(I0, seed_L)

    @jit
    def _scan_infections(infections, R):
        curr_I = R * jnp.dot(infections[-l:], g_rev[-l:])  # New I
        return jnp.append(infections[-(l-1):], curr_I), curr_I

    _, infections = lax.scan(_scan_infections,
                             init=jnp.pad(I0_vec, (l - seed_L, 0)),
                             xs=R)
    return jnp.append(I0_vec, infections)


@jit
def apply_delay(infections, delay):
    I_padded = jnp.pad(infections, (len(delay) - 1, 0), constant_values=0)
    return jnp.convolve(I_padded, delay, mode="valid")


def _apply_delays(infections, delay):
    out = apply_delay(infections, delay)
    return out, out


@partial(jit, static_argnums=4)
def forward_simulate_I(I0, R, gen_rev, delays, seed_L):
    infections = get_infections(I0, R, gen_rev, seed_L)
    infections, _I = lax.scan(_apply_delays, init=infections, xs=delays)
    return infections


v_fs_I = jit(vmap(forward_simulate_I,
                  in_axes=(-1, -1, None, None, None),
                  out_axes=-1), static_argnums=4)


@partial(jit, static_argnums=1)
def reporting_to_vec(rho, L):
    return jnp.pad(rho, (0, L - len(rho)), mode="wrap")


@partial(jit, static_argnums=5)
def forward_simulate_EC(I0, R, rho, gen_rev, delays, seed_L):
    infections = get_infections(I0, R, gen_rev, seed_L)
    infections, _I = lax.scan(_apply_delays, init=infections, xs=delays)
    return rho * infections[seed_L:]


def model_lineage(gen_rev, delays, X, seed_L, N_lineage,
                  N=None, Y=None, cases=None):
    T = X.shape[0]
    k = X.shape[1]
    # R is a linear regression
    with numpyro.plate("k", k):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))

    # With lineage-specific advantages
    with numpyro.plate("N_lineage_m1", N_lineage-1):
        v = numpyro.sample("v", dist.Normal(0.0, 1.0))
    
    R = jnp.exp((X @ beta + jnp.append(0.0, v)[:, None])).T
            
    with numpyro.plate("N_lineage", N_lineage):
        I0 = numpyro.sample("I0", dist.Uniform(0.0, 10000.0))

    # rho is weekend-seasonal
    with numpyro.plate("rho_parms", 7):
        rho = numpyro.sample("rho", dist.Beta(5., 5.))
    rho_vec = reporting_to_vec(rho, T)

    # Simulate lineage prevalences
    I_prev = v_fs_I(I0, R, gen_rev, delays, seed_L)

    # Compute expected cases
    total_prev = I_prev.sum(axis=1)
    EC = total_prev[seed_L:] * rho_vec

    # sample cases using Poisson distribution
    numpyro.sample("cases",
                   dist.Poisson(rate=EC),
                   obs=cases)

    # Compute frequency
    freq = jnp.divide(I_prev, total_prev[:, None])[seed_L:, :]
    numpyro.sample("Y",
                   dist.Multinomial(total_count=N, probs=freq),
                   obs=Y)


def model_lineage_pred(gen_rev, delays, X, seed_L, N_lineage,
                       N=None, Y=None, cases=None):
    T = X.shape[0]
    k = X.shape[1]

    # R is a linear regression
    with numpyro.plate("k", k):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        
    # With lineage-specific advantages
    with numpyro.plate("N_lineage_m1", N_lineage-1):
        v = numpyro.sample("v", dist.Normal(0.0, 1.0))
        
    ga = numpyro.deterministic("ga", jnp.exp(jnp.append(0.0, v)))
    R = numpyro.deterministic("R",
                              jnp.exp((X @ beta + jnp.append(0.0, v)[:, None])).T)

    with numpyro.plate("N_lineage", N_lineage):
        I0 = numpyro.sample("I0", dist.Uniform(0.0, 10000.0))

    # rho is weekend-seasonal
    with numpyro.plate("rho_parms", 7):
        rho = numpyro.sample("rho", dist.Beta(5., 5.))
    rho_vec = reporting_to_vec(rho, T)
    mean_rho = jnp.mean(rho_vec)

    # Simulate lineage prevalences
    I_prev = v_fs_I(I0, R, gen_rev, delays, seed_L)
    I_smooth = mean_rho * I_prev[seed_L:, :]
    I_smooth = numpyro.deterministic("I_smooth", I_smooth)

    # Compute expected cases
    total_prev = I_prev.sum(axis=1)
    total_smooth_prev = numpyro.deterministic("total_smooth_prev", mean_rho * I_prev.sum(axis=1)[seed_L:])
    EC = numpyro.deterministic("EC", total_prev[seed_L:].T * rho_vec)

    # sample cases using Poisson distribution
    numpyro.sample("cases",
                   dist.Poisson(rate=EC))

    freq = numpyro.deterministic("freq", jnp.divide(I_prev, total_prev[:, None])[seed_L:, :])
    numpyro.sample("Y", dist.Multinomial(total_count=N, probs=freq))
