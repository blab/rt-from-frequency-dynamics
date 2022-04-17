import jax.numpy as jnp
import numpy as np
from scipy.stats import gamma, lognorm


# https://github.com/lo-hfk/epyestim/blob/ca2ca928b744f324dade248c24a40872b69a5222/epyestim/distributions.py
def continuous_dist_to_pmf(dist):
    max_dist = dist.ppf(0.9999)
    xs = jnp.linspace(0.5, max_dist + 0.5, int(max_dist + 1))
    pmf = jnp.diff(dist.cdf(xs), prepend=0)
    return pmf / pmf.sum()


def discretise_gamma(mn, std):
    a = (mn / std) ** 2
    scale = std**2 / mn
    return continuous_dist_to_pmf(gamma(a=a, scale=scale))


def discretise_lognorm(mn, std):
    gam = 1 + std**2 / mn**2
    LN = lognorm(scale=mn / np.sqrt(gam), s=np.sqrt(np.log(gam)))
    return continuous_dist_to_pmf(LN)


def pad_delays(delays):
    lens = -jnp.array([len(delay) for delay in delays])
    lens -= min(lens)
    for i in range(len(delays)):
        delays[i] = jnp.pad(delays[i], (0, lens[i]))
    return jnp.stack(delays)  # Return as matrix


def get_standard_delays():
    gen = discretise_gamma(mn=5.2, std=1.72)
    delays = [discretise_lognorm(mn=6.9, std=2.0)]
    delays = pad_delays(delays)
    return gen, delays


def is_obs_idx(v):
    return jnp.where(jnp.isnan(v), jnp.zeros_like(v), jnp.ones_like(v))


def pad_to_obs(v, obs_idx, eps=1e-12):
    return v * obs_idx + (1 - obs_idx) * eps
