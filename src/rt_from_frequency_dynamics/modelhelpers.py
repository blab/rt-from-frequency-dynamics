# Discretize pmf
import jax.numpy as jnp
import numpy as np
from patsy import dmatrix, bs
from scipy.stats import gamma, lognorm


# https://github.com/lo-hfk/epyestim/blob/ca2ca928b744f324dade248c24a40872b69a5222/epyestim/distributions.py
def continuous_dist_to_pmf(dist):
    max_dist = dist.ppf(0.9999)
    xs = jnp.linspace(0.5, max_dist + 0.5, int(max_dist + 1))
    pmf = jnp.diff(dist.cdf(xs), prepend=0)
    return (pmf / pmf.sum())


def discretise_gamma(mn, std):
    a = (mn / std)**2
    scale = std ** 2 / mn
    return continuous_dist_to_pmf(gamma(a=a, scale=scale))


def discretise_lognorm(mn, std):
    gam = 1 + std**2 / mn**2
    LN = lognorm(scale=mn/np.sqrt(gam), s=np.sqrt(np.log(gam)))
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


def break_points_to_mat(break_points, L):
    t = jnp.arange(0, L)
    X = []
    X.append(t > -1)
    for i in range(len(break_points)-1):
        X.append((t > break_points[i]) & (t <= break_points[i+1]))
    return jnp.stack(X, axis=0)


def make_breakpoint_matrix(cases, k):
    T = len(cases)
    break_points = jnp.linspace(0, T, k + 1)
    X = break_points_to_mat(break_points, T)
    return X.T


def make_breakpoint_splines(cases, k):
    T = len(cases)
    t = jnp.linspace(0, 1., T)
    X = dmatrix(f"bs(t, df={k}, degree=4, include_intercept=True)-1", {"t": t})
    return jnp.array(X)
