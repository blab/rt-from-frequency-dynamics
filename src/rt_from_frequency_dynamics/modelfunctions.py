from functools import partial
from jax import lax, jit, vmap
import jax.numpy as jnp


@partial(jit, static_argnums=3)
def get_infections(I0, R, g_rev, seed_L):
    max_age = len(g_rev)

    I0_vec = jnp.repeat(I0, seed_L)

    @jit
    def _scan_infections(infections, R):
        curr_I = R * jnp.dot(infections, g_rev)  # New I
        return jnp.append(infections[-(max_age - 1) :], curr_I), curr_I

    _, infections = lax.scan(
        _scan_infections, init=jnp.pad(I0_vec, (max_age - seed_L, 0)), xs=R
    )
    return jnp.append(I0_vec, infections)


@partial(jit, static_argnums=3)
def get_infections_intros(m, R, g_rev, seed_L):
    max_age = len(g_rev)

    @jit
    def _scan_infections(infections, xs):
        R, im = xs
        curr_I = R * jnp.dot(infections, g_rev) + im  # New I
        return jnp.append(infections[-(max_age - 1):], curr_I), curr_I

    _, infections = lax.scan(
        _scan_infections,
        init=jnp.zeros(max_age),
        xs=(jnp.pad(R, (seed_L, 0), constant_values=1.0), m),
    )
    return infections


@jit
def apply_delay(infections, delay):
    I_padded = jnp.pad(infections, (len(delay) - 1, 0), constant_values=0)
    return jnp.convolve(I_padded, delay, mode="valid")


def _apply_delays(infections, delay):
    out = apply_delay(infections, delay)
    return out, out


# @partial(jit, static_argnums=4)
# def forward_simulate_I(I0, R, gen_rev, delays, seed_L):
#     infections = get_infections(I0, R, gen_rev, seed_L)
#     infections, _ = lax.scan(_apply_delays, init=infections, xs=delays)
#     return infections


@partial(jit, static_argnums=4)
def forward_simulate_I(m, R, gen_rev, delays, seed_L):
    infections = get_infections_intros(m, R, gen_rev, seed_L)
    infections, _ = lax.scan(_apply_delays, init=infections, xs=delays)
    return infections


v_fs_I = jit(
    vmap(forward_simulate_I, in_axes=(-1, -1, None, None, None), out_axes=-1),
    static_argnums=4,
)


@partial(jit, static_argnums=1)
def reporting_to_vec(rho, L):
    return jnp.pad(rho, (0, L - len(rho)), mode="wrap")


@partial(jit, static_argnums=5)
def forward_simulate_EC(I0, R, rho, gen_rev, delays, seed_L):
    infections = get_infections(I0, R, gen_rev, seed_L)
    infections, _I = lax.scan(_apply_delays, init=infections, xs=delays)
    return rho * infections[seed_L:]
