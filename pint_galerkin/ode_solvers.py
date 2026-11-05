import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial


def combine_elements(F_jk, F_ij):
    return F_ij @ F_jk


def par_scan(elements):
    return jax.lax.associative_scan(
        combine_elements,
        elements,
        reverse=True,
    )


def par_init(F, N_t):
    return jnp.broadcast_to(F, (N_t,) + F.shape)


def par_extract(F_t, c_M):
    c_t = jax.vmap(lambda A: A @ c_M)(F_t)
    return jnp.vstack((c_t, c_M))


def par_exp_integrate_time_invariant(
    Gamma: jnp.ndarray,
    c_tf: jnp.ndarray,
    t_f: float,
    t_0: float,
    N_t: int,
):
    dt = (t_f - t_0) / N_t
    F = jsp.linalg.expm(dt * Gamma)

    elems = par_init(F, N_t)
    F_t = par_scan(elems)
    c_t = par_extract(F_t, c_tf)

    t_grid = jnp.linspace(t_0, t_f, N_t + 1)
    return c_t, t_grid, dt


def seq_exp_integrate_time_invariant(
    Gamma: jnp.ndarray,
    c_tf: jnp.ndarray,
    t_f: float,
    t_0: float,
    N_t: int,
):
    dt = (t_f - t_0) / N_t
    F = jsp.linalg.expm(dt * Gamma)

    def step(c, _):
        c = F @ c
        return c, c

    _, c_t = jax.lax.scan(step, c_tf[:, None], xs=None, length=N_t)
    c_t = jnp.squeeze(c_t, axis=-1)
    c_t = jnp.vstack((c_t[::-1], c_tf))
    t_grid = jnp.linspace(t_0, t_f, N_t + 1)

    return c_t, t_grid, dt
