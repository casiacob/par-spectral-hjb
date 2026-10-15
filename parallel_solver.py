import jax.numpy as jnp
from jax import lax, vmap


def combine_fc(elem1, elem2):
    Fij, cij = elem1
    Fjk, cjk = elem2

    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_reverse_scan(elems):
    return lax.associative_scan(vmap(combine_fc), elems, reverse=True)


def par_init(F: jnp.ndarray, c: jnp.ndarray, x_f: jnp.ndarray):
    tF = jnp.vstack((F[:-1], jnp.zeros((1, F[-1].shape[0], F[-1].shape[1]))))
    tc = jnp.vstack((c[:-1], F[-1] @ x_f + c[-1]))
    elems = (tF, tc)
    return elems


def solve_par_ode(
    Gamma: jnp.ndarray,
    c_tf: jnp.ndarray,
    volume: float,
    dt: float,
    time_grid_size: int,
):
    G = Gamma / volume
    F = jnp.kron(jnp.ones((time_grid_size, 1, 1)), jnp.eye(G.shape[0]) + dt * G)
    c = jnp.zeros((time_grid_size, G.shape[0]))
    elems = par_init(F, c, c_tf)
    elems = par_reverse_scan(elems)
    return elems[1]
