import jax.numpy as jnp
import jax.scipy as jsp
import jax
from objects import StochasticOCP


def setup_grids(L, N_x, N_w):
    x_grid = jnp.linspace(-L, L, N_x)
    m = jnp.arange(1, N_w)
    w_grid = jnp.pi * m / (2 * L)
    return x_grid, w_grid


def project_final_cost(ocp, x_grid, w_grid, L):
    p_x = jnp.exp(-ocp.l_f(x_grid) / ocp.lambda_param)
    phi = jnp.sin(jnp.outer(w_grid, x_grid + L)) / jnp.sqrt(L)
    gamma = jax.scipy.integrate.trapezoid(phi * p_x, x=x_grid, axis=1)
    c_f = gamma
    return c_f, p_x, phi


def spectral_matrix(ocp: StochasticOCP, phi, x_grid, w_grid, L):
    dphi = w_grid[:, None] / jnp.sqrt(L) * jnp.cos(jnp.outer(w_grid, x_grid + L))
    ddphi = -(w_grid[:, None] ** 2) * phi

    a_x = ocp.a(x_grid)
    l_x = ocp.l(x_grid)
    S_x = ocp.S(x_grid)

    A_phi = -(l_x / ocp.lambda_param) * phi + a_x * dphi + 0.5 * S_x * ddphi

    integrand = phi[:, None, :] * A_phi[None, :, :]
    F = jax.scipy.integrate.trapezoid(integrand, x=x_grid, axis=2)
    return F


def construct_value_function(ocp: StochasticOCP, c_t, phi):
    p_xt = jax.vmap(lambda c: jnp.dot(c, phi))(c_t)
    V_xt = -ocp.lambda_param * jnp.log(p_xt)
    return V_xt, p_xt


def simulate_closed_loop_spectral(ocp, c, w_grid, x0, dt, N_t, L, key):
    key, subkey = jax.random.split(key)
    Q = ocp.lambda_param * ocp.R_inv
    dw = jax.random.normal(subkey, shape=(N_t,)) * jnp.sqrt(Q)
    sqrt_dt = jnp.sqrt(dt)

    def euler_maruyama_step(carry, inp):
        x_t = carry
        dw_t, c_t = inp

        phi = jnp.sin(w_grid * (x_t + L)) / jnp.sqrt(L)

        dphi = w_grid * jnp.cos(w_grid * (x_t + L)) / jnp.sqrt(L)

        p_x = jnp.dot(c_t, phi)
        dp_x = jnp.dot(c_t, dphi)

        a_x = ocp.a(x_t)
        b_x = ocp.b(x_t)
        u_t = (ocp.lambda_param / p_x) * ocp.R_inv * b_x * dp_x
        x_tp1 = x_t + (a_x + b_x * u_t) * dt + sqrt_dt * b_x * dw_t
        return x_tp1, (x_tp1, u_t)

    _, (x_traj, u_traj) = jax.lax.scan(
        euler_maruyama_step, init=x0, xs=(dw, c[:-1]), length=N_t
    )
    x_traj = jnp.hstack((x0, x_traj))
    return x_traj, u_traj
