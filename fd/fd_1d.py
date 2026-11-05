import jax.numpy as jnp
from pint_galerkin.objects import StochasticOCP
import jax


def setup_space_grid(L, N_x):
    x, dx = jnp.linspace(-L, L, N_x, retstep=True)
    return x, dx


def final_values(ocp: StochasticOCP, grid_points: jnp.ndarray):
    p_x_tf = jax.vmap(lambda x: jnp.exp(-ocp.l_f(x) / ocp.lambda_param))(grid_points)
    return p_x_tf


def build_fd_matrix(ocp: StochasticOCP, grid_points: jnp.ndarray, h: float):
    l_x = jax.vmap(ocp.l)(grid_points)
    a_x = jax.vmap(ocp.a)(grid_points)
    S_x = jax.vmap(ocp.S)(grid_points)
    d = -l_x / ocp.lambda_param - S_x / (h**2)
    dl = -a_x[1:] / (2 * h) + 0.5 * S_x[1:] / (h**2)
    du = a_x[:-1] / (2 * h) + 0.5 * S_x[:-1] / (h**2)
    return d, dl, du


def reconstruct_value_function(ocp: StochasticOCP, p_x_t: jnp.ndarray) -> jnp.ndarray:
    V = -ocp.lambda_param * jnp.log(p_x_t)
    return V


def simulate_closed_loop(
    ocp: StochasticOCP, V_matrix, x0, grid_points, dx, dt, N_t, key
):
    Q = ocp.lambda_param * ocp.R_inv
    key, subkey = jax.random.split(key)
    dw = jax.random.normal(subkey, shape=(N_t,)) * jnp.sqrt(Q)
    sqrt_dt = jnp.sqrt(dt)
    grad_V_matrix = jax.vmap(lambda v: jnp.gradient(v, dx))(V_matrix)

    def euler_maruyama_step(carry, dw_t):
        x_t, t_idx = carry
        a_x = ocp.a(x_t)
        b_x = ocp.b(x_t)
        grad_V_t = grad_V_matrix[t_idx]
        grad_V_x = jnp.interp(x_t, grid_points, grad_V_t)
        u_t = -ocp.R_inv * b_x * grad_V_x
        x_tp1 = x_t + (a_x + b_x * u_t) * dt + sqrt_dt * b_x * dw_t
        t_idx = t_idx + 1
        return (x_tp1, t_idx), (x_tp1, u_t)

    _, (x_traj, u_traj) = jax.lax.scan(euler_maruyama_step, (x0, 0), xs=dw, length=N_t)
    x_traj = jnp.hstack((x0, x_traj))
    return x_traj, u_traj


def backward_seq_implicit_euler(
    d: jnp.ndarray,
    dl: jnp.ndarray,
    du: jnp.ndarray,
    p_tf: jnp.ndarray,
    t_f: float,
    t_0: float,
    N_t: int,
):
    dt = (t_f - t_0) / N_t
    main_diagonal = 1.0 - dt * d
    lower_diagonal = -dt * dl
    upper_diagonal = -dt * du
    lower_diagonal = jnp.pad(lower_diagonal, (1, 0))
    upper_diagonal = jnp.pad(upper_diagonal, (0, 1))

    def step(p, _):
        p = jax.lax.linalg.tridiagonal_solve(
            lower_diagonal, main_diagonal, upper_diagonal, p
        )
        return p, p

    _, p_t = jax.lax.scan(step, p_tf[:, None], xs=None, length=N_t)
    p_t = jnp.squeeze(p_t, axis=-1)
    p_t = jnp.vstack((p_t[::-1], p_tf))
    t_grid = jnp.linspace(t_0, t_f, N_t + 1)

    return p_t, t_grid, dt
