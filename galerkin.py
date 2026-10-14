import jax
import jax.numpy as jnp
from objects import StochasticOCP


def setup_grids(
    space_bounds: float,
    space_grid_size: int,
    final_time: float,
    start_time: float,
    time_grid_size: int,
):
    # space and frequencies grid
    x_grid = jnp.linspace(
        -space_bounds, space_bounds, space_grid_size, endpoint=False
    ).reshape(-1, 1)
    volume = 2 * space_bounds
    dx = volume / space_grid_size
    w_grid = (jnp.fft.fftfreq(space_grid_size, d=dx) * 2 * jnp.pi).reshape(-1, 1)

    # temporal grid
    t_grid = jnp.linspace(start_time, final_time, time_grid_size)
    dt = (final_time - start_time) / (time_grid_size - 1)
    return x_grid, w_grid, dx, volume, t_grid, dt


def project_final_cost(
    ocp: StochasticOCP,
    space_grid: jnp.ndarray,
    frequency_grid: jnp.ndarray,
    dx: float,
    volume: float,
) -> jnp.ndarray:

    # p(x, t_f)
    p_tf_vals = jax.vmap(lambda x: jnp.exp(-ocp.l_f(x) / ocp.lambda_param))(
        space_grid
    ).flatten()

    x_0 = space_grid[0, 0]
    phase_shift = jnp.exp(-1j * frequency_grid.flatten() * x_0)

    integral_approx = jnp.fft.fft(p_tf_vals) * dx * phase_shift

    c_tf = integral_approx / volume
    return c_tf


def galerkin_matrix(
    ocp: StochasticOCP, space_grid: jnp.ndarray, frequency_grid: jnp.ndarray, dx: float
) -> jnp.ndarray:
    x_0 = space_grid[0, 0]
    phase_shift = jnp.exp(-1j * frequency_grid.flatten() * x_0)

    def column_m(w_m):
        def fpk_operator(x):
            term1 = -ocp.l(x) / ocp.lambda_param  # -l(x)/lambda
            term2 = 1j * jnp.dot(w_m, ocp.a(x))  # i w_m^T a(x)
            term3 = -0.5 * jnp.dot(w_m, jnp.dot(ocp.S(x), w_m))  # 0.5 w_m^T S(x) w_m
            F_xw_m = term1 + term2 + term3

            return F_xw_m * jnp.exp(1j * jnp.dot(w_m, x))

        A_phi_m_vals = jax.vmap(fpk_operator)(space_grid).flatten()  # A phi_m
        col_m = jnp.fft.fft(A_phi_m_vals) * dx * phase_shift
        return col_m

    Gamma_T = jax.vmap(column_m)(frequency_grid)
    return Gamma_T.T


def solve_ode(
    Gamma: jnp.ndarray,
    c_tf: jnp.ndarray,
    volume: float,
    dt: float,
    time_grid_size: int,
) -> jnp.ndarray:
    G = Gamma / volume

    # def rk4_step(c: jnp.ndarray, h: float) -> jnp.ndarray:
    #     k1 = jnp.dot(G, c)
    #     k2 = jnp.dot(G, c + 0.5 * h * k1)
    #     k3 = jnp.dot(G, c + 0.5 * h * k2)
    #     k4 = jnp.dot(G, c + h * k3)
    #     return c + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def euler_step(c: jnp.ndarray, h: float) -> jnp.ndarray:
        return c + h * jnp.dot(G, c)

    def body(c_t, _):
        c_tm1 = euler_step(c_t, dt)
        return c_tm1, c_t

    _, c_trajectory = jax.lax.scan(
        body, c_tf, xs=None, length=time_grid_size, reverse=True
    )

    return c_trajectory


def construct_value_functions(
    ocp: StochasticOCP,
    coefficients: jnp.ndarray,
    space_grid: jnp.ndarray,
    frequency_grid: jnp.ndarray,
) -> jnp.ndarray:
    # V(x, t) = -lambda log p(x, t)
    phi = jnp.exp(1j * (space_grid @ frequency_grid.T))
    p_xt = coefficients @ phi.T  # p = c(t)^T phi(x)
    V_xt = -ocp.lambda_param * jnp.log(jnp.real(p_xt))
    return V_xt


def construct_optimal_control(
    ocp: StochasticOCP,
    coefficients: jnp.ndarray,
    space_grid: jnp.ndarray,
    frequency_grid: jnp.ndarray,
) -> jnp.ndarray:
    phi = jnp.exp(1j * (space_grid @ frequency_grid.T))
    grad_phi = 1j * frequency_grid.T * phi
    p_xt = jnp.real(coefficients @ phi.T)
    grad_p_xt = jnp.real(coefficients @ grad_phi.T)
    grad_V_xt = -ocp.lambda_param * grad_p_xt / p_xt
    b_x = jax.vmap(ocp.b)(space_grid).squeeze()
    u_star = -ocp.R_inv[0, 0] * b_x[None, :] * grad_V_xt
    return u_star
