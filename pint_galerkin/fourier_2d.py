import jax.numpy as jnp
import jax
from .objects import StochasticOCP


def setup_grids_2d(L, N_x, N_w):
    Lx, Ly = L
    Nx, Ny = N_x
    Nwx, Nwy = N_w

    x_grid = jnp.linspace(-Lx, Lx, Nx, endpoint=True)
    y_grid = jnp.linspace(-Ly, Ly, Ny, endpoint=True)
    X, Y = jnp.meshgrid(x_grid, y_grid, indexing="ij")
    X = X.flatten()
    Y = Y.flatten()
    XY = jnp.stack((X, Y), axis=1)

    m = jnp.arange(1, Nwx)
    n = jnp.arange(1, Nwy)
    w_m = jnp.pi * m / (2 * Lx)
    w_n = jnp.pi * n / (2 * Ly)
    W_m, W_n = jnp.meshgrid(w_m, w_n, indexing="ij")
    W_m = W_m.flatten()
    W_n = W_n.flatten()

    return x_grid, y_grid, X, Y, XY, W_m, W_n


def project_final_cost_2d(
    ocp: StochasticOCP, x_grid, y_grid, X, Y, XY, W_m, W_n, N_x, L
):
    Lx, Ly = L
    Nx, Ny = N_x

    p_xy = jax.vmap(lambda x: jnp.exp(-ocp.l_f(x) / ocp.lambda_param))(XY)
    p_xy = p_xy.reshape(Nx, Ny)

    phi_m = jnp.sin(jnp.outer(W_m, X + Lx)) / jnp.sqrt(Lx)
    phi_n = jnp.sin(jnp.outer(W_n, Y + Ly)) / jnp.sqrt(Ly)
    phi = phi_m * phi_n

    phi = phi.reshape(-1, Nx, Ny)

    integrand = phi * p_xy

    c_f = jax.scipy.integrate.trapezoid(
        jax.scipy.integrate.trapezoid(integrand, x=y_grid, axis=2), x=x_grid, axis=1
    )

    return c_f, p_xy, phi


def spectral_matrix_2d(
    ocp: StochasticOCP, phi, x_grid, y_grid, X, Y, XY, Freq_m, Freq_n, N_x, L
):
    Lx, Ly = L
    Nx, Ny = N_x

    a_xy = jax.vmap(ocp.a)(XY).reshape(Nx, Ny, 2)
    l_xy = jax.vmap(ocp.l)(XY).reshape(Nx, Ny)
    S_xy = jax.vmap(ocp.S)(XY).reshape(Nx, Ny, 2, 2)

    ddphi_dx2 = -(Freq_m[:, None, None] ** 2) * phi
    ddphi_dy2 = -(Freq_n[:, None, None] ** 2) * phi

    arg_x = jnp.outer(Freq_m, X + Lx)
    arg_y = jnp.outer(Freq_n, Y + Ly)

    sin_x = jnp.sin(arg_x) / jnp.sqrt(Lx)
    sin_y = jnp.sin(arg_y) / jnp.sqrt(Ly)
    cos_x = jnp.cos(arg_x) / jnp.sqrt(Lx)
    cos_y = jnp.cos(arg_y) / jnp.sqrt(Ly)

    dphi_dx = (Freq_m[:, None] * cos_x * sin_y).reshape(-1, Nx, Ny)
    dphi_dy = (Freq_n[:, None] * sin_x * cos_y).reshape(-1, Nx, Ny)
    ddphi_dxdy = (Freq_m[:, None] * Freq_n[:, None] * cos_x * cos_y).reshape(-1, Nx, Ny)

    term_l = -(l_xy / ocp.lambda_param)[None, :, :] * phi

    term_a = a_xy[..., 0][None, :, :] * dphi_dx + a_xy[..., 1][None, :, :] * dphi_dy

    term_S = 0.5 * (
        S_xy[..., 0, 0][None, :, :] * ddphi_dx2
        + S_xy[..., 1, 1][None, :, :] * ddphi_dy2
        + 2.0 * S_xy[..., 0, 1][None, :, :] * ddphi_dxdy
    )

    A_phi = term_l + term_a + term_S

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    weights_x = jnp.full_like(x_grid, dx).at[0].set(dx / 2.0).at[-1].set(dx / 2.0)
    weights_y = jnp.full_like(y_grid, dy).at[0].set(dy / 2.0).at[-1].set(dy / 2.0)
    weights_2d = weights_x[:, None] * weights_y[None, :]

    phi_flat = phi.reshape(phi.shape[0], -1)
    A_phi_weighted_flat = (A_phi * weights_2d[None, :, :]).reshape(A_phi.shape[0], -1)

    F = phi_flat @ A_phi_weighted_flat.T

    return F


def construct_value_function_2d(ocp: StochasticOCP, c_t, phi):
    p_xt = jnp.einsum("tb, bxy -> txy", c_t, phi)
    V_xt = -ocp.lambda_param * jnp.log(p_xt)
    return V_xt, p_xt


def closed_loop_euler_maruyama_2d(
    ocp,
    c,
    W_m,
    W_n,
    x0,
    dt,
    L,
    key,
):
    L_theta, L_omega = L
    N_t = c.shape[0] - 1
    noise_scale = jnp.sqrt(ocp.lambda_param * ocp.R_inv)
    dW = jax.random.normal(key, shape=(N_t,)) * jnp.sqrt(dt)

    def euler_maruyama_step(x_t, inputs):
        dW_t, c_t = inputs
        theta, omega = x_t

        arg_theta = W_m * (theta + L_theta)
        arg_omega = W_n * (omega + L_omega)

        sin_theta = jnp.sin(arg_theta) / jnp.sqrt(L_theta)
        sin_omega = jnp.sin(arg_omega) / jnp.sqrt(L_omega)
        cos_theta = jnp.cos(arg_theta) / jnp.sqrt(L_theta)
        cos_omega = jnp.cos(arg_omega) / jnp.sqrt(L_omega)

        phi = sin_theta * sin_omega
        dphi_dtheta = W_m * cos_theta * sin_omega
        dphi_domega = W_n * sin_theta * cos_omega

        p_x = jnp.dot(c_t, phi)
        grad_p = jnp.array(
            [
                jnp.dot(c_t, dphi_dtheta),
                jnp.dot(c_t, dphi_domega),
            ]
        )

        b_x = ocp.b(x_t)
        u_t = ocp.lambda_param * ocp.R_inv * jnp.dot(b_x, grad_p) / p_x
        # u_t = jnp.zeros(2)

        x_tp1 = x_t + (ocp.a(x_t) + b_x * u_t) * dt + b_x * noise_scale * dW_t
        return x_tp1, (x_tp1, u_t)

    _, (x_steps, u_traj) = jax.lax.scan(
        euler_maruyama_step,
        x0,
        (dW, c[:-1]),
    )

    x_traj = jnp.concatenate((x0[None, :], x_steps), axis=0)
    return x_traj, u_traj
