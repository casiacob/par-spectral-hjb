import jax.numpy as jnp
import jax.scipy as jsp
import jax
from pint_galerkin.objects import StochasticOCP


def setup_space_grid(ub, lb, N):
    x, dx = jnp.linspace(lb[0], ub[0], N, retstep=True)
    y, dy = jnp.linspace(lb[1], ub[1], N, retstep=True)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    nodes = jnp.hstack((xx.flatten().reshape(-1, 1), yy.flatten().reshape(-1, 1)))
    return xx, yy, dx, dy, nodes


def final_values(ocp: StochasticOCP, nodes: jnp.ndarray):
    p_x_tf = jax.vmap(lambda x: jnp.exp(-ocp.l_f(x) / ocp.lambda_param))(nodes)
    return p_x_tf


def build_fd_matrix(ocp: StochasticOCP, nodes: jnp.ndarray, N, dx, dy):
    D = jnp.diag(-jnp.ones(N - 1), k=-1) + jnp.diag(jnp.ones(N - 1), k=1)
    DD = (
        jnp.diag(jnp.ones(N - 1), k=-1)
        - 2 * jnp.eye(N)
        + jnp.diag(jnp.ones(N - 1), k=1)
    )

    D1 = D / (2 * dx)
    D2 = D / (2 * dy)
    D11 = DD / (dx**2)
    D22 = DD / (dy**2)

    I1 = jnp.eye(N)
    I2 = jnp.eye(N)

    Dx = jnp.kron(D1, I2)
    Dxx = jnp.kron(D11, I2)

    Dy = jnp.kron(I1, D2)
    Dyy = jnp.kron(I1, D22)

    Dxy = jnp.kron(D1, D2)

    l_x = jax.vmap(ocp.l)(nodes)
    a_x = jax.vmap(ocp.a)(nodes)
    S_x = jax.vmap(ocp.S)(nodes)

    F = -1 / ocp.lambda_param * jnp.diag(l_x)
    F = F + jnp.diag(a_x[:, 0]) @ Dx
    F = F + jnp.diag(a_x[:, 1]) @ Dy
    F = F + 0.5 * jnp.diag(S_x[:, 0, 0]) @ Dxx
    F = F + 0.5 * jnp.diag(S_x[:, 1, 1]) @ Dyy
    F = F + jnp.diag(S_x[:, 0, 1]) @ Dxy

    mask2d = jnp.ones((N, N))
    mask2d = mask2d.at[0, :].set(0).at[-1, :].set(0).at[:, 0].set(0).at[:, -1].set(0)
    mask_flat = mask2d.flatten()
    F = mask_flat[:, None] * F
    return F


def backward_seq_implicit_euler(
    F: jnp.ndarray, p_tf: jnp.ndarray, t_f: float, t_0: float, N_t: int
):
    dt = (t_f - t_0) / N_t

    def step(p, _):
        p = jsp.linalg.solve(jnp.eye(F.shape[0]) - dt * F, p)
        return p, p

    _, p_t = jax.lax.scan(step, p_tf[:, None], xs=None, length=N_t)
    p_t = jnp.squeeze(p_t, axis=-1)
    p_t = jnp.vstack((p_t[::-1], p_tf))
    t_grid = jnp.linspace(t_0, t_f, N_t + 1)

    return p_t, t_grid, dt


def reconstruct_value_function(ocp: StochasticOCP, p_x_t: jnp.ndarray) -> jnp.ndarray:
    V = -ocp.lambda_param * jnp.log(p_x_t)
    return V


def simulate_closed_loop(
    ocp: StochasticOCP, V_matrix, x0, xx, yy, dx, dy, dt, N_t, key
):
    N = xx.shape[0]

    # 1. Ensure Q is a pure scalar float
    Q_scalar = jnp.squeeze(ocp.lambda_param * ocp.R_inv)

    key, subkey = jax.random.split(key)
    # Generate 1D scalar noise: shape is (N_t,)
    dw = jax.random.normal(subkey, shape=(N_t,)) * jnp.sqrt(Q_scalar)
    sqrt_dt = jnp.sqrt(dt)

    V_2d = V_matrix.reshape((N_t + 1, N, N))
    grad_V_x, grad_V_y = jnp.gradient(V_2d, dx, dy, axis=(1, 2))

    def euler_maruyama_step(carry, dw_t):
        x_t, t_idx = carry

        # 2. Keep spatial evaluations as strict 1D vectors (shape: (2,))
        a_x = jnp.array(ocp.a(x_t)).flatten()
        b_x = jnp.array(ocp.b(x_t)).flatten()

        idx_x = (x_t[0] - xx[0, 0]) / dx
        idx_y = (x_t[1] - yy[0, 0]) / dy
        coords = jnp.array([[idx_x], [idx_y]])

        g_x = jax.scipy.ndimage.map_coordinates(
            grad_V_x[t_idx], coords, order=1, mode="nearest"
        )[0]
        g_y = jax.scipy.ndimage.map_coordinates(
            grad_V_y[t_idx], coords, order=1, mode="nearest"
        )[0]
        grad_V_xt = jnp.array([g_x, g_y])

        # 3. Compute control as a pure scalar float using a dot product
        R_inv_scalar = jnp.squeeze(ocp.R_inv)
        u_t = -R_inv_scalar * jnp.dot(b_x, grad_V_xt)

        # 4. Standard scalar multiplication prevents shape dimension explosions
        # State stays strictly (2,) + (2,)*float + (2,)*float
        x_tp1 = x_t + (a_x + b_x * u_t) * dt + sqrt_dt * (b_x * dw_t)

        t_idx = t_idx + 1
        return (x_tp1, t_idx), (x_tp1, u_t)

    # xs=dw passes a single float `dw_t` to each step
    _, (x_traj, u_traj) = jax.lax.scan(euler_maruyama_step, (x0, 0), xs=dw, length=N_t)
    x_traj = jnp.vstack((x0, x_traj))
    return x_traj, u_traj
