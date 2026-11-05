import jax.numpy as jnp
import jax
from pint_galerkin.fourier_2d import (
    setup_grids_2d,
    project_final_cost_2d,
    spectral_matrix_2d,
    construct_value_function_2d,
    closed_loop_euler_maruyama_2d,
)

from pint_galerkin.ode_solvers import (
    par_exp_integrate_time_invariant,
    seq_exp_integrate_time_invariant,
)
import matplotlib.pyplot as plt
from jax import config
from models.vdp_model import ocp_vdp

config.update("jax_enable_x64", True)


L = (3.0, 3.0)
x0 = jnp.array([1.0, 0.0])

N_x = (250, 250)
N_w = (11, 11)
t_0, t_f = 0.0, 8.0
N_t = 400


x_grid, y_grid, X, Y, XY, W_m, W_n = setup_grids_2d(L, N_x, N_w)
c_f, p_x, phi = project_final_cost_2d(
    ocp_vdp, x_grid, y_grid, X, Y, XY, W_m, W_n, N_x, L
)
Gamma = spectral_matrix_2d(ocp_vdp, phi, x_grid, y_grid, X, Y, XY, W_m, W_n, N_x, L)
c_t_seq, t_eval, dt = seq_exp_integrate_time_invariant(Gamma, c_f, t_f, t_0, N_t)
c_t_par, _, _ = par_exp_integrate_time_invariant(Gamma, c_f, t_f, t_0, N_t)
print(jnp.allclose(c_t_seq, c_t_par))
V_xt, p_xt = construct_value_function_2d(ocp_vdp, c_t_seq, phi)
approx_p_x = jnp.einsum("w,wxy->xy", c_f, phi)

n_paths = 100
keys = jax.random.split(jax.random.PRNGKey(0), n_paths)

simulate_one = lambda key: closed_loop_euler_maruyama_2d(
    ocp_vdp, c_t_seq, W_m, W_n, x0, dt, L, key
)

trajectories, controls = jax.jit(jax.vmap(simulate_one))(keys)

mean_traj = jnp.mean(trajectories, axis=0)
mean_controls = jnp.mean(controls, axis=0)

plt.plot(t_eval, mean_traj)
plt.show()

plt.plot(t_eval[:-1], mean_controls)
plt.show()
