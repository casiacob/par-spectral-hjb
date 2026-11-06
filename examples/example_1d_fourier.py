import jax.numpy as jnp

from pint_galerkin.nonlinear_model_1d import ocp_nonlin_1d
from jax import config
from pint_galerkin.fourier_1d import (
    setup_grids,
    project_final_cost,
    spectral_matrix,
    construct_value_function,
    simulate_closed_loop_spectral,
)
import jax
import matplotlib.pyplot as plt

from pint_galerkin.ode_solvers import seq_exp_integrate_time_invariant

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)


L = 1.5
N_freq = 30
N_x = 400
t_f = 4.0
t_0 = 0.0
N_t = 400

x_grid, w_grid = setup_grids(L, N_x, N_freq)

c_f, p_x, phi_sin = project_final_cost(ocp_nonlin_1d, x_grid, w_grid, L)
Gamma = spectral_matrix(ocp_nonlin_1d, phi_sin, x_grid, w_grid, L)
c_t_seq, t_eval, dt = seq_exp_integrate_time_invariant(Gamma, c_f, t_f, t_0, N_t)
V_x_t, p_x_t = construct_value_function(ocp_nonlin_1d, c_t_seq, phi_sin)

n_paths = 100
keys = jax.random.split(jax.random.PRNGKey(0), n_paths)
x0 = 1.0
simulate_one = lambda key: simulate_closed_loop_spectral(
    ocp_nonlin_1d, c_t_seq, w_grid, x0, dt, N_t, L, key
)

trajectories, controls = jax.jit(jax.vmap(simulate_one))(keys)

mean_traj = jnp.mean(trajectories, axis=0)
mean_controls = jnp.mean(controls, axis=0)

plt.plot(t_eval, mean_traj)
plt.show()

plt.plot(t_eval[:-1], mean_controls)
plt.show()