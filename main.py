import jax.numpy as jnp
from jax import config
from galerkin import (
    setup_grids,
    project_final_cost,
    galerkin_matrix,
    solve_ode,
    construct_value_functions,
    construct_optimal_control,
)
from parallel_solver import solve_par_ode
from objects import StochasticOCP
import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

def dynamics(x: jnp.ndarray):
    return jnp.array([x[0] - 0.1 * x[0] ** 3])


def control_matrix(x: jnp.ndarray):
    return jnp.array([[1.0]])


def diffusion_matrix(x: jnp.ndarray):
    return jnp.array([[0.5]])  # Corresponds to L*Q*L^T


def state_cost(x: jnp.ndarray):
    return 0.5 * x[0] ** 2


def final_cost(x: jnp.ndarray):
    return 2.0 * x[0] ** 2


ocp = StochasticOCP(
    a=dynamics,
    b=control_matrix,
    S=diffusion_matrix,
    l=state_cost,
    l_f=final_cost,
    R_inv=jnp.array([[1e1]]),
    lambda_param=1.0,
)

L_bound = 4.0  # state space bound
N_x = 63  # spatial grid size
t_f = 4.0
t_0 = 0.0
N_t = 400  # temporal grid size

x_grid, w_grid, dx, volume, t_grid, dt = setup_grids(L_bound, N_x, t_f, t_0, N_t)
c_tf = project_final_cost(ocp, x_grid, w_grid, dx, volume)
Gamma = galerkin_matrix(ocp, x_grid, w_grid, dx)

coefficients = solve_ode(Gamma, c_tf, volume, dt, N_t)

par_coefficients = solve_par_ode(Gamma, c_tf, volume, dt, N_t)

V = construct_value_functions(ocp, coefficients, x_grid, w_grid)
u_star = construct_optimal_control(ocp, coefficients, x_grid, w_grid)

x_flat = x_grid.flatten()
T, X = jnp.meshgrid(t_grid, x_flat)

plt.subplot(1, 2, 1)
plt.contourf(T, X, V.T, levels=30)
plt.title("$V(x, t)$")
plt.xlabel("$t$")
plt.ylabel("$x(t)$")

plt.subplot(1, 2, 2)
plt.contourf(T, X, u_star.T, levels=30)
plt.title("$u^*(x, t)$")
plt.xlabel("$t$")
plt.show()

print(jnp.allclose(coefficients, par_coefficients))
print(jnp.max(jnp.abs(coefficients - par_coefficients)))