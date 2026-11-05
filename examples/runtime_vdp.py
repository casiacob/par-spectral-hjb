from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

from pint_galerkin.fourier_2d import (
    setup_grids_2d,
    project_final_cost_2d,
    spectral_matrix_2d,
)

import time
import numpy as np
import jax
import jax.numpy as jnp

from models.vdp_model import ocp_vdp

from pint_galerkin.ode_solvers import (
    par_exp_integrate_time_invariant,
    seq_exp_integrate_time_invariant,
)

L = (3.0, 3.0)
x0 = jnp.array([1.0, 0.0])

N_x = (250, 250)
N_w = (11, 11)
t_0, t_f = 0.0, 8.0
N_t_values = [400, 800, 1600, 3200]
n_runs = 10

x_grid, y_grid, X, Y, XY, W_m, W_n = setup_grids_2d(L, N_x, N_w)
c_f, p_x, phi = project_final_cost_2d(
    ocp_vdp, x_grid, y_grid, X, Y, XY, W_m, W_n, N_x, L
)
Gamma = spectral_matrix_2d(ocp_vdp, phi, x_grid, y_grid, X, Y, XY, W_m, W_n, N_x, L)
jax.block_until_ready((Gamma, c_f))


results = []

print(f"{'N_t':>6} {'Sequential (s)':>16} {'Parallel (s)':>16} {'Speedup':>10}")

for N_t in N_t_values:
    seq_solver = jax.jit(
        lambda Gamma, c_f: seq_exp_integrate_time_invariant(Gamma, c_f, t_f, t_0, N_t)
    )
    par_solver = jax.jit(
        lambda Gamma, c_f: par_exp_integrate_time_invariant(Gamma, c_f, t_f, t_0, N_t)
    )

    jax.block_until_ready(seq_solver(Gamma, c_f))
    jax.block_until_ready(par_solver(Gamma, c_f))

    timings = {"seq": [], "par": []}
    solvers = {"seq": seq_solver, "par": par_solver}

    for run in range(n_runs):
        for name in ("seq", "par"):
            start = time.perf_counter()
            output = solvers[name](Gamma, c_f)
            jax.block_until_ready(output)
            timings[name].append(time.perf_counter() - start)

    seq_mean = np.mean(timings["seq"])
    par_mean = np.mean(timings["par"])
    speedup = seq_mean / par_mean

    results.append((N_t, seq_mean, par_mean, speedup))

    print(f"{N_t:6d} {seq_mean:16.6f} " f"{par_mean:16.6f} {speedup:10.2f}")