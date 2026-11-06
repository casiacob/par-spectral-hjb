from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

import time
import numpy as np
import jax

from pint_galerkin.nonlinear_model_1d import ocp_nonlin_1d
from pint_galerkin.fourier_1d import (
    setup_grids,
    project_final_cost,
    spectral_matrix,
)
from pint_galerkin.ode_solvers import (
    par_exp_integrate_time_invariant,
    seq_exp_integrate_time_invariant,
)

L = 1.5
N_freq = 30
N_x = 400
t_f = 4.0
t_0 = 0.0

N_t_values = [200, 400, 800, 1600, 3200, 6400]
n_runs = 10

x_grid, w_grid = setup_grids(L, N_x, N_freq)
c_f, p_x, phi_sin = project_final_cost(ocp_nonlin_1d, x_grid, w_grid, L)
Gamma = spectral_matrix(ocp_nonlin_1d, phi_sin, x_grid, w_grid, L)
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
            timings[name].append(time.perf_counter()- start)

    seq_mean = np.mean(timings["seq"])
    par_mean = np.mean(timings["par"])
    speedup = seq_mean / par_mean

    results.append((N_t, seq_mean, par_mean, speedup))

    print(f"{N_t:6d} {seq_mean:16.6f} " f"{par_mean:16.6f} {speedup:10.2f}")