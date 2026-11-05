from pint_galerkin.objects import StochasticOCP
import jax.numpy as jnp

lambda_param = 0.3
inv_R = 10.0


def a(x):
    mu = 1.0
    q, v = x
    return jnp.array([v, mu * (1.0 - q**2) * v - q])


def b(x):
    return jnp.array([0.0, 1.0])


def S(x):
    b_x = b(x)
    return lambda_param * inv_R * jnp.outer(b_x, b_x)


def l(x):
    q, v = x
    return 0.5 * q**2 + 0.5 * v**2


def l_f(x):
    q, v = x
    return 0.5 * q**2 + 0.5 * v**2


ocp_vdp = StochasticOCP(a, b, S, l_f, l, inv_R, lambda_param)
