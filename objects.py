from typing import NamedTuple, Callable
import jax.numpy as jnp


class StochasticOCP(NamedTuple):
    a: Callable
    b: Callable
    S: Callable
    l_f: Callable
    l: Callable
    R_inv: jnp.ndarray
    lambda_param: float
