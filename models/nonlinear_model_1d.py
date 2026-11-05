from pint_galerkin.objects import StochasticOCP

lambda_param = 0.1
inv_R = 1.0


def a(x):
    return x**3


def b(x):
    return 1.0


def S(x):
    return lambda_param * b(x) * inv_R * b(x)


def l(x):
    return 0.5 * x**2


def l_f(x):
    return 0.5 * x**2 * 10


ocp_nonlin_1d = StochasticOCP(a, b, S, l_f, l, inv_R, lambda_param)
