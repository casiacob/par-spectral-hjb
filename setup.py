from setuptools import find_packages, setup

setup(
    name="pint_galerkin",
    version="0.0.1",
    author="Casian Iacob",
    author_email="casian.iacob@aalto.fi",
    packages=find_packages(
        include=["pint_galerkin", "pint_galerkin.*"]
    ),
    install_requires=["jax[cuda12]", "matplotlib"],
    zip_safe=False,
)
