Clone the repository
```
$ git clone https://github.com/casiacob/par-spectral-hjb
```

Create a conda environment and install the package
```
$ conda create --name pint-soc python=3.10
$ conda activate pint-soc
$ cd par-spectral-hjb
$ pip install .
```

Run an example (requires GPU).
```
$ cd examples
$ python runtime_1d.py
```