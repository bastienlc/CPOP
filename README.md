# CPOP

Project for the course "Time Series" of the MVA master at ENS Paris-Saclay (work in progress).

Paper : [Detecting changes in slope with an L0 penalty](https://arxiv.org/pdf/1701.01672.pdf), Robert Maidstone, Paul Fearnhead, Adam Letchford, 2017.

This project implements in python the CPOP (Continuous-piecewise-linear Pruned Optimal Partitioning) algorithm proposed in the paper above to detect changepoints in a univariate time series using a continuous piecewise linear model and an L0 penalty.

The authors made their code (in R and C++) available, but we decided to implement the algorithm from scratch in python to better understand it and to make it easier to use.

## Usage
Detect changepoints in a univariate time series using the CPOP algorithm :
```python
from src import cpop
import numpy as np

y = np.random.normal(size=100, scale=1)

changepoints = CPOP(
    y=y, beta=2 * np.log(len(y)), sigma=1, verbose=True, use_cython=True
)
```

## Report

The report is available [here](report/report.pdf).

## Code

The main code is available in the `src` folder. It contains an implementation of the CPOP algorithm in Python with some Cython optimizations.

## Cython compilation

To compile the Cython code, run the following command from the root of the project :
```bash
make build
```
or if you don't have `make` installed, run the following command from the `src` folder :
```bash
python setup.py build_ext --inplace
```
In either case, make sure you have python with Cython installed, and a C compiler (gcc for instance). If the algorithm cannot find the compiled Cython code, it will automatically fallback to the pure python implementation.

## TO DO

- [x] Implement the CPOP algorithm in Python
- [x] Implement the pruning steps
- [x] Add Cython optimizations
- [ ] Add tests for the core functions (coefficients computation and pruning)
- [ ] Optimize the algorithm
    - [ ] Use a better data structure to store the segmentations
    - [ ] Use a better data structure to store the coefficients
    - [ ] Add a Cython implementation of the coefficients computation
    - [ ] Optimize the Cython implementation of the pruning steps

## Contributors

[@bastienlc](https://github.com/bastienlc),
[@s89ne](https://github.com/s89ne)
