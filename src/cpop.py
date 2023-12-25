from typing import Callable, List

import numpy as np

from .c_optimality_intervals import (
    get_optimality_intervals as c_get_optimality_intervals,
)
from .optimality_intervals import get_optimality_intervals
from .update_coefficients import update_coefficients
from .utils import SegmentingCostCoefficientsStore, get_values

linear_segment_cost = lambda i: i


def CPOP(
    y: np.ndarray,
    beta: float,
    h: Callable = linear_segment_cost,
    sigma: float = 1,
    verbose: bool = False,
    use_cython: bool = False,
):
    n = len(y)
    tau_hat = [[0]]
    f = SegmentingCostCoefficientsStore()
    # We can use the update_coefficients function by setting the right initial coefficients
    f.set([], 0, np.array([-beta, 0, 0]))

    K = 2 * beta + h(1) + h(n)
    y_cumsum = np.cumsum(y)
    y_linear_cumsum = np.cumsum(y * np.arange(1, n + 1))
    y_squarred_cumsum = np.cumsum(y**2)

    for t in range(1, n):
        for tau in tau_hat:
            f.set(
                tau,
                t,
                update_coefficients(
                    tau,
                    f,
                    y_cumsum,
                    y_linear_cumsum,
                    y_squarred_cumsum,
                    t,
                    sigma,
                    beta,
                    h,
                ),
            )

        # Functional pruning
        # keep the taus for which the interval is not empty
        if use_cython:
            segmenting_cost_coefficients = np.array(f.batch_get(tau_hat, t))
            optimality_intervals = c_get_optimality_intervals(
                tau_hat, segmenting_cost_coefficients, t
            )
        else:
            optimality_intervals = get_optimality_intervals(tau_hat, f, t)

        indices_to_keep = [i for i, x in enumerate(optimality_intervals) if x != []]
        new_taus = [tau_hat[i] + [t] for i in indices_to_keep]

        # Inequality based pruning
        tau_hat = inequality_based_pruning(tau_hat, f, t, K)

        # Finish functional pruning
        if t < n - 1:  # We don't need to do this at the last iteration
            tau_hat += new_taus

        # Clean the store
        f.clean(tau_hat, t)

        if verbose:
            print(f"Iterations {t}/{n} : {len(tau_hat)} taus stored", end="\r")

    if verbose:
        print()

    return tau_hat[np.argmin(get_values(tau_hat, f, n - 1))]


def inequality_based_pruning(
    tau_hat: List[List[int]], f: SegmentingCostCoefficientsStore, t: int, K: float
):
    minimums = get_values(tau_hat, f, t)
    filter = minimums <= np.min(minimums) + K

    return [tau_hat[i] for i, x in enumerate(filter) if x]
