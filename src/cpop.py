from typing import Callable, List, Optional

import numpy as np

from .coefficients import get_recursive_coefficients, get_segment_coefficients
from .optimality_intervals import get_optimality_intervals
from .utils import (
    SegmentingCostCoefficientsStore,
    compute_costs,
    inequality_based_pruning,
    linear_segment_cost,
    precompute_sums,
)


def CPOP(
    y: np.ndarray,
    beta: float,
    h: Optional[Callable] = linear_segment_cost,
    sigma: Optional[float] = 1,
    verbose: Optional[bool] = False,
    use_cython: Optional[bool] = False,
) -> List[int]:
    r"""Computes the optimal segmentation of y using the CPOP algorithm.

    Parameters
    ----------
    y : np.ndarray
        The time series to segment.
    beta : float
        The L0 penalty for the number of segments.
    h : Callable, optional
        The segment length penalty function, by default linear_segment_cost.
    sigma : float, optional
        The standard deviation of the white gaussian noise, by default 1.
    verbose : bool, optional
        Whether to print the progress of the algorithm, by default False.
    use_cython : bool, optional
        Whether to use the cython implementation, by default False.

    Returns
    -------
    List[int]
        The changepoints of the optimal segmentation. These changepoints are indiced from 0 to n-1, and exclude :math:`\tau_0=0` and :math:`\tau_{m+1}=n`.

    Notations
    ---------
    y_1, ..., y_n
        The data indiced from 1 to n in accordance with the paper.
    y[0], ..., y[n-1]
        The same data indiced in a pythonic way.
    1, ..., n
        The time indices we are going to use. We will make sure to substract 1 to t when indexing y.
    """

    if use_cython:
        try:
            from .c_optimality_intervals import (
                optimality_intervals as c_get_optimality_intervals,
            )
        except ImportError:
            print(
                "ERROR : The cython implementation of optimality_intervals was not found. Make sure you compiled it correctly. Falling back to the python implementation."
            )
            use_cython = False

    # Initialization
    n = len(y)
    # tau_hat is the set of all the segmentations we keep track of. We start with the empty segmentation.
    tau_hat = [[0]]
    f = SegmentingCostCoefficientsStore()
    K = 2 * beta + h(1) + h(n)
    # Precompute the cumulative sums of y, y * t and y^2 to speed up the computations
    y_cumsum, y_linear_cumsum, y_squarred_cumsum = precompute_sums(y)

    # We keep to the indices of the paper (y = y_1, ..., y_n = y[1-1], ..., y[n-1]) to avoid confusions. In this case t ranges from 1 to n.
    for t in range(1, n + 1):
        for tau in tau_hat:
            # Compute the coefficients of the optimal cost for the segmentation tau of y_1, ..., y_t with tau
            if tau == [0]:  # Limit case where tau = [0]
                coefficients = get_segment_coefficients(y, t, sigma, h)
            else:  # General case using the recursion
                coefficients = get_recursive_coefficients(
                    tau,
                    f,
                    y_cumsum,
                    y_linear_cumsum,
                    y_squarred_cumsum,
                    t,
                    sigma,
                    beta,
                    h,
                )
            # Store those coefficients
            f.set(
                tau,
                t,
                coefficients,
            )

        # For each tau in tau_hat, compute the intervals of phi on which tau is the optimal segmentation
        if use_cython:  # Use the cython implementation
            segmenting_cost_coefficients = np.array(f.batch_get(tau_hat, t))
            optimality_intervals = c_get_optimality_intervals(
                tau_hat, segmenting_cost_coefficients, t
            )
        else:  # Use the python implementation
            optimality_intervals = get_optimality_intervals(tau_hat, f, t)

        # Functional pruning : we only keep the segmentations that are optimal for some phi
        tau_star = [tau_hat[i] for i, x in enumerate(optimality_intervals) if x != []]
        next_tau_hat = [tau + [t] for tau in tau_star]

        # Inequality based pruning : we only keep the segmentations that are not dominated by another segmentation
        # Contrarily to what is written in the paper we don't apply it to next_tau_hat as the optimal values of the segmentations in next_tau_hat will be computed in the next iteration
        tau_hat = inequality_based_pruning(tau_hat, f, t, K)

        # Finish functional pruning
        if t < n:  # We don't want to do this at the last iteration
            tau_hat += next_tau_hat

        # Clean the store so that we don't store the coefficients that are not needed anymore. Maybe we should do this only every few iterations.
        f.clean(tau_hat, t)

        if verbose:
            print(f"Iterations {t}/{n} : {len(tau_hat)} taus stored", end="\r")

    if verbose:
        print()

    # Return the changepoints that minimize the cost
    changepoints = tau_hat[np.argmin(compute_costs(tau_hat, f, n))]

    return [x - 1 for x in changepoints[1:]]
