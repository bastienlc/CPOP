from typing import List, Optional

import numpy as np

from .coefficients import get_recursive_coefficients, get_segment_coefficients
from .costs import LogCost, SegmentCost
from .optimality_intervals import get_optimality_intervals
from .stores import (
    add_tau,
    add_taus_and_coefficients_to_stores,
    get_coefficients,
    get_indices,
    get_tau,
    init_coefficients_store,
    init_tau_store,
)
from .utils import compute_costs, inequality_based_pruning, precompute_sums


def CPOP(
    y: np.ndarray,
    beta: float,
    h: SegmentCost = LogCost(1),
    sigma: Optional[float] = 1,
    verbose: Optional[bool] = False,
) -> List[int]:
    r"""Computes the optimal segmentation of y using the CPOP algorithm.

    Parameters
    ----------
    y : np.ndarray
        The time series to segment.
    beta : float
        The L0 penalty for the number of segments.
    h : SegmentCost, optional
        The segment length penalty, by default LogCost.
    sigma : float, optional
        The standard deviation of the white gaussian noise, by default 1.
    verbose : bool, optional
        Whether to print the progress of the algorithm, by default False.

    Returns
    -------
    List[int]
        The changepoints of the optimal segmentation. These changepoints are indiced from 0 to n-1, and exclude :math:`\tau_0` and :math:`\tau_{m+1}`.

    Notations
    ---------
    y_1, ..., y_n
        The data indiced from 1 to n in accordance with the paper.
    y[0], ..., y[n-1]
        The same data indiced in a pythonic way.
    1, ..., n
        The time indices we are going to use. We will make sure to substract 1 to t when indexing y.
    """

    # Initialization
    n = len(y)
    # tau_store is the set of all the segmentations we keep track of. We start with the empty segmentation.
    tau_store = init_tau_store(y)
    tau_store = add_tau(tau_store, np.array([0]), 1)
    # coefficients_store is the set of all the segmentations costs we keep track of.
    coefficients_store = init_coefficients_store()

    K = 2 * beta + h(1) + h(n)
    # Precompute the cumulative sums of y, y * t and y^2 to speed up the computations
    y_cumsum, y_linear_cumsum, y_squarred_cumsum = precompute_sums(y)

    # We keep to the indices of the paper (y = y_1, ..., y_n = y[1-1], ..., y[n-1]) to avoid confusions. In this case t ranges from 1 to n.
    for t in range(1, n):
        indices = get_indices(tau_store)
        new_coefficients = np.zeros((len(indices), 3), dtype=float)
        for i, tau_index in enumerate(indices):
            tau = get_tau(tau_store, tau_index)
            # Compute the coefficients of the optimal cost for the segmentation tau of y_1, ..., y_t with tau
            if len(tau) == 1 and tau[0] == 0:  # Limit case where tau = [0]
                new_coefficients[i, :], _, _ = get_segment_coefficients(y, t, sigma, h)
            else:  # General case using the recursion
                new_coefficients[i, :], _, _ = get_recursive_coefficients(
                    tau[-1],
                    get_coefficients(coefficients_store, tau_index),
                    y_cumsum,
                    y_linear_cumsum,
                    y_squarred_cumsum,
                    t,
                    sigma,
                    beta,
                    h,
                )

        # Functional pruning : we only keep the segmentations that are optimal for some phi
        indices_pruned_func = get_optimality_intervals(tau_store, new_coefficients)

        # Inequality based pruning : we only keep the segmentations that are not dominated by another segmentation
        # Contrarily to what is written in the paper we don't apply it to next_tau_hat as the optimal values of the segmentations in next_tau_hat will be computed in the next iteration
        indices_pruned_ineq = inequality_based_pruning(
            tau_store, new_coefficients, t, K
        )

        (tau_store, coefficients_store) = add_taus_and_coefficients_to_stores(
            tau_store,
            coefficients_store,
            new_coefficients,
            indices_pruned_func,
            indices_pruned_ineq,
            t,
            n,
        )

        if verbose and t % (n // 10) == 0:
            print(f"Iterations {t}/{n} : {len(tau_store[0])} taus stored", end="\r")

    if verbose:
        print()

    # Return the changepoints that minimize the cost
    best_index = tau_store[0][
        np.argmin(compute_costs(coefficients_store[tau_store[0]]))
    ]

    changepoints = tau_store[2][best_index, : tau_store[1][best_index]]

    # We substract 1 to the changepoints to get the indices from 0 to n-1, and remove the first changepoint (tau_0)
    return changepoints[1:] - 1
