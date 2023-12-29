from typing import Tuple, Union

import numpy as np
from numba import njit

from .types import TauStore


@njit
def compute_costs(
    coefficients: np.ndarray,
    phi: Union[None, float] = None,
) -> np.ndarray:
    r"""
    Computes the cost of segmenting y[:t] with each segmentation in tau_store. If phi is given, then the cost at phi is returned, otherwise the minimum cost is returned.

    Parameters
    ----------
    coefficients : np.ndarray
        The optimal coefficients for each tau in tau_store.
    phi : Union[None, float], optional
        The value at which we want to compute the cost, by default None. If None, the minimum cost is returned. If not None, the cost at phi is returned.

    Returns
    -------
    np.ndarray
        The cost of segmenting y[:t] with each segmentation in tau_store.
    """

    if phi is None:  # Return the minimum value for each polynomial
        return coefficients[:, 0] - coefficients[:, 1] ** 2 / 4 / coefficients[:, 2]
    else:  # Return the value of each polynomial at phi
        return (
            coefficients[:, 0]
            + coefficients[:, 1] * phi
            + coefficients[:, 2] * phi**2
        )


@njit
def linear_segment_cost(seg_len: int) -> float:
    """
    The segment cost function used in the paper. It is linear in the segment length, and is equal to 0 for a segment of length 0.

    Parameters
    ----------
    seg_len : int
        The length of the segment.

    Returns
    -------
    float
        The segment cost.
    """
    return float(seg_len)


@njit
def precompute_sums(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes the cumulative sums of y, y * t and y**2. This is used to speed up the computations. The cumulative sums are returned with an additional 0 at the end to handle potential empty segments.

    Parameters
    ----------
    y : np.ndarray
        The time series to segment.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The cumulative sums of y, y * t and y**2.
    """

    y_cumsum = np.zeros((len(y) + 1))
    y_cumsum[:-1] = np.cumsum(y)
    y_linear_cumsum = np.zeros((len(y) + 1))
    y_linear_cumsum[:-1] = np.cumsum(y * np.arange(1, len(y) + 1))
    y_squarred_cumsum = np.zeros((len(y) + 1))
    y_squarred_cumsum[:-1] = np.cumsum(y**2)
    return y_cumsum, y_linear_cumsum, y_squarred_cumsum


@njit
def inequality_based_pruning(
    tau_store: TauStore, coefficients: np.ndarray, t: int, K: float
) -> np.ndarray:
    """
    Prunes the segmentations in tau_store that have a cost larger than the minimum cost + K.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.
    coefficients : np.ndarray
        The optimal coefficients for each tau in tau_store.
    t : int
        The current time.
    K : float
        The pruning parameter as defined in the paper. Pruning with this parameter guarantees that the pruned segmentations will not be optimal.

    Returns
    -------
    np.ndarray
        The indices of the taus to remove.
    """
    minimums = compute_costs(coefficients)
    filter = minimums > np.min(minimums) + K

    return tau_store[0][filter]


@njit
def custom_isin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Custom implementation of np.isin. This function is needed because np.isin is not supported by numba.

    Parameters
    ----------
    a : np.ndarray
        The array to check.
    b : np.ndarray
        The values to check for.

    Returns
    -------
    np.ndarray
        The indices of the values in b that are in a.
    """
    result = np.zeros(len(a), dtype=np.bool_)

    for el in b:
        result[a == el] = True

    return result
