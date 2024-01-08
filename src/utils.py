from typing import List, Tuple, Union

import numpy as np

from .coefficients import get_recursive_coefficients, get_segment_coefficients
from .costs import SegmentCost
from .stores import TauStore


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


def reconstruct_segmentation(
    y: np.ndarray,
    changepoints: Union[List[int], np.ndarray],
    sigma: float,
    beta: float,
    h: SegmentCost,
) -> np.ndarray[float]:
    """
    Computes the optimal values at each changepoint of the segmentation. This is useful if we are interested not only in the changepoints but also in the values of the segmentation at each changepoint. We could store these values at each iteration of the algorithm, but it's more efficient to compute them at the end.

    Parameters
    ----------
    y : np.ndarray
        The time series to segment.
    changepoints : Union[List[int], np.ndarray]
        The optimal changepoints, index from 0 to n-1.
    sigma : float
        The standard deviation of the white gaussian noise.
    beta : float
        The L0 penalty for the number of segments.
    h : SegmentCost
        The segment length penalty.

    Returns
    -------
    np.ndarray[float]
        The optimal values at each changepoint.
    """

    y_cumsum, y_linear_cumsum, y_squarred_cumsum = precompute_sums(y)

    coeffs, alpha, gamma = get_segment_coefficients(y, changepoints[1] + 1, sigma, h)

    alphas = [alpha]
    gammas = [gamma]

    for i, t in enumerate(changepoints[2:]):
        coeffs, alpha, gamma = get_recursive_coefficients(
            changepoints[(i + 2) - 1] + 1,
            coeffs,
            y_cumsum,
            y_linear_cumsum,
            y_squarred_cumsum,
            t + 1,
            sigma,
            beta,
            h,
        )
        alphas.append(alpha)
        gammas.append(gamma)

    phis = [-coeffs[1] / (2 * coeffs[2])]

    for alpha, gamma in zip(alphas[::-1], gammas[::-1]):
        phis.append(alpha + gamma * phis[-1])

    return np.array(phis[::-1])


def continuous_piecewise_linear_approximation(
    changepoints: Union[List[int], np.ndarray], phis: Union[List[int], np.ndarray]
) -> np.ndarray:
    """
    Computes the continuous piecewise linear approximation of a time series given its changepoints and the values of the segmentation at each changepoint.

    Parameters
    ----------
    changepoints : Union[List[int], np.ndarray]
        The optimal changepoints.
    phis : Union[List[int], np.ndarray]
        The optimal values at each changepoint.

    Returns
    -------
    np.ndarray
        The continuous piecewise linear approximation.
    """

    segments_approximations = [np.array([phis[0]])]
    for i in range(1, len(changepoints)):
        slope = (phis[i] - phis[i - 1]) / (changepoints[i] - changepoints[i - 1])
        x = np.arange(changepoints[i - 1] + 1, changepoints[i] + 1)
        y = slope * (x - changepoints[i - 1]) + phis[i - 1]
        segments_approximations.append(y)

    return np.concatenate(segments_approximations)
