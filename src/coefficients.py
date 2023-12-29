from typing import Callable

import numpy as np
from numba import njit


@njit
def get_recursive_coefficients(
    s: int,
    past_coefficients: np.ndarray,
    y_cumsum: np.ndarray,
    y_linear_cumsum: np.ndarray,
    y_squarred_cumsum: np.ndarray,
    t: int,
    sigma: float,
    beta: float,
    h: Callable,
) -> np.ndarray:
    r"""
    Computes the optimal coefficients for :math:`f_{\tau}^{t}(\phi)` in the case where tau!=[0].

    :math:`y` is split between :math:`y_1, ..., y_{tau_k}` (recursion) and :math:`y_{tau_k+1}, ..., y_t` (new segment cost).
    We compute A, B, C, D, E and F the coefficients of :math:`\mathcal{C}(y_{tau_k+1}, ..., y_t, \phi', \phi)` which is a polynomial in :math:`\phi'` and :math:`\phi`. We can then compute the coefficients of :math:`f_{\tau}^{t}(\phi)` using the coefficients of :math:`f_{\tau_1, ..., tau_{k-1}}^{tau_k}(\phi')` and minimizing the segment cost over :math:`\phi'`.

    Parameters
    ----------
    s : int
        The last changepoint of the segmentation :math:`\tau`.
    past_coefficients : np.ndarray
        The coefficients of the optimal cost for the segmentation :math:`tau_1, ..., tau_{k-1}` of :math:`y_1, ..., y_{tau_k}`.
    y_cumsum : np.ndarray
        The cumulative sum of the time series. y_cumsum[i] = y[0] + ... + y[i] and y_cumsum[-1] = 0.
    y_linear_cumsum : np.ndarray
        The cumulative sum of the time series weighted by the time. y_linear_cumsum[i] = y[0] * 1 + ... + y[i] * (i + 1) and y_linear_cumsum[-1] = 0.
    y_squarred_cumsum : np.ndarray
        The cumulative sum of the time series squared. y_squarred_cumsum[i] = y[0]^2 + ... + y[i]^2 and y_squarred_cumsum[-1] = 0.
    t : int
        The current time. We are computing the cost of segmenting :math:`y_1, ..., y_t` = y[0], ..., y[t-1] with :math:`\tau`.
    sigma : float
        The standard deviation of the white gaussian noise.
    beta : float
        The L0 penalty for the number of segments.
    h : Callable
        The segment length penalty function.

    Returns
    -------
    np.ndarray
        The three coefficients in the polynomial :math:`f_{\tau}^{t}(\phi)\ =a_{\tau}^{t}+b_{\tau}^{t}\phi+c_{\tau}^{t}\phi^2`
    """

    # s = :math:`tau_k`, we make sure to substract 1 to s when indexing y.
    # y is split between y[0], ..., y[s-1] (recursion) and y[s], ..., y[t-1] (new segment cost)
    seg_len = t - s

    A = (seg_len + 1) * (2 * seg_len + 1) / (6 * seg_len * sigma**2)
    B = (seg_len + 1) / sigma**2 - (seg_len + 1) * (2 * seg_len + 1) / (
        3 * seg_len * sigma**2
    )
    # y_linear_cumsum[t - 1] - y_linear_cumsum[s - 1] = y[s] * (s + 1) + ... + y[t - 1] * t
    # y_cumsum[t - 1] - y_cumsum[s - 1] = y[s] + ... + y[t - 1]
    # y_linear_cumsum[t - 1] - y_linear_cumsum[s - 1] - s * (y_cumsum[t - 1] - y_cumsum[s - 1])
    # = y[s] * 1 + ... + y[t - 1] * (t - s)
    C = (
        -2
        / (seg_len * sigma**2)
        * (
            y_linear_cumsum[t - 1]
            - y_linear_cumsum[s - 1]
            - s * y_cumsum[t - 1]
            + s * y_cumsum[s - 1]
        )
    )
    D = 1 / sigma**2 * (y_squarred_cumsum[t - 1] - y_squarred_cumsum[s - 1])
    # y_cumsum[t - 1] - y_cumsum[s - 1] = y[s] + ... + y[t - 1]
    E = -C - 2 / sigma**2 * (y_cumsum[t - 1] - y_cumsum[s - 1])
    F = (seg_len - 1) * (2 * seg_len - 1) / (6 * seg_len * sigma**2)

    # Minimizing over :math:`\phi'` we have the following polynomial in :math:`\phi`
    a = A - B**2 / 4 / (past_coefficients[0] + F)
    b = C - (past_coefficients[1] + E) * B / 2 / (past_coefficients[0] + F)
    c = (
        past_coefficients[2]
        + D
        - (past_coefficients[1] + E) ** 2 / 4 / (past_coefficients[0] + F)
        + beta
        + h(seg_len)
    )

    return np.array([a, b, c])


@njit
def get_segment_coefficients(
    y: np.ndarray,
    t: int,
    sigma: float,
    h: Callable,
) -> np.ndarray:
    r"""Computes the optimal coefficients for :math:`f_{[0]}^{t}(\phi)` in the limit case where tau=[0]. In all fairness we could have used the same function as for the recursive case but we wanted to make it clear that we are in the limit case.

    Parameters
    ----------
    y : np.ndarray
        The complete time series.
    t : int
        The current time. We are computing the cost of the segment y_1, ..., y_t = y[0], ..., y[t-1].
    sigma : float
        The standard deviation of the white gaussian noise.
    h : Callable
        The segment length penalty function.

    Returns
    -------
    np.ndarray
        The three coefficients in the polynomial :math:`f_{[0]}^{t}(\phi)\ =a_{[0]}^{t}+b_{[0]}^{t}\phi+c_{[0]}^{t}\phi^2`
    """
    y = y[:t]
    seg_len = len(y)

    A = (seg_len + 1) * (2 * seg_len + 1) / (6 * seg_len * sigma**2)
    B = (seg_len + 1) / sigma**2 - (seg_len + 1) * (2 * seg_len + 1) / (
        3 * seg_len * sigma**2
    )
    C = -2 / (seg_len * sigma**2) * np.sum(y * np.arange(1, seg_len + 1))
    D = 1 / sigma**2 * np.sum(y**2)
    E = -C - 2 / sigma**2 * np.sum(y)
    F = (seg_len - 1) * (2 * seg_len - 1) / (6 * seg_len * sigma**2)

    if F == 0:  # Case where there is a single point. We return the ||.||_2^2 error.
        return np.array([y[0] ** 2, -2 * y[0], 1])
    else:  # Case where there are at least two points. The optimal cost is a polynomial in phi.
        a = D - E**2 / 4 / F + h(seg_len)
        b = C - B * E / 2 / F
        c = A + B**2 / 4 / F

        return np.array([a, b, c])
