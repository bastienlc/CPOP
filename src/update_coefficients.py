from typing import Callable, List

import numpy as np

from .utils import SegmentingCostCoefficientsStore


def update_coefficients(
    tau: List[int],
    f: SegmentingCostCoefficientsStore,
    y_cumsum: np.ndarray,
    y_linear_cumsum: np.ndarray,
    y_squarred_cumsum: np.ndarray,
    t: int,
    sigma: float,
    beta: float,
    h: Callable,
) -> np.ndarray:
    r"""
    Computes the coefficients in :math:`f_{\tau}^{t}(\phi)\ =a_{\tau}^{t}+b_{\tau}^{t}\phi+c_{\tau}^{t}\phi^2`
    """
    s = tau[-1]
    seg_len = t - s

    A = (seg_len + 1) * (2 * seg_len + 1) / (6 * seg_len * sigma**2)
    B = (seg_len + 1) / sigma**2 - (seg_len + 1) * (2 * seg_len + 1) / (
        3 * seg_len * sigma**2
    )
    C = (
        -2
        / (seg_len * sigma**2)
        * (y_linear_cumsum[t] - y_linear_cumsum[s] - s * y_cumsum[t] + s * y_cumsum[s])
    )
    D = 1 / sigma**2 * (y_squarred_cumsum[t] - y_squarred_cumsum[s])
    E = -C - 2 / sigma**2 * (y_cumsum[t] - y_cumsum[s])
    F = (seg_len - 1) * (2 * seg_len - 1) / (6 * seg_len * sigma**2)

    previous_coefficients = f.get(tau[:-1], tau[-1])

    a = A - B**2 / 4 / (previous_coefficients[0] + F)
    b = C - (previous_coefficients[1] + E) * B / 2 / (previous_coefficients[0] + F)
    c = (
        previous_coefficients[2]
        + D
        - (previous_coefficients[1] + E) ** 2 / 4 / (previous_coefficients[0] + F)
        + beta
        + h(seg_len)
    )

    return np.array([a, b, c])
