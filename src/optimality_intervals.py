from math import sqrt
from typing import List

import numpy as np

from .stores import TauStore, get_indices

EPS = 1e-6


def minimum_at_minus_infinity(coefficients: np.ndarray) -> int:
    r"""Returns the index of the tau that is minimum at minus infinity.

    Parameters
    ----------
    coefficients : np.ndarray
        The optimal coefficients for each tau

    Returns
    -------
    int
        The index of the tau that is minimum at infinity.
    """

    indices = np.arange(len(coefficients))

    # Candidates based on the SMALLER quadratic coefficient
    candidates = indices[coefficients[:, 2] == coefficients[:, 2].min()]

    # If there are several candidates, we take the one with the BIGGEST linear coefficient
    if len(candidates) > 1:
        candidates = candidates[
            coefficients[candidates, 1] == coefficients[candidates, 1].max()
        ]

    # If there are still several candidates, we take the one with the SMALLEST constant coefficient
    if len(candidates) > 1:
        candidates = candidates[
            coefficients[candidates, 0] == coefficients[candidates, 0].min()
        ]

    # If there are still several candidates, we take the first one as they are all equal
    return candidates[0]


def get_optimality_intervals(
    tau_store: TauStore, coefficients: np.ndarray
) -> np.ndarray:
    r"""Computes the intervals of :math:`\phi` on which each tau in tau_store is the optimal tau. In the paper this corresponds to :math:`Int_{\tau}^{t}`. This algorithm is the one proposed in the paper. Returns the indices of the taus that are never optimal.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.
    coefficients : np.ndarray
        The optimal coefficients for each tau in tau_store.

    Returns
    -------
    np.ndarray
        The indices of the taus to remove.
    """

    temp_indices = get_indices(tau_store)
    indices_store_map = np.zeros(len(tau_store[1]), dtype=int)
    indices_store_map[temp_indices] = np.arange(len(temp_indices))
    indices_to_keep: List[int] = []

    # If there is only one segmentation, it is optimal on the whole interval
    if len(temp_indices) == 1:
        return np.array([])

    # Which tau is minimum at - inf ?
    current_phi = -np.inf
    current_index = temp_indices[minimum_at_minus_infinity(coefficients)]
    indices_to_keep.append(current_index)

    # Check where the current tau intersects with the other taus. The first one to intersect is the next optimal tau.
    while (current_index in temp_indices and len(temp_indices) > 1) or (
        current_index not in temp_indices and len(temp_indices) > 0
    ):
        x_taus: List[float] = []
        indices_from_store_to_remove: List[int] = []
        for index in temp_indices:
            if index == current_index:  # Skip the current tau
                x_taus.append(np.inf)
            else:
                # Polynomial difference
                coefs = (
                    coefficients[indices_store_map[index]]
                    - coefficients[indices_store_map[current_index]]
                )
                # Solve for the roots
                if coefs[2] != 0:
                    delta = coefs[1] ** 2 - 4 * coefs[2] * coefs[0]
                    if delta < 0:
                        # This tau can be removed, skip it
                        indices_from_store_to_remove.append(index)
                        x_taus.append(np.inf)
                        continue
                    else:
                        root1 = (-coefs[1] + sqrt(delta)) / 2 / coefs[2]
                        root2 = (-coefs[1] - sqrt(delta)) / 2 / coefs[2]
                else:
                    if coefs[1] != 0:
                        root1 = -coefs[0] / coefs[1]
                        root2 = -coefs[0] / coefs[1]
                    else:
                        if coefs[0] >= 0:
                            # This tau can be removed, skip it
                            indices_from_store_to_remove.append(index)
                            x_taus.append(np.inf)
                            continue
                        else:
                            # This case should not happen, otherwise we would have selected this tau earlier
                            raise ValueError(
                                "Unexpected case in get_optimality_intervals."
                            )

                # Only keep roots larger than the current phi
                if root1 > current_phi + EPS and root2 > current_phi + EPS:
                    x_taus.append(min(root1, root2))  # lol
                elif root1 > current_phi + EPS:
                    x_taus.append(root1)
                elif root2 > current_phi + EPS:
                    x_taus.append(root2)
                else:
                    indices_from_store_to_remove.append(index)
                    x_taus.append(np.inf)

        # First intersection
        arg_min = np.argmin(x_taus)
        current_index = temp_indices[arg_min]
        current_phi = x_taus[arg_min]

        indices_to_keep.append(current_index)

        # Remove the indices that are never optimal
        temp_indices = temp_indices[
            ~np.isin(temp_indices, indices_from_store_to_remove)
        ]

    return get_indices(tau_store)[~np.isin(get_indices(tau_store), indices_to_keep)]
