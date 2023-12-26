from math import sqrt
from typing import List, Tuple

import numpy as np

from .utils import SegmentingCostCoefficientsStore, compute_costs

EPS = 1e-6


def get_optimality_intervals(
    tau_hat: List[List[int]], f: SegmentingCostCoefficientsStore, t: int
) -> List[List[Tuple[float, float]]]:
    r"""Computes the intervals of :math:`\phi` on which each tau in tau_hat is the optimal tau. In the paper this corresponds to :math:`Int_{\tau}^{t}`. This algorithm is the one proposed in the paper.

    Parameters
    ----------
    tau_hat : List[List[int]]
        The segmentations for which we want to compute the optimality intervals.
    f : SegmentingCostCoefficientsStore
        The store containing the coefficients of the optimal cost for all the segmentations.
    t : int
        The current time.

    Returns
    -------
    List[List[Tuple[float, float]]]
        The optimality intervals for each segmentation in tau_hat.
    """

    # If there is only one segmentation, it is optimal on the whole interval
    if len(tau_hat) == 1:
        return [[(-np.inf, np.inf)]]

    # We use the indices of tau_hat to keep track of the taus that are still in the set
    tau_temp_indices = list(range(len(tau_hat)))
    optimality_intervals = [[] for _ in range(len(tau_hat))]
    current_phi = -np.inf
    current_tau_index = np.argmin(compute_costs(tau_hat, f, t, phi=current_phi))

    while (current_tau_index in tau_temp_indices and len(tau_temp_indices) > 1) or (
        current_tau_index not in tau_temp_indices and len(tau_temp_indices) > 0
    ):
        x_taus = np.zeros(len(tau_temp_indices))
        indices_to_remove = []
        for i, tau_index in enumerate(tau_temp_indices):
            if tau_index == current_tau_index:
                # We add a value so that the indices of x_taus match the indices of tau_temp
                x_taus[i] = np.inf
            else:
                coefficients = f.get(tau_hat[tau_index], t) - f.get(
                    tau_hat[current_tau_index], t
                )
                # Solve for the roots
                delta = coefficients[1] ** 2 - 4 * coefficients[2] * coefficients[0]
                if delta < 0:
                    indices_to_remove.append(tau_index)
                    x_taus[i] = np.inf
                else:
                    root1 = -coefficients[1] + sqrt(delta) / 2 / coefficients[2]
                    root2 = -coefficients[1] - sqrt(delta) / 2 / coefficients[2]

                    # Only keep roots larger than the current phi
                    if root1 > current_phi + EPS and root2 > current_phi + EPS:
                        x_taus[i] = min(root1, root2)
                    elif root1 > current_phi + EPS:
                        x_taus[i] = root1
                    elif root2 > current_phi + EPS:
                        x_taus[i] = root2
                    else:
                        indices_to_remove.append(tau_index)
                        x_taus[i] = np.inf

        new_index = np.argmin(x_taus)
        new_tau_index = tau_temp_indices[new_index]
        new_phi = x_taus[new_index]
        optimality_intervals[current_tau_index].append((current_phi, new_phi))
        current_phi = new_phi
        current_tau_index = new_tau_index

        tau_temp_indices = [
            tau_index
            for tau_index in tau_temp_indices
            if tau_index not in indices_to_remove
        ]

    return optimality_intervals