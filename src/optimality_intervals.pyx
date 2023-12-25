from typing import List, Tuple

from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np

from libc.math cimport sqrt, fmin

ctypedef List[List[int]] TauHatType
ctypedef List[List[Tuple[float, float]]] OptimalityIntervalsType
ctypedef np.float np_float

@boundscheck(False)
@wraparound(False)
cpdef np.ndarray[np_float, ndim=1] c_get_values(
    TauHatType tau_hat,
    np.ndarray[np_float, ndim=2] segmenting_cost_coefficients,
    float phi,
):
    if not np.isinf(phi):  # Return the value of each polynomial at phi
        return (
            segmenting_cost_coefficients[:, 0]
            + segmenting_cost_coefficients[:, 1] * phi
            + segmenting_cost_coefficients[:, 2] * phi**2
        )
    else:
        return np.inf * np.sign(segmenting_cost_coefficients[:, 2])

@boundscheck(False)
@wraparound(False)
cpdef OptimalityIntervalsType get_optimality_intervals(
        TauHatType tau_hat, np.ndarray[np_float, ndim=2] segmenting_cost_coefficients, int t):
    """Returns the intervals on which each tau in tau_hat is the optimal tau"""
    if len(tau_hat) == 1:
        return [[(-np.inf, np.inf)]]

    # Let's use the indices of tau_hat to keep track of the taus that are still in the set
    cdef list[int] tau_temp_indices = list(range(len(tau_hat)))
    cdef OptimalityIntervalsType optimality_intervals = [[] for _ in range(len(tau_hat))]
    cdef float current_phi = -np.inf
    cdef int current_tau_index = np.argmin(c_get_values(tau_hat, segmenting_cost_coefficients, current_phi))
    cdef np.ndarray[np_float, ndim=1] x_taus
    cdef int new_tau
    cdef float new_phi
    cdef np.ndarray[np_float, ndim=1] coefficients
    cdef list[int] indices_to_remove

    while (current_tau_index in tau_temp_indices and len(tau_temp_indices) > 1) or (
            current_tau_index not in tau_temp_indices and len(tau_temp_indices) > 0
    ):
        x_taus = np.zeros(len(tau_temp_indices))
        indices_to_remove = []
        for i, tau_index in enumerate(tau_temp_indices):
            if tau_index == current_tau_index:
                # We add a value only so that the indices of x_taus match the indices of tau_temp_indices
                x_taus[i] = np.inf
            else:
                coefficients = segmenting_cost_coefficients[tau_index, :] - segmenting_cost_coefficients[current_tau_index, :]
                # Solve for the roots
                delta = coefficients[1] ** 2 - 4 * coefficients[2] * coefficients[0]
                if delta < 0:
                    indices_to_remove.append(tau_index)
                    x_taus[i] = np.inf
                else:
                    root1 = -coefficients[1] + sqrt(delta) / 2 / coefficients[2]
                    root2 = -coefficients[1] - sqrt(delta) / 2 / coefficients[2]

                    # Only keep roots larger than the current phi
                    if root1 > current_phi and root2 > current_phi:
                        x_taus[i] = fmin(root1, root2)
                    elif root1 > current_phi:
                        x_taus[i] = root1
                    elif root2 > current_phi:
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
