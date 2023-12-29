"""Instead of storing data in classes or dictionaries which is not efficient and doesn't lead to good performance optimization, we store the data in lists of numpy arrays. This increases the performance compared to the original implementation but is not very easy to read. This file contains the functions to manipulate these stores."""

from typing import List, Tuple

import numpy as np
from numba import njit

from .types import CoefficientsStore, TauStore
from .utils import custom_isin

INITIAL_STORE_SIZE = 10


@njit
def init_tau_store(y: np.ndarray) -> TauStore:
    """Initialize the tau store. The memory usage is not optimal (each tau has a fixed size corresponding to the biggest possible segmentation).

    Parameters
    ----------
    y : np.ndarray
        The time series to segment.

    Returns
    -------
    TauStore
        The initialized store.
    """

    return (
        np.zeros((0), dtype=np.int64),
        np.zeros((INITIAL_STORE_SIZE), dtype=np.int64),
        np.zeros((INITIAL_STORE_SIZE, len(y)), dtype=np.int64),
    )


@njit
def init_coefficients_store() -> CoefficientsStore:
    """Initialize the coefficients store.

    Returns
    -------
    CoefficientsStore
        The initialized store.
    """

    return np.zeros((INITIAL_STORE_SIZE, 3), dtype=np.float64)


@njit
def get_indices(tau_store: TauStore) -> np.ndarray:
    """Get the indices of the current taus.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.

    Returns
    -------
    np.ndarray
        The indices of the current taus.
    """
    return tau_store[0]


@njit
def get_tau(tau_store: TauStore, index: int) -> np.ndarray:
    """Get the tau at the given index.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.
    index : int
        The index of the tau to get.

    Returns
    -------
    np.ndarray
        The tau at the given index.
    """
    return tau_store[2][index, : tau_store[1][index]]


@njit
def get_coefficients(coefficients_store: CoefficientsStore, index: int) -> np.ndarray:
    r"""Get the coefficients for the tau at the given index.

    If :math:`\tau=(\tau_0,\tau_1,\dots,\tau_k)`, then the coefficients are those of :math:`f_{\tau_0,\tau_1,\dots,\tau_{k-1}}^{\tau_k}`.

    Parameters
    ----------
    coefficients_store : CoefficientsStore
        The store containing the coefficients.
    index : int
        The index of the coefficients to get.

    Returns
    -------
    np.ndarray
        The coefficients for the tau at the given index.
    """
    return coefficients_store[index, :]


@njit
def add_taus(
    tau_store: TauStore, taus: np.ndarray, lengths: np.ndarray
) -> Tuple[TauStore, np.ndarray]:
    """Add the given taus to the store. The TauStore and the CoefficientsStore must be updated at the same time to keep the correspondence between the two.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.
    taus : np.ndarray
        The taus to add.
    lengths:
        The lengths of the taus to add.

    Returns
    -------
    Tuple[TauStore, np.ndarray]
        The updated store and the indices of the added taus.
    """

    spaces_to_fill = np.arange(len(tau_store[1]))
    spaces_to_fill = spaces_to_fill[~custom_isin(spaces_to_fill, tau_store[0])][
        : len(taus)
    ]
    tau_store[1][spaces_to_fill] = lengths
    tau_store[2][spaces_to_fill] = taus

    new_indices = np.zeros((len(tau_store[0]) + len(spaces_to_fill)), dtype=np.int64)
    new_indices[: len(tau_store[0])] = tau_store[0]
    new_indices[len(tau_store[0]) :] = spaces_to_fill

    return (
        new_indices,
        tau_store[1],
        tau_store[2],
    ), spaces_to_fill


@njit
def add_tau(tau_store: TauStore, tau: np.ndarray, length: int) -> TauStore:
    """Add the given tau to the store.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.
    tau : np.ndarray
        The tau to add.
    length:
        The length of the tau to add.

    Returns
    -------
    TauStore
        The updated store.
    """
    full_tau = np.zeros((1, tau_store[2].shape[1]), dtype=np.int64)
    full_tau[0, :length] = tau
    new_store, _ = add_taus(tau_store, full_tau, np.array([length], dtype=np.int64))
    return new_store


@njit
def increase_stores(
    tau_store: TauStore, coefficients_store: CoefficientsStore
) -> Tuple[TauStore, CoefficientsStore]:
    """Increases the size of the stores.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.

    coefficients_store : CoefficientsStore
        The store containing the coefficients.

    Returns
    -------
    TauStore, CoefficientsStore
        The updated stores.
    """

    return (
        tau_store[0],
        np.concatenate((tau_store[1], np.zeros(len(tau_store[1]), dtype=np.int64))),
        np.concatenate(
            (
                tau_store[2],
                np.zeros((len(tau_store[2]), tau_store[2].shape[1]), dtype=np.int64),
            )
        ),
    ), np.concatenate((coefficients_store, np.zeros((len(coefficients_store), 3))))


@njit
def decrease_stores(
    tau_store: TauStore, coefficients_store: CoefficientsStore
) -> Tuple[TauStore, CoefficientsStore]:
    """Remove the garbage from the stores. This changes the indices of the taus and coefficients.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.

    coefficients_store : CoefficientsStore
        The store containing the coefficients.

    Returns
    -------
    TauStore, CoefficientsStore
        The updated stores.
    """
    if len(tau_store[0]) < len(tau_store[1] // 4):
        indices_to_keep = tau_store[0]
        return (
            (
                np.arange(len(tau_store[0])),
                tau_store[1][indices_to_keep],
                tau_store[2][indices_to_keep],
            ),
            coefficients_store[indices_to_keep],
        )


@njit
def add_taus_and_coefficients_to_stores(
    tau_store: TauStore,
    coefficients_store: CoefficientsStore,
    computed_coefficients: np.ndarray,
    indices_pruned_func: np.ndarray,
    indices_pruned_ineq: np.ndarray,
    t: int,
    n: int,
) -> Tuple[TauStore, CoefficientsStore]:
    """Add the computed coefficients to the stores. Also add the new taus to the store for the next iteration. This function implements algorithmic logic and not just storage logic. This is due to the fact that we need to update the stores at the same time to keep the correspondence between the two.

    Parameters
    ----------
    tau_store : TauStore
        The store containing the current taus.
    coefficients_store : CoefficientsStore
        The store containing the coefficients.
    computed_coefficients : np.ndarray
        The computed coefficients. The indices of the coefficients do not correspond to the indices in tau_store. They are ordered from 0 to len(tau_store[0]) - 1.
    indices_pruned_func : np.ndarray
        The indices pruned by functional pruning. These indices correspond to the indices in tau_store.
    indices_pruned_ineq : np.ndarray
        The indices pruned by inequality pruning. These indices correspond to the indices in tau_store.
    t : int
        The current time.
    n : int
        The length of the time series (maximum time).

    Returns
    -------
    TauStore, CoefficientsStore
        The updated stores.
    """

    indices = get_indices(
        tau_store
    )  # Indices state when the coefficients were computed
    coefficients_indices_map = np.zeros(
        len(coefficients_store), dtype=np.int64
    )  # Array to map indices to coefficients indices
    coefficients_indices_map[indices] = np.arange(len(indices))

    # Update stores sizes if needed
    number_of_taus_to_add = (
        len(tau_store[0]) - len(indices_pruned_func) - len(indices_pruned_ineq)
    )
    if number_of_taus_to_add > len(tau_store[1]) - len(tau_store[0]):
        tau_store, coefficients_store = increase_stores(tau_store, coefficients_store)

    # Functional pruning : add new taus to the store if they were not pruned
    if t < n:
        indices_not_pruned_func = indices[~custom_isin(indices, indices_pruned_func)]

        taus_not_pruned_func = tau_store[2][indices_not_pruned_func]
        for i in range(len(taus_not_pruned_func)):
            taus_not_pruned_func[i, tau_store[1][indices_not_pruned_func[i]]] = t

        tau_store, added_indices = add_taus(
            tau_store,
            taus_not_pruned_func,
            tau_store[1][indices_not_pruned_func] + 1,
        )
        # Store the coefficients of the past iteration corresponding to the new taus
        coefficients_store[added_indices] = computed_coefficients[
            coefficients_indices_map[indices_not_pruned_func]
        ]

    # Inequality pruning : keep the previous taus if they were not pruned
    indices_not_pruned_ineq = indices[~custom_isin(indices, indices_pruned_ineq)]

    # Update the coefficients for the taus that were added at the previous iteration. The other coefficients are not needed.
    indices_to_store_coefficients_for: List[int] = []
    for index in indices_not_pruned_ineq:
        if tau_store[2][index, tau_store[1][index] - 1] == t - 1:
            indices_to_store_coefficients_for.append(index)
    indices_to_store_coefficients_for = np.array(
        indices_to_store_coefficients_for, dtype=np.int64
    )

    coefficients_store[indices_to_store_coefficients_for] = computed_coefficients[
        coefficients_indices_map[indices_to_store_coefficients_for]
    ]

    if t < n:
        new_indices = tau_store[0][
            np.logical_or(
                ~custom_isin(tau_store[0], indices_pruned_ineq),
                custom_isin(tau_store[0], added_indices),
            )
        ]
    else:
        new_indices = indices_not_pruned_ineq
        # At the last iteration we need to store all the coefficients
        coefficients_store[new_indices] = computed_coefficients[
            coefficients_indices_map[new_indices]
        ]

    tau_store = (new_indices, tau_store[1], tau_store[2])

    # Decrease stores sizes if needed
    if len(tau_store[0]) < len(tau_store[1]) // 4:
        tau_store, coefficients_store = decrease_stores(tau_store, coefficients_store)

    return tau_store, coefficients_store
