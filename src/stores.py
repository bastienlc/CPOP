"""Instead of storing data in classes or dictionaries which is not efficient and doesn't lead to good performance optimization, we store the data in lists of numpy arrays. This increases the performance compared to the original implementation but is not very easy to read. This file contains the functions to manipulate these stores."""

from typing import List, Tuple

import numpy as np

INITIAL_STORE_SIZE = 10

TauStore = List[np.ndarray]
"""A list to store current segmentations.

TauStore
    List[np.ndarray, np.ndarray, np.ndarray]

TauStore[0]
    np.ndarray[ndim=1,dtype=int]
        The indices at which the current taus are stored. This is needed because we don't want to reconstruct the store at each iteration.

TauStore[1]
    np.ndarray[ndim=1,dtype=int]
        The length of the current taus. Only the indices in TauStore[0] are valid, the other ones are garbage. We need this because the taus are of variable length. We looked into using a ragged array but it was not efficient in our case.

TauStore[2]
    np.ndarray[ndim=2,dtype=int]
        The current taus. Only the indices in TauStore[0] are valid, the other ones are garbage. Each tau is of variable length and the last values are garbage. The values are valid up to the index in TauStore[1] - 1.
"""

CoefficientsStore = np.ndarray
r"""An array to store segmentation costs. The segmentation cost is a quadratic polynomial in :math:`\phi`. We can store it as its three coefficients.

CoefficientsStore
    np.ndarray[ndim=2,dtype=float]
        The current coefficients. Each row is a segmentation and each column is a coefficient. The first column is the constant coefficient, the second one is the linear coefficient and the third one is the quadratic coefficient. The rows index correspond exactly to the ones in TauStore[0], the other ones are garbage. The costs we store are generally not the ones for the current time, but the ones that will be used in recursive calls.
"""


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

    return [
        np.array([], dtype=int),
        np.zeros(INITIAL_STORE_SIZE, dtype=int),
        np.zeros((INITIAL_STORE_SIZE, len(y)), dtype=int),
    ]


def init_coefficients_store() -> CoefficientsStore:
    """Initialize the coefficients store.

    Returns
    -------
    CoefficientsStore
        The initialized store.
    """

    return np.zeros((INITIAL_STORE_SIZE, 3), dtype=float)


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
    spaces_to_fill = spaces_to_fill[~np.isin(spaces_to_fill, tau_store[0])][: len(taus)]
    tau_store[1][spaces_to_fill] = lengths
    tau_store[2][spaces_to_fill] = taus

    return [
        np.concatenate((tau_store[0], spaces_to_fill)),
        tau_store[1],
        tau_store[2],
    ], spaces_to_fill


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
    new_store, _ = add_taus(tau_store, np.array([tau]), np.array([length]))
    return new_store


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

    return [
        tau_store[0],
        np.concatenate((tau_store[1], np.zeros(len(tau_store[1]), dtype=int))),
        np.concatenate(
            (
                tau_store[2],
                np.zeros((len(tau_store[2]), tau_store[2].shape[1]), dtype=int),
            )
        ),
    ], np.concatenate((coefficients_store, np.zeros((len(coefficients_store), 3))))


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
            [
                np.arange(len(tau_store[0])),
                tau_store[1][indices_to_keep],
                tau_store[2][indices_to_keep],
            ],
            coefficients_store[indices_to_keep],
        )


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
        len(coefficients_store), dtype=int
    )  # Array to map indices to coefficients indices
    coefficients_indices_map[indices] = np.arange(len(indices))

    # Update stores sizes if needed
    indices_pruned = np.intersect1d(indices_pruned_func, indices_pruned_ineq)
    number_of_taus_to_add = len(tau_store[0]) - len(indices_pruned)

    if number_of_taus_to_add > len(tau_store[1]) - len(tau_store[0]):
        tau_store, coefficients_store = increase_stores(tau_store, coefficients_store)

    # Functional pruning : add new taus to the store if they were not pruned
    if t < n - 1:
        indices_not_pruned_func = indices[~np.isin(indices, indices_pruned_func)]

        taus_not_pruned_func = tau_store[2][indices_not_pruned_func]
        taus_not_pruned_func[
            np.arange(len(taus_not_pruned_func)), tau_store[1][indices_not_pruned_func]
        ] = t

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
    indices_not_pruned_ineq = indices[~np.isin(indices, indices_pruned_ineq)]

    if t < n - 1:
        tau_store[0] = tau_store[0][
            np.logical_or(
                np.isin(tau_store[0], indices_not_pruned_ineq),
                np.isin(tau_store[0], added_indices),
            )
        ]
    else:
        tau_store[0] = indices_not_pruned_ineq
        # At the last iteration we need to store all the coefficients
        coefficients_store[tau_store[0]] = computed_coefficients[
            coefficients_indices_map[tau_store[0]]
        ]

    # Decrease stores sizes if needed
    if len(tau_store[0]) < len(tau_store[1]) // 4:
        tau_store, coefficients_store = decrease_stores(tau_store, coefficients_store)

    return tau_store, coefficients_store
