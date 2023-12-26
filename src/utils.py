from typing import List, Tuple, Union

import numpy as np


class SegmentingCostCoefficientsStore:
    """The segmenting cost is a quadratic polynomial in phi. We can store it as its three coefficients.

    Methods
    -------
    get(tau: List[int], t: int) -> np.ndarray
        Returns the coefficients of the segmenting cost for the segmentation tau of y[:t].
    batch_get(tau_list: List[List[int]], t: int) -> np.ndarray
        Returns the coefficients of the segmenting cost for the segmentations in tau_list of y[:t]. The resulting array has shape (len(tau_list), 3).
    set(tau: List[int], t: int, value: np.ndarray) -> None
        Sets the coefficients of the segmenting cost for the segmentation tau of y[:t] to value.
    clean(current_tau_hat: List[List[int]], current_t: int) -> None
        Deletes the coefficients that are not needed anymore, i.e. they will not be called in the recursive formula for updating the coefficients.
    """

    def __init__(self):
        self.dictionary = {}

    def _get_tau_key(self, tau: List[int]) -> str:
        key = ",".join(map(str, tau))
        return key

    def _get_t_key(self, t: int) -> str:
        return str(t)

    def _get_key(self, tau: List[int], t: int):
        key = self._get_t_key(t) + "|" + self._get_tau_key(tau)
        return key

    def _get_tau_and_t(self, key: str) -> Tuple[List[int], int]:
        tau = key.split("|")[1].split(",")
        if tau == [""]:
            tau = []
        tau = [int(x) for x in tau]
        t = int(key.split("|")[0])
        return tau, t

    def get(self, tau: List[int], t: int) -> np.ndarray:
        r"""
        Returns the coefficients of the segmenting cost for the segmentation tau of y[:t].

        Parameters
        ----------
        tau : List[int]
            The segmentation for which we want the coefficients.
        t : int
            The current time. We are computing the cost of segmenting y[:t] with tau.

        Returns
        -------
        np.ndarray
            The three coefficients in the polynomial :math:`f_{\tau}^{t}(\phi)\ =a_{\tau}^{t}+b_{\tau}^{t}\phi+c_{\tau}^{t}\phi^2`
        """

        try:
            value = self.dictionary[self._get_key(tau, t)]
        except KeyError:
            raise KeyError(
                f"Error getting value from SegmentingCostCoefficientsStore at t={t} and tau={tau}"
            )
        return value

    def batch_get(self, tau_list: List[List[int]], t: int) -> np.ndarray:
        r"""
        Returns the coefficients of the segmenting cost for the segmentations in tau_list of y[:t].

        Parameters
        ----------
        tau_list : List[List[int]]
            The segmentations for which we want the coefficients.
        t : int
            The current time. We are computing the cost of segmenting y[:t] with tau.

        Returns
        -------
        np.ndarray
            The three coefficients in the polynomial :math:`f_{\tau}^{t}(\phi)\ =a_{\tau}^{t}+b_{\tau}^{t}\phi+c_{\tau}^{t}\phi^2` for each tau in tau_list. The resulting array has shape (len(tau_list), 3).
        """

        values = np.zeros((len(tau_list), 3))
        try:
            for i, tau in enumerate(tau_list):
                values[i, :] = self.dictionary[self._get_key(tau, t)]
        except KeyError:
            raise KeyError(
                f"Error getting value from SegmentingCostCoefficientsStore at t={t} and tau={tau}"
            )
        return values

    def set(self, tau: List[int], t: int, value: np.ndarray) -> None:
        r"""
        Sets the coefficients of the segmenting cost for the segmentation tau of y[:t] to value.

        Parameters
        ----------
        tau : List[int]
            The segmentation for which we want to set the coefficients.
        t : int
            The current time. We are storing the cost of segmenting y[:t] with tau.
        value : np.ndarray
            The three coefficients in the polynomial :math:`f_{\tau}^{t}(\phi)\ =a_{\tau}^{t}+b_{\tau}^{t}\phi+c_{\tau}^{t}\phi^2`

        Returns
        -------
        None
        """

        self.dictionary[self._get_key(tau, t)] = value

    def clean(self, current_tau_hat: List[List[int]], current_t: int) -> None:
        r"""
        Deletes the coefficients that are not needed anymore, i.e. they will not be called in the recursive formula for updating the coefficients.

        Parameters
        ----------
        current_tau_hat : List[List[int]]
            The current tau_hat, i.e. the segmentations we are still considering at this time.
        current_t : int
            The current time.

        Returns
        -------
        None
        """

        keys = self.dictionary.keys()
        keys_to_delete = []
        current_tau_hat = [self._get_tau_key(tau) for tau in current_tau_hat]

        for key in keys:
            tau, t = self._get_tau_and_t(key)
            if t < current_t and self._get_tau_key(tau + [t]) not in current_tau_hat:
                keys_to_delete.append(key)

        self.dictionary = {
            k: self.dictionary[k] for k in keys if k not in keys_to_delete
        }


def compute_costs(
    tau_hat: List[List[int]],
    f: SegmentingCostCoefficientsStore,
    t: int,
    phi: Union[None, float] = None,
) -> np.ndarray:
    r"""
    Computes the cost of segmenting y[:t] with each segmentation in tau_hat. If phi is given, then the cost at phi is returned, otherwise the minimum cost is returned.

    Parameters
    ----------
    tau_hat : List[List[int]]
        The segmentations for which we want to compute the cost.
    f : SegmentingCostCoefficientsStore
        The store containing the coefficients of the optimal cost for all the segmentations.
    t : int
        The current time. We are computing the cost of segmenting y[:t] with each segmentation in tau_hat.
    phi : Union[None, float], optional
        The value at which we want to compute the cost, by default None. If None, the minimum cost is returned. If not None, the cost at phi is returned.

    Returns
    -------
    np.ndarray
        The cost of segmenting y[:t] with each segmentation in tau_hat. The resulting array has shape (len(tau_hat),).
    """

    coefficients = f.batch_get(tau_hat, t)
    if phi is None:  # Return the minimum value for each polynomial
        return coefficients[:, 0] - coefficients[:, 1] ** 2 / 4 / coefficients[:, 2]
    elif not np.isinf(phi):  # Return the value of each polynomial at phi
        return (
            coefficients[:, 0]
            + coefficients[:, 1] * phi
            + coefficients[:, 2] * phi**2
        )
    else:  # Handle the case where phi is inf. We return the value of the polynomial at -inf or +inf depending on the sign of the coefficient of phi^2
        return np.inf * np.sign(coefficients[:, 2])


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
    tau_hat: List[List[int]], f: SegmentingCostCoefficientsStore, t: int, K: float
) -> List[List[int]]:
    """
    Prunes the segmentations in tau_hat that have a cost larger than the minimum cost + K.

    Parameters
    ----------
    tau_hat : List[List[int]]
        The segmentations to prune.
    f : SegmentingCostCoefficientsStore
        The store containing the coefficients of the optimal cost for all the segmentations.
    t : int
        The current time.
    K : float
        The pruning parameter as defined in the paper. Pruning with this parameter guarantees that the pruned segmentations will not be optimal.

    Returns
    -------
    List[List[int]]
        The segmentations that were not pruned.
    """
    minimums = compute_costs(tau_hat, f, t)
    filter = minimums <= np.min(minimums) + K

    return [tau_hat[i] for i, x in enumerate(filter) if x]
