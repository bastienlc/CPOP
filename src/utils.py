from typing import List, Tuple, Union

import numpy as np


class SegmentingCostCoefficientsStore:
    """The segmenting cost is a quadratic polynomial in phi. We can store it as its three coefficients."""

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
        try:
            value = self.dictionary[self._get_key(tau, t)]
        except KeyError:
            raise KeyError(
                f"Error getting value from SegmentingCostCoefficientsStore at t={t} and tau={tau}"
            )
        return value

    def batch_get(self, tau_list: List[List[int]], t: int) -> np.ndarray:
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
        self.dictionary[self._get_key(tau, t)] = value

    def clean(self, current_tau_hat: List[List[int]], current_t: int) -> None:
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


def get_values(
    tau_hat: List[List[int]],
    f: SegmentingCostCoefficientsStore,
    t: int,
    phi: Union[None, float] = None,
):
    coefficients = f.batch_get(tau_hat, t)
    if phi is None:  # Return the minimum value for each polynomial
        return coefficients[:, 0] - coefficients[:, 1] ** 2 / 4 / coefficients[:, 2]
    elif not np.isinf(phi):  # Return the value of each polynomial at phi
        return (
            coefficients[:, 0]
            + coefficients[:, 1] * phi
            + coefficients[:, 2] * phi**2
        )
    else:
        return np.inf * np.sign(coefficients[:, 2])


def linear_segment_cost(seg_len: int) -> float:
    return seg_len


def precompute_sums(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_cumsum = np.zeros((len(y) + 1))
    y_cumsum[:-1] = np.cumsum(y)
    y_linear_cumsum = np.zeros((len(y) + 1))
    y_linear_cumsum[:-1] = np.cumsum(y * np.arange(1, len(y) + 1))
    y_squarred_cumsum = np.zeros((len(y) + 1))
    y_squarred_cumsum[:-1] = np.cumsum(y**2)
    return y_cumsum, y_linear_cumsum, y_squarred_cumsum


def inequality_based_pruning(
    tau_hat: List[List[int]], f: SegmentingCostCoefficientsStore, t: int, K: float
):
    minimums = get_values(tau_hat, f, t)
    filter = minimums <= np.min(minimums) + K

    return [tau_hat[i] for i, x in enumerate(filter) if x]
