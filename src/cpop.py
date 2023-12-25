from typing import Callable, List, Tuple, Union

import numpy as np

linear_segment_cost = lambda i: i


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


def CPOP(
    y: np.ndarray,
    beta: float,
    h: Callable = linear_segment_cost,
    sigma: float = 1,
    verbose: bool = False,
):
    n = len(y)
    tau_hat = [[0]]
    f = SegmentingCostCoefficientsStore()
    # We can use the update_coefficients function by setting the right initial coefficients
    f.set([], 0, np.array([-beta, 0, 0]))

    K = 2 * beta + h(1) + h(n)

    for t in range(1, n):
        for tau in tau_hat:
            update_coefficients(tau, f, y, t, sigma, beta, h)

        # Functional pruning
        # keep the taus for which the interval is not empty
        indices_to_keep = [
            i for i, x in enumerate(get_optimality_intervals(tau_hat, f, t)) if x != []
        ]
        new_taus = [tau_hat[i] + [t] for i in indices_to_keep]

        # Inequality based pruning
        tau_hat = inequality_based_pruning(tau_hat, f, t, K)

        # Finish functional pruning
        if t < n - 1:  # We don't need to do this at the last iteration
            tau_hat += new_taus

        # Clean the store
        f.clean(tau_hat, t)

        if verbose:
            print(f"Iterations {t}/{n} : {len(tau_hat)} taus stored", end="\r")

    if verbose:
        print()

    return tau_hat[np.argmin(get_values(tau_hat, f, n - 1))]


def update_coefficients(
    tau: List[int],
    f: SegmentingCostCoefficientsStore,
    y: np.ndarray,
    t: int,
    sigma: float,
    beta: float,
    h: Callable,
) -> np.ndarray:
    r"""
    Computes the coefficients in :math:`f_{\tau}^{t}(\phi)\ =a_{\tau}^{t}+b_{\tau}^{t}\phi+c_{\tau}^{t}\phi^2`
    """
    s = tau[-1]
    y = y[s + 1 : t + 1]
    seg_len = t - s

    A = (seg_len + 1) * (2 * seg_len + 1) / (6 * seg_len * sigma**2)
    B = (seg_len + 1) / sigma**2 - (seg_len + 1) * (2 * seg_len + 1) / (
        3 * seg_len * sigma**2
    )
    C = -2 / (seg_len * sigma**2) * np.sum(y * np.arange(1, seg_len + 1))
    D = 1 / sigma**2 * np.sum(y**2)
    E = -C - 2 / sigma**2 * np.sum(y)
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

    f.set(tau, t, np.array([a, b, c]))


def get_values(
    tau_hat: List[List[int]],
    f: SegmentingCostCoefficientsStore,
    t: int,
    phi: Union[None, float] = None,
):
    coefficients = f.batch_get(tau_hat, t)
    if phi is None:  # Return the minimum value for each polynomial
        return coefficients[:, 0] - coefficients[:, 1] ** 2 / 4 / coefficients[:, 2]
    else:  # Return the value of each polynomial at phi
        return (
            coefficients[:, 0]
            + coefficients[:, 1] * phi
            + coefficients[:, 2] * phi**2
        )


def get_optimality_intervals(
    tau_hat: List[List[int]], f: SegmentingCostCoefficientsStore, t: int
) -> List[List[Tuple[float, float]]]:
    """Returns the intervals on which each tau in tau_hat is the optimal tau"""
    if len(tau_hat) == 1:
        return [[(-np.inf, np.inf)]]

    # Let's use the indices of tau_hat to keep track of the taus that are still in the set
    tau_temp = list(range(len(tau_hat)))
    optimality_intervals = [[] for _ in range(len(tau_hat))]
    current_phi = -np.inf
    current_tau = np.argmin(get_values(tau_hat, f, t, phi=current_phi))

    while (current_tau in tau_temp and len(tau_temp) > 1) or (
        current_tau not in tau_temp and len(tau_temp) > 0
    ):
        x_taus = []
        for tau in tau_temp:
            if tau == current_tau:
                # We add a value only so that the indices of x_taus match the indices of tau_hat
                x_taus.append(np.inf)
            else:
                coefficients = f.get(tau_hat[tau], t) - f.get(tau_hat[current_tau], t)
                # Solve for the roots
                delta = coefficients[1] ** 2 - 4 * coefficients[2] * coefficients[0]
                if delta < 0:
                    tau_temp.remove(tau)
                    x_taus.append(np.inf)
                    continue
                else:
                    root1 = -coefficients[1] + np.sqrt(delta) / 2 / coefficients[2]
                    root2 = -coefficients[1] - np.sqrt(delta) / 2 / coefficients[2]
                # Only keep roots larger than the current phi
                roots = [root for root in [root1, root2] if root > current_phi]
                if len(roots) == 0:
                    tau_temp.remove(tau)
                    x_taus.append(np.inf)
                else:
                    x_taus.append(np.min(roots))

        new_tau = np.argmin(x_taus)
        new_phi = x_taus[new_tau]
        optimality_intervals[current_tau].append((current_phi, new_phi))
        current_phi = new_phi
        current_tau = new_tau

    return optimality_intervals


def inequality_based_pruning(
    tau_hat: List[List[int]], f: SegmentingCostCoefficientsStore, t: int, K: float
):
    minimums = get_values(tau_hat, f, t)
    filter = minimums <= np.min(minimums) + K

    return [tau_hat[i] for i, x in enumerate(filter) if x]
