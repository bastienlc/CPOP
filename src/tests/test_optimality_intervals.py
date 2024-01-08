import numpy as np
from matplotlib import pyplot as plt

from ..optimality_intervals import get_optimality_intervals


def test_get_optimality_intervals():
    tau_store = (
        np.array([0, 1, 2], dtype=int),
        np.array([1, 2, 3, 0, 0], dtype=int),
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=int,
        ),
    )
    coefficients = np.array([[12, 10, 3], [4, 5, 3], [7, 8, 3]], dtype=float)

    indices_to_remove = get_optimality_intervals(tau_store, coefficients)
    # Each tau is optimal on some interval
    assert np.array_equal(indices_to_remove, np.array([]))
