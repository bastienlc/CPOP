import numpy as np

from ..stores import add_taus_and_coefficients_to_stores


def test_add_taus_and_coefficients_to_stores():
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
    coefficients_store = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0], [0, 0, 0]], dtype=float
    )

    computed_coefficients = np.array(
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]], dtype=float
    )

    indices_pruned_func = np.array([0], dtype=int)
    indices_pruned_ineq = np.array([1], dtype=int)

    new_tau_store, new_coefficients_store = add_taus_and_coefficients_to_stores(
        tau_store,
        coefficients_store,
        computed_coefficients,
        indices_pruned_func,
        indices_pruned_ineq,
        4,
        5,
    )

    # 1 is pruned by ineq pruning, 0 is pruned by func pruning
    assert np.array_equal(new_tau_store[0], np.array([0, 2, 3, 4], dtype=int))
    assert np.array_equal(new_tau_store[1], np.array([1, 2, 3, 3, 4], dtype=int))
    assert np.array_equal(
        new_tau_store[2],
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 1, 4, 0, 0],
                [0, 1, 2, 4, 0],
            ],
            dtype=int,
        ),
    )
    assert np.array_equal(
        new_coefficients_store,
        np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [22, 23, 24], [25, 26, 27]],
            dtype=float,
        ),
    )

    new_tau_store, new_coefficients_store = add_taus_and_coefficients_to_stores(
        tau_store,
        coefficients_store,
        computed_coefficients,
        indices_pruned_func,
        indices_pruned_ineq,
        3,
        5,
    )

    assert np.array_equal(
        new_coefficients_store,
        np.array(
            [[1, 2, 3], [4, 5, 6], [25, 26, 27], [22, 23, 24], [25, 26, 27]],
            dtype=float,
        ),
    )
