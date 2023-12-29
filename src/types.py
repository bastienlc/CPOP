"""Instead of storing data in classes or dictionaries which is not efficient and doesn't lead to good performance optimization, we store the data in lists of numpy arrays. This increases the performance compared to the original implementation but is not very easy to read. This file contains the functions to manipulate these stores."""

from typing import Tuple

import numpy as np

INITIAL_STORE_SIZE = 10


TauStore = Tuple[np.ndarray, np.ndarray, np.ndarray]
"""A tuple to store current segmentations.

TauStore
    Tuple[np.ndarray, np.ndarray, np.ndarray]

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
        The current coefficients. Each row is a segmentation and each column is a coefficient. The first column is the constant coefficient, the second one is the linear coefficient and the third one is the quadratic coefficient. The rows index correspond exactly to the ones in TauStore[0], the other ones are garbage. The values correspond to the coefficients of the segmentations that were in TauStore[0] at the previous iteration. We only need to store the coefficients for the past iteration because we compute the coefficients for the current iteration using the coefficients of the past iteration.
"""
