import numpy as np


class SegmentCost:
    """
    A generic class for segment cost functions.

    Methods
    -------
    __call__(seg_len: int) -> float
        Returns the segment cost for a segment of length seg_len.
    """

    def __init__(self):
        pass

    def __call__(self, seg_len: int) -> float:
        """
        Returns the segment cost for a segment of length seg_len.

        Parameters
        ----------
        seg_len : int
            The length of the segment.

        Returns
        -------
        float
            The segment cost.
        """
        raise NotImplementedError


class LinearCost(SegmentCost):
    """
    The segment cost used in the paper. It is linear in the segment length, and is equal to 0 for a segment of length 0.

    Methods
    -------
    __init__(scale: float)
        Initializes the segment cost with a scale parameter. The segment cost will be equal to scale * seg_len.

    __call__(seg_len: int) -> float
        Returns the segment cost for a segment of length seg_len.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, seg_len: int) -> float:
        return self.scale * float(seg_len)


class LogCost(SegmentCost):
    """
    A logarithmic segment cost. It is equal to 0 for a segment of length 0.

    Methods
    -------
    __init__(scale: float)
        Initializes the segment cost with a scale parameter. The segment cost will be equal to scale * log(seg_len).

    __call__(seg_len: int) -> float
        Returns the segment cost for a segment of length seg_len.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, seg_len: int) -> float:
        return self.scale * np.log(float(seg_len))
