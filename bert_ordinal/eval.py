try:
    import numba
except ModuleNotFoundError as err:
    raise RuntimeError("bert_ordinal.eval requires numba") from err

import numpy as np


@numba.njit
def _jit_qwk(a1: np.ndarray, a2: np.ndarray, num_labels: int) -> float:
    hist1 = np.zeros((num_labels, ))
    hist2 = np.zeros((num_labels, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(num_labels):
        for j in range(num_labels):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def qwk(a1, a2, num_labels: int) -> float:
    """
    This method calculates the Quadratic Weighted Kappa between label arrays
    produced by two annotators -- or predicted and true labels -- `a1` and
    `a2`.

    Args:
        a (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 1. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        b (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 2. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        num_labels (`int`):
            The number of labels which bound the values in `a` and `b`.

    Attribution:
    This code is based on Apache-v2 licensed code (c) Jean-Francois Puget
    Source: https://www.kaggle.com/code/cpmpml/ultra-fast-qwk-calc-method/notebook
    """

    if len(a1) != len(a2):
        raise ValueError(f"Lengths of annotations passed to qwk must be equal {len(a1)} != {len(a2)}")
    a1 = np.asarray(a1, dtype=np.int32)
    a2 = np.asarray(a2, dtype=np.int32)
    return _jit_qwk(a1, a2, num_labels)


@numba.njit
def calc_label_dist_lcm(num_labels):
    res = 1
    for nl in num_labels:
        res = np.lcm(nl - 1, res)
    return res


@numba.njit
def _jit_qwk_multi_norm(a1: np.ndarray, a2: np.ndarray, num_labels, label_dist_lcm):
    max_labels = np.max(num_labels)
    hist1 = np.zeros((max_labels,), dtype=np.int32)
    hist2 = np.zeros((max_labels,), dtype=np.int32)

    o_int = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        # Try and avoid floating point issues by keeping everything as integers
        # This makes hist1 and hist2 bigger by a constant factor label_dist_lcm each
        scaling_factor = label_dist_lcm // (num_labels[k] - 1)
        hist1[i] += scaling_factor
        hist2[j] += scaling_factor
        o_int += (i - j) ** 2 * scaling_factor ** 2

    e_int = 0
    for i in range(max_labels):
        for j in range(max_labels):
            # e is bigger by a factor of label_dist_lcm ** 2
            e_int += hist1[i] * hist2[j] * (i - j) * (i - j)

    # label_dist_lcm ** 2 factor and scaling_factor in general gets cancelled
    # out here
    return 1.0 - ((o_int * a1.shape[0]) / e_int)


def qwk_multi_norm(a1, a2, num_labels, label_dist_lcm=None):
    """
    This method calculates the Quadratic Weighted Kappa between label arrays
    produced by two annotators -- or predicted and true labels -- `a1` and
    `a2`. This one

    Args:
        a (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 1. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        b (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 2. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        num_labels (`numpy.ndarray` of `int32` or convertible iterable)
            The number of labels
        label_dist_lcm (`numpy.ndarray`, *optional*):
            A common multiple of `num_labels[i] - 1` for all `num_labels[i]`.
            This is typically the Lowest Common Multiple (LCM). If not
            presupplied it will be calculated based on `num_labels`.

    Attribution:
    This code is based on Apache-v2 licensed code (c) Jean-Francois Puget
    Source: https://www.kaggle.com/code/cpmpml/ultra-fast-qwk-calc-method/notebook
    """

    if len(a1) != len(a2):
        raise ValueError(f"Lengths of annotations passed to qwk_multi_norm must be equal {len(a1)} != {len(a2)}")
    a1 = np.asarray(a1, dtype=np.int32)
    a2 = np.asarray(a2, dtype=np.int32)
    num_labels = np.asarray(num_labels, dtype=np.int32)
    if label_dist_lcm is None:
        label_dist_lcm = calc_label_dist_lcm(num_labels)
    return _jit_qwk_multi_norm(a1, a2, num_labels, label_dist_lcm)