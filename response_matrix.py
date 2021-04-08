import numpy as np
import utils

def Qmatrix(q01, q10):
    """Single-bit resposne matrix. qij := p(i|j) for bitwise error"""
    return np.asarray([[1 - q10, q01],
                       [q10, 1 - q01]])


def Rmatrix(q01_arr, q10_arr):
    """Efficiently compute a tensor-structure response matrix with normal ordering on indices."""
    out = Qmatrix(q01_arr[0], q10_arr[0])
    for j in range(1, len(q01_arr)):
        out = np.kron(out, Qmatrix(q01_arr[j], q10_arr[j]))
    return out


def slice_for_Rj(n, j):
    """Construct an array slice for all indices corresponding to an Rj cut.

    TODO: could be more efficient taking
    Args:
        n: number of bits
        j: "order" of R_j to compute
    Returns:
        slices: tuple(np.ndarray, np.ndarray) of zipped (x,y) corrdinates, such
            that R[slices] = R_j
    """
    out = []
    d = 1 << n
    for i in range(d):
        for k in range(d):
            if k < i:
                continue
            if utils.distance(i, k) == j:
                out.append((i,k))
    out = np.asarray(list(zip(*out)))
    slices = tuple(np.hstack((out, out[::-1,:])))
    return slices


def make_truncated_Rinv(R, w1, w2=None):
    """Compute $R^{-1}$ using a series truncation.

    Args:
        R: Baseline response matrix.
        w1: Cutoff for the _infinite_ series for computing A^{-1}
        w2: Cutoff for the decomposition of R into {R_j: j=1, ..., w2}
    """