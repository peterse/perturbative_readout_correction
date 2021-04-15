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


def invert_pfull_truncated(R, p_prime, w1, w2=None):
    """Compute $R^{-1}p'$ using a series truncation.

    Args:
        R: Baseline response matrix.
        p_prime: Probability distribution representing observations WITH errors
        w1: Cutoff for the _infinite_ series for computing A^{-1}
        w2: Cutoff for the decomposition of R into {R_j: j=1, ..., w2}

    Returns:
        p_fixed: output distribution acheived by inverting R up to truncation
            (w1, w2).
    """

    if w2 is None:
        w2 = w1

    d = R.shape[0]
    n = int(np.log2(d))

    # Compute the inverse of diagonal of R efficiently
    Rdiag_inv = np.zeros((d, d), dtype=float)
    Rdiag_inv[np.diag_indices(d)] = np.reciprocal(np.diag(R))

    # Initialize series truncation for partitioned R
    S = np.zeros((d, d))
    for j in range(1, w2 + 1):
        Rj = np.zeros((d, d))
        Rj[slice_for_Rj(n, j)] = R[slice_for_Rj(n, j)]
        S -= Rdiag_inv @ Rj

    # update output distribution
    v = Rdiag_inv.dot(p_prime)
    p_fixed = np.copy(v)
    for k in range(1, w1 + 1):
        v = S.dot(v)
        p_fixed += v

    return p_fixed


def invert_p0_truncated(R, p_prime, w):
    """Compute $r_T ⋅ p'$ using a matrix Projection.

    Args:
        R: Baseline response matrix.
        p_prime: Probability distribution representing observations WITH errors
        w: Projection point for the matrix $R$ so that R_T includes no indices
            with basis weight greater than w

    Returns:
        p0: corrected probability of the all-zeros bitstring
    """

    d = R.shape[0]
    n = int(np.log2(d))

    # Sort everything by basis weight
    idx = utils.idxsort_by_weight(n)
    R_ord = R[:,idx][idx,:]
    p_prime_ord = p_prime[idx]

    # Compute the inverse of diagonal of R efficiently
    cutoff = sum([utils.ncr(n, r) for r in range(w+1)])
    R_trunc = R_ord[:,:cutoff][:cutoff,:]
    p_prime_trunc = p_prime_ord[:cutoff]

    R_T_inv = np.linalg.inv(R_trunc)
    return R_T_inv.dot(p_prime_trunc)[0]

