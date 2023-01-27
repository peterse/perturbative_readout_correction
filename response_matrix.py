import numpy as np
import utils


def Qmatrix(q01, q10):
    """Single-bit resposne matrix. qij := p(i|j) for bitwise error

    Args:
        q01: p(0|1) single qubit flip likelihood.
        q10: p(1|0) single qubit flip likelihood.

    Returns:
        (2, 2) single qubit response matrix characterized by q01, q10.
    """
    return np.asarray([[1 - q10, q01], [q10, 1 - q01]])


def Rmatrix(q01_arr, q10_arr):
    """Generate a tensor-structure response matrix with normal ordering.

    Args:
        q01_arr: list of likelihoods p(0|1) ordered by corresponding qubit.
        q10_arr: list of likelihoods p(1|0) ordered by corresponding qubit.

    Returns:
        (2**n, 2**n) response matrix characterizing readout error for n qubits.
    """
    out = Qmatrix(q01_arr[0], q10_arr[0])
    for j in range(1, len(q01_arr)):
        out = np.kron(out, Qmatrix(q01_arr[j], q10_arr[j]))
    return out

def resample_Rmatrix(R: np.ndarray, qmax, delta, sampler=0):
    """Resample a response matrix stochastically with violation error delta.

    Each element of the resulting response matrix will be have a multiplicative
    error bounded by q_max with probability no more than delta.

    `sampler` kwarg will determine which distribution these values are re-
    sampled from:
        0: Gaussian with sigma^2 = delta * R_{ij}^2 / (qmax^2)
        1: Uniform with
    """
    if sampler == 0:
        # Variance of Gaussian that is concentrated like delta, up to R_{ij}
        conc_param = np.sqrt(delta) / qmax
    elif sampler == 1:
        # Half-width of Uniform distr that is concentrated like delta, up to R_{ij}
        conc_param = np.sqrt(12 * delta) / qmax
    n = R.shape[0]
    out = np.zeros_like(R)
    for i in range(n):
        for j in range(n):
            if sampler == 0:
                rnew = max(0, np.random.normal(R[i,j], conc_param * R[i,j]))
            elif sampler == 1:
                rnew = max(0, np.random.uniform(low=R[i,j]-conc_param/2, high=R[i,j]+conc_param/2))
            out[i,j] = rnew
    # renormalize
    for i in range(n):
        out[:,i] = out[:,i] / sum(out[:,i])
    return out


# def resample_Rmatrix(R: np.ndarray, releps, delta):
#     """Resample a response matrix stochastically with relative error releps.
#
#     Each element of the resulting response matrix will be have a multiplicative
#     error bounded by q with probability no more than delta.
#     than delta.
#     """
#     n = R.shape[0]
#     k = np.sqrt(1/delta)
#     out = np.zeros_like(R)
#     for i in range(n):
#         for j in range(n):
#             rnew = max(0, np.random.normal(R[i,j], R[i,j] * releps/ k , 1))
#             out[i,j] = rnew
#     # renormalize
#     for i in range(n):
#         out[:,i] = out[:,i] / sum(out[:,i])
#     return out

def generate_characteristic_R(qmax, n, returnmax=False):
    """Generate a response matrix with a characteristic error rate.

    Args:
        qmax: the 'characteristic rate' that should be the maximum over
            individual flip rates
    """
    q01_arr = np.random.random(n) * qmax
    q10_arr = np.random.random(n) * qmax
    if returnmax is True:
        return (Rmatrix(q01_arr, q10_arr), max(max(q01_arr), max(q10_arr)))
    return Rmatrix(q01_arr, q10_arr)


def slice_for_Rj(n, j):
    """Construct an array slice for all indices corresponding to an Rj cut.

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
                out.append((i, k))
    out = np.asarray(list(zip(*out)))
    slices = tuple(np.hstack((out, out[::-1, :])))
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
    R_ord = R[:, idx][idx, :]
    p_prime_ord = p_prime[idx]

    # Compute the inverse of diagonal of R efficiently
    cutoff = sum([utils.ncr(n, r) for r in range(w + 1)])
    R_trunc = R_ord[:, :cutoff][:cutoff, :]
    p_prime_trunc = p_prime_ord[:cutoff]

    R_T_inv = np.linalg.inv(R_trunc)
    return R_T_inv.dot(p_prime_trunc)[0]
