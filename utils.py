from functools import reduce
import operator as op


def kbits(n, k):
    """Generate integer form for all length-n bitstrings of weight k.

    Output indices are ordered consistently but arbitrarily.

    Returns:
        Generator for indices that are ordered by their binary weight.
    """
    limit=1<<n
    val=(1<<k)-1
    while val<limit:
        yield val
        minbit=val&-val #rightmost 1 bit
        fillbit = (val+minbit)&~val  #rightmost 0 to the left of that bit
        val = val+minbit | (fillbit//(minbit<<1))-1

def idxsort_by_weight(n):
    out = [0]
    for k in range(1, n+1):
        out += list(kbits(n, k))
    return out


def ncr(n, r):
    """Efficient computation of n-choose-r"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom
