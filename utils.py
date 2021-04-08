from functools import reduce
import operator as op
import numpy as np


def kbits(n, k):
    """Generate integer form for all length-n bitstrings of weight k.

    Output indices are ordered consistently but arbitrarily.

    DISCLAIMER: ripped from StackOverflow, I don't take credit for this code.
    Args:
        n, k: integers
    Returns:
        Generator for indices that are ordered by their binary weight.
    """
    limit = 1 << n
    val = (1 << k) - 1
    while val < limit:
        yield val
        minbit = val & -val  #rightmost 1 bit
        fillbit = (val + minbit) & ~val  #rightmost 0 to the left of that bit
        val = val + minbit | (fillbit // (minbit << 1)) - 1


def idxsort_by_weight(n):
    """Construct a list of all length-n bitstrings sorted by weight.

    Within each weight class strings are sorted arbitrariy (based on the
    implementation of `kbits`).

    Args:
        n: Number of bits

    Returns:
        List[Int] with length 2**n containing integers that are sorted by
            binary weight
    """
    out = [0]
    for k in range(1, n + 1):
        out += list(kbits(n, k))
    return out


def binarr(m):
    """Produce an ordered column of all binary vectors length m.

    Example for m=3:
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
    """
    d = np.arange(2 ** m)
    return (((d[:,None] & (1 << np.arange(m)))) > 0).astype(int)[:,::-1]


def ncr(n, r):
    """Efficient computation of n-choose-r.

    DISCLAIMER: ripped from StackOverflow, I don't take credit for this code.

    Args:
        n, r: integers

    Returns:
        n-choose-r
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def numberOfSetBits(i):
    """Compute the weight of an integer's binary representation.

    DISCLAIMER: ripped from StackOverflow, I don't take credit for this code.

    Args:
        i: Integer
    Returns:
        weight(i)
    """
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def distance(a, b):
    """Compute the distance between binary represenations a and b.

    This counts the number of indices in which bin(a) and bin(b) differ, which
    is simply the weight of a XOR b.

    Args:
        a, b: integers

    Returns:
        binary distance between `a` and `b`.
    """
    return numberOfSetBits(a ^ b)
