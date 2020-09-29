import numpy as np
from scipy import sparse
import sisl as si
import operator
from functools import wraps
import itertools


class LCSR:
    """LCSR: List of Compressed Sparse Row matrices

    This object is useful for doing maths with several csr-matrices at once.
    In particular this addresses a weakness in sisl: Math with two SparseCSRs is
    unreasonably slow. Instead you can do a quick conversion to LCSR, do the math,
    and convert back, like so:

    >>> dHS = HSnew.copy(); dHS._csr = (LCSR(dHS._csr) - LCSR(HSold._csr)).tosisl()

    For larger SparseCSRs this can be orders of magnitude faster then the direct
    sisl way `dHS = HSnew - HSold`. But it is also faster even with small systems --
    including conversion!

    Parameters
    ----------
    obj : list or tuple of scipy.sparse.csr_matrix OR sisl.SparseCSR
    """
    def __init__(self, obj):
        if isinstance(obj, (tuple, list)):
            self._csrs = obj
        elif isinstance(obj, si.SparseCSR):
            self._csrs = [obj.tocsr(i) for i in range(obj.dim)]
        else:
            raise TypeError("Pass a list, tuple or sisl csr.")

    @property
    def dim(self):
        return len(self._csrs)

    def tosisl(self):
        return si.SparseCSR.fromsp(*self._csrs)


def _LCSR_binop(op):
    @wraps(op)
    def _op(self, other):
        if isinstance(other, LCSR):
            it = zip(self._csrs, other._csrs)
        else:
            it = zip(self._csrs, itertools.repeat(other))
        return LCSR([op(s, o) for s, o in it])
    return _op


_binops = {
    "add", "iadd", "sub", "isub", "mul", "imul", "truediv", "itruediv"
}
for bop in _binops:
    setattr(LCSR, f"__{bop}__", _LCSR_binop(getattr(operator, bop)))

