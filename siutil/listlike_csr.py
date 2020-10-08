import numpy as np
from scipy import sparse
import sisl as si
import operator
from functools import wraps
import itertools
import warnings
import scipy
from itertools import zip_longest


SCIPYVERSION = tuple(int(x) for x in scipy.__version__.split(".")[:3])


def _upcast_3index(idx):
    if (
        isinstance(idx, int)
        or isinstance(idx, list)
        or (isinstance(idx, np.ndarray) and len(idx.shape) == 1)
        ):
        idx = (idx,)
    if not isinstance(idx, tuple):
        raise TypeError(f"Cannot handle indexing with a {type(idx)}.")
    if len(idx) > 3:
        raise ValueError("Too many dimensions to index with")
    if len(idx) < 3:
        idx = idx + (slice(None, None),) * (3 - len(idx))
    return idx[0], idx[1:3]


class LCSR:
    """LCSR: List of Compressed Sparse Row matrices

    This object is useful for doing maths with several csr-matrices at once.
    In particular this addresses two weaknesses in sisl: 
    1. Math with two SparseCSRs is unreasonably slow.
    2. Indexing SparseCSRs isn't really fancy, and it always results in dense results.

    You can use LCSR like so, for example:

    >>> dHS = HSnew.copy(); dHS._csr = (LCSR(dHS._csr) - LCSR(HSold._csr)).tosisl()

    For larger SparseCSRs this can be orders of magnitude faster then the direct
    sisl way `dHS = HSnew - HSold`. It is also fast with small systems.
    Depending on the exact system, conversion back to sisl can take a few seconds however.

    See also `LSpGeom` for a class that combines the LCSR with other spgeom information,
    so that you won't have to directly set the `_csr` attribute on sisl spgeoms manually.

    LCSR also supports index operations where the first parameter is the `dim`.
    Note the this index is a bit special, as it is always outer indexing.
    The other two support the full fancy indexing via scipy.

    Parameters
    ----------
    obj : list or tuple of scipy.sparse.csr_matrix OR sisl.SparseCSR
    """
    def __init__(self, obj):
        if isinstance(obj, (tuple, list)):
            self._csrs = list(obj)
        elif isinstance(obj, si.SparseCSR):
            self._csrs = [obj.tocsr(i) for i in range(obj.dim)]
        else:
            raise TypeError("Pass a list, tuple or sisl csr.")
        self._kwargs = dict()

    @property
    def dim(self):
        return len(self._csrs)

    def explicit_diagonal(self):
        """In-place operation: Ensure diagonal is explicitly set."""
        idxdiag = np.arange(min(self._csrs[0].shape))
        with warnings.catch_warnings():
            # Scipy will give SparseEfficiencyWarning, and advices using lil-matrix
            # for sparsity mutation. However, lil-matrix does not support explicit
            # zeros. So we have to do it like this.
            warnings.simplefilter("ignore")
            if SCIPYVERSION >= (1, 4):
                for m in self._csrs:
                    m[idxdiag, idxdiag] = m[idxdiag, idxdiag]
            else:  # smart indexing not working so well
                for m in self._csrs:
                    m[idxdiag, idxdiag] = np.squeeze(np.asarray(m[idxdiag, idxdiag]))

    def tosisl(self, safediag=False):
        if safediag:
            self.explicit_diagonal()
        return si.SparseCSR.fromsp(*self._csrs)

    def __getitem__(self, index):
        sidx, cidx = _upcast_3index(index)
        if isinstance(sidx, int):
            return self._csrs[sidx][cidx]
        elif isinstance(sidx, slice):
            return LCSR([csr[cidx] for csr in self._csrs[sidx]])
        else:
            return LCSR([self._csrs[i][cidx] for i in sidx])

    def __setitem__(self, index, value):
        sidx, cidx = _upcast_3index(index)
        bc_sidx = False
        if isinstance(value, sparse.csr_matrix):
            bc_sidx = True
            value = LCSR([value])
        if isinstance(sidx, int):
            if len(value._csrs) != 1:
                raise ValueError("Dimension mismatch")
            self._csrs[sidx][cidx] = value._csrs[0]
            return

        if isinstance(sidx, slice):
            selfcsrs = self._csrs[sidx]
            valcsrs = value._csrs
            if bc_sidx:
                valcsrs = valcsrs * len(selfcsrs)
            for scsr, ocsr in zip(selfcsrs, valcsrs):
                scsr[cidx] = ocsr
        else:
            vcsrs = value._csrs if not bc_sidx else value._csrs * len(sidx)
            for i, ocsr in zip(sidx, vcsrs):
                self._csrs[i][cidx] = ocsr

    def eliminate_zeros(self, atol=0):
        for csr in self._csrs:
            if atol > 0:
                csr.data[np.abs(csr.data) < atol] = 0
            csr.eliminate_zeros()

    def _init_child(self, *args, **kwargs):
        ukwargs = self._kwargs.copy()
        ukwargs.update(kwargs)
        return type(self)(*args, **ukwargs)

    def copy(self):
        return self._init_child([csr.copy() for csr in self._csrs])

    def multiply(self, other):
        # * operator for sparse matrices is matmat mul (annoyingly). Provide mul
        if isinstance(other, LCSR):
            it = zip_longest(self._csrs, other._csrs)
        else:
            it = zip(self._csrs, itertools.repeat(other))
        return self._init_child([m1.multiply(m2).tocsr() for m1, m2 in it])


def _LCSR_binop(op, swap=False):
    order_operands = lambda a, b: (a, b)
    if swap:
        order_operands = lambda a, b: (b, a)
    @wraps(op)
    def _op(self, other):
        if isinstance(other, LCSR):
            if self.dim != other.dim:
                raise ValueError(f"{self} and {other} have mismatching dims!")
            it = zip(self._csrs, other._csrs)
        elif np.isscalar(other):
            it = zip(self._csrs, itertools.repeat(other))
        else:
            raise NotImplementedError
        return self._init_child([op(*order_operands(s, o)) for s, o in it])
    return _op


_binops = {
    "add", "radd", "iadd",
    "sub", "rsub", "isub",
    "mul", "rmul", "imul",
    "truediv", "rtruediv", "itruediv",
}
for bop in _binops:
    swap = bop.startswith("r")
    cbop = bop[1:] if swap else bop
    setattr(LCSR, f"__{bop}__", _LCSR_binop(getattr(operator, cbop), swap=swap))


class LSpGeom(LCSR):
    """LSPGeom : List-based sparse geometry

    This is a wrapper for LCSR that also contains and preserves other
    spgeom information (geometry, spin). The `tosisl()` method then
    gives directly the right spgeom object instead of "just" the SparseCSR.

    Example:
    >>> dHS = (LSpGeom(HSnew) - LSpGeom(HSold)).tosisl()

    Parameters
    ----------
    obj : sisl.SparseOrbital
    """
    def __init__(self, obj, **kwargs):
        if isinstance(obj, si.SparseOrbital):
            super().__init__(obj._csr)
        else:
            super().__init__(obj)
        self._kwargs = kwargs

        def maybeset(k, v):
            kwargs[k] = kwargs.get(k, v)

        if isinstance(obj, si.SparseOrbital):
            maybeset("spgeom_type", type(obj))
            maybeset("geometry", obj.geometry)
            maybeset("orthogonal", obj.orthogonal)
            if hasattr(obj, "spin"):
                maybeset("spin", obj.spin)
        if "spgeom_type" not in self._kwargs:
            raise ValueError("LSpGeom needs a spgeom type")
        if "geometry" not in self._kwargs:
            raise ValueError("LSpGeom needs a geometry")
        if "orthogonal" not in self._kwargs:
            raise ValueError("LSpGeom needs to know if its orthogonal")

    def tosisl(self, safediag=False):
        if safediag:
            self.explicit_diagonal()
        ukwargs = self._kwargs.copy()
        kind = ukwargs.pop("spgeom_type")

        geometry = ukwargs.pop("geometry")
        P = self._csrs
        S = None
        if not ukwargs.pop("orthogonal"):
            S = P[-1]
            P = P[:-1]
        return kind.fromsp(geometry=geometry, P=P, S=S, **ukwargs)
