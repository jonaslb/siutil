import numpy as np
from scipy import sparse
import sisl as si
from siutil.listlike_csr import (
    LCSR,
    LSpGeom
)

def _allclose_csr(a, b, rtol=1e-6, atol=1e-12):
    c = (abs(a-b) - rtol * abs(b)).data
    return np.all(c <= atol)

def test_lcsr_binop():
    dim = 3
    mats1 = [sparse.random(100,100,0.1).tocsr() for _ in range(dim)]
    mats2 = [sparse.random(100,100,0.1).tocsr() for _ in range(dim)]
    lcsr1 = LCSR([m.copy() for m in mats1])
    lcsr2 = LCSR([m.copy() for m in mats2])

    mt = [m1 + m2 for m1, m2 in zip(mats1, mats2)]
    lcsrt = lcsr1 + lcsr2
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))

    mt = [m1 - m2 for m1, m2 in zip(mats1, mats2)]
    lcsrt = lcsr1 - lcsr2
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))

    lcsrt = lcsr1.copy()
    lcsrt -= lcsr2
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))

    mt = [m1 - m2 for m1, m2 in zip(mats1, mats1)]
    lcsrt = lcsr1 - lcsr1
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))

    lcsrt = lcsr1.copy()
    lcsrt -= lcsr1
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))

    scalar = np.pi
    mt = [m1 * scalar for m1 in mats1]
    lcsrt = lcsr1 * scalar
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))
    lcsrt = lcsr1.copy()
    lcsrt *= scalar
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, lcsrt._csrs))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i] for i in range(dim)]))
    assert all(_allclose_csr(md, ml) for md, ml in zip(mt, [lcsrt[i, :, :] for i in range(dim)]))

def test_lspgeom_sameassisl():
    Hs = si.Hamiltonian(si.geom.graphene(), orthogonal=True, spin=si.Spin.POLARIZED)
    dim = Hs.dim
    assert dim == 2
    Hs.set_nsc((7, 7, 1))
    Hs.construct([[0.1, 1.5, 3, 5, 9], [-2, 1, 0.1, 0.01, 0.001]])
    Hs = Hs.tile(4, 0).tile(4, 1)
    Hs.finalize()
    LHs = LSpGeom(Hs)

    HS2 = Hs + Hs
    LHs2 = LHs + LHs
    assert all(_allclose_csr(HS2.tocsr(i), LHs2[i]) for i in range(dim))

    Hzero = Hs - Hs
    LHszero = LHs - LHs
    assert all(_allclose_csr(Hzero.tocsr(i), LHszero[i]) for i in range(dim))
    Hzero.eliminate_zeros(atol=1e-3)
    LHszero.eliminate_zeros(atol=1e-3)
    assert all(_allclose_csr(Hzero.tocsr(i), LHszero[i]) for i in range(dim))

    Hnz = Hs - 0.5 * Hs
    Hnz2 = Hnz.copy()
    LHsnz = LHs - 0.5 * LHs
    LHsnz2 = LHsnz.copy()
    assert all(_allclose_csr(Hnz.tocsr(i), LHsnz[i]) for i in range(dim))
    Hnz.eliminate_zeros(atol=1e-2)
    LHsnz.eliminate_zeros(atol=1e-2)
    assert all(_allclose_csr(Hnz.tocsr(i), LHsnz[i]) for i in range(dim))
    diff1 = Hnz2 - Hnz
    diff2 = LHsnz2 - LHsnz
    assert all(_allclose_csr(diff1.tocsr(i), diff2[i], rtol=1e-10) for i in range(dim))

def test_lspgeom_sislconvert():
    Hs = si.Hamiltonian(si.geom.graphene(), orthogonal=True, spin=si.Spin.POLARIZED)
    dim = Hs.dim
    assert dim == 2
    Hs.set_nsc((7, 7, 1))
    Hs.construct([[0.1, 1.5, 3, 5, 9], [-2, 1, 0.1, 0.01, 0.001]])
    Hs = Hs.tile(4, 0).tile(4, 1)
    Hs.finalize()
    LHs = LSpGeom(Hs)

    Hs2 = LHs.tosisl()
    assert all(_allclose_csr(Hs.tocsr(i), Hs2.tocsr(i)) for i in range(dim))
    assert Hs.geometry.equal(Hs2.geometry)

    LHs += LHs * 0.3
    Hs += Hs * 0.3
    Hs2 = LHs.tosisl()
    assert all(_allclose_csr(Hs.tocsr(i), Hs2.tocsr(i)) for i in range(dim))
    assert Hs.geometry.equal(Hs2.geometry)

