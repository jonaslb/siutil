import numpy as np
import sisl as si
from siutil.geomutil import geom_tile_from_matrix, geom_remove_dupes_pbc, geom_uc_match, geom_sc_match, geom_uc_wrap, geom_sc_geom, geom_periodic_match_geom
import pytest
from scipy.spatial.distance import cdist


def test_gtile_matrix():
    g = si.geom.graphene()
    gsq = geom_tile_from_matrix(g, [[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
    assert gsq.na == 4
    fxyz0 = g.fxyz
    celloffset, fxyz = np.divmod(gsq.xyz.dot(g.sc.icell.T), 1)
    fxyz[np.isclose(fxyz, 1, atol=1e-11)] = 0
    eq = cdist(fxyz0, fxyz) < 1e-3
    assert np.allclose(np.sum(eq, axis=0), 1)  # each atom (4) only corresponds to one original
    assert np.allclose(np.sum(eq, axis=1), 2)  # each atom (2) corresponds to itself and a 'copy'


def test_geom_uc_match():
    g = si.geom.graphene()
    g2 = g.move(g.xyz[1])
    
    ucm = geom_uc_match(g, g)
    assert np.allclose(ucm, [[0, 0], [1, 1]])  # 0=>0 and 1=>1

    ucm = geom_uc_match(g, g2)
    assert np.allclose(ucm, np.array([[1, 0]]))  # 1=>0


def test_scgeom():
    g = si.geom.graphene()
    gsc = geom_sc_geom(g)
    sc_off = g.sc.sc_off
    acells, fxyz1 = np.divmod(gsc.xyz.dot(g.sc.icell.T), 1)
    acells[np.isclose(fxyz1, 1)] += 1

    for i, acell in enumerate(acells):
        isc_off = i // 2
        assert np.allclose(acell, sc_off[isc_off])


def test_geom_periodic_match_geom():
    g2 = si.geom.graphene()
    g1 = g2.sub(0)
    gsc = geom_sc_geom(g1)
    uca, sca, offsets = geom_periodic_match_geom(g2, gsc, (0, 0), ret_cell_offsets=True)
    assert np.allclose(offsets, g1.sc.sc_off)
    assert np.allclose(uca, 0)
    assert np.allclose(sca, np.arange(9))



