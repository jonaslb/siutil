import numpy as np
from scipy.spatial.distance import cdist
from .atomutil import atoms_match


def geom_tile_from_matrix(geom, tile):
    """Choose a new periodicity for a geometry. `tile` is a matrix where each row represents the
    linear combination of old lattice vectors that form a new lattice vector. Must be integers."""
    tile = np.array(tile, dtype=int)
    assert tile.shape == (3, 3)
    newcell = tile.dot(geom.sc.cell)

    # Decide new nsc by looking at the supercell dimensions before/after (solid method?)
    inewcell = np.linalg.inv(newcell)
    super_vectors = (geom.sc.nsc // 2)[:, None] * geom.sc.cell
    new_nsc = np.ceil(np.max(np.abs(super_vectors.dot(inewcell)), axis=1)) * 2 + 1

    # Now what atoms to include... This part of the algorithm is inefficient. TODO to enhance.
    tile_test = np.sum(np.abs(tile), axis=0)  # gives more than we need for sure
    g = geom.copy()
    for ax, t in enumerate(tile_test):
        g = g.tile(t, ax)
    g.set_supercell(newcell)
    g.set_nsc(new_nsc)
    # Fold atoms into the new cell
    g.xyz -= np.floor(g.fxyz).dot(g.cell)
    # Remove duplicates
    g = geom_remove_dupes_pbc(g)
    return g


def geom_remove_dupes_pbc(geom, eps=1e-3):
    """Get a geom similar to given geom but without any pbc dupes.
    This is a simplistic function that only works for neighboring cells.
    """
    na0 = geom.na
    gt = geom.tile(2, 0).tile(2, 1).tile(2, 2)
    dupes = np.nonzero(
        np.linalg.norm(gt.xyz.reshape(-1, 1, 3) - gt.xyz.reshape(1, -1, 3), axis=2)
        < eps
    )
    dupes = np.array(dupes) % na0
    dupes = list(set(ia1 for ia0, ia1 in dupes.T if ia0 < ia1))
    return geom.remove(dupes)


def geom_uc_match(geom0, geom1, match_specie=True):
    """Returns an nx2 matrix where n in number of matches and col 1 is idx match in g0 and col 2 is
    idx match in g1."""
    if match_specie:
        samespecie = atoms_match(geom0.atoms, geom1.atoms)

    # TODO: Only calc distances where atoms are same specie, also allow taking precomputed samespecie
    isclose = (
        cdist(geom0.xyz, geom1.xyz) < 1e-3
    )
    if match_specie:
        match = np.logical_and(isclose, samespecie)
    else:
        match = isclose
    match = np.array(np.nonzero(match))
    return match.T


def geom_sc_match(geom0, geom1, match_specie=True):
    """Returns an nx2 matrix where n in number of matches and col 1 is idx match in g0 sc and col 2
    is idx match in g1 sc."""
    g0sc = geom_sc_geom(geom0)
    g1sc = geom_sc_geom(geom1)
    return geom_uc_match(g0sc, g1sc, match_specie=match_specie)


def geom_uc_wrap(geom):
    """Wrap any atoms outside the unit cell into the unit cell."""
    g = geom.copy()
    g.xyz -= np.floor(g.fxyz).dot(g.sc.cell)
    return g


def geom_sc_geom(geom, uc_lowerleft=True, wrap=False):
    """Return a geometry where the unit cell is the supercell of the given geometry
    (incl. ordering). Works for spgeom as well.
    """
    nsc = np.array(geom.sc.nsc)
    na = geom.na
    # Tile the geom out
    g = geom.copy()
    for ia, nt in enumerate(nsc):
        g = g.tile(nt, ia)
    # Reorder according to sc indices
    lowerleft = -(nsc - 1) / 2
    tosub = []
    for offset in geom.sc.sc_off:
        t = offset - lowerleft
        start = t[0] * na + nsc[0] * t[1] * na + nsc[0] * nsc[1] * t[2] * na
        tosub.append(np.arange(int(start), int(start + na)))
    g = g.sub(np.concatenate(tosub))
    if uc_lowerleft:
        g = g.move(geom.xyz[0] - g.xyz[0])
    if wrap:
        g = geom_uc_wrap(g)
    return g


def geom_periodic_match_geom(unitg, superg, pair, ret_cell_offsets=False):
    """Given a unit geometry (unitg), a larger geometry (superg) and pair (eg. (0,1)),
    return a list of pairs of atoms, where the first is a unitg atom and the second is a periodic
    repetition of that atom in superg (wrt unitg cell)"""
    ua, sa = pair
    superg = superg.move(unitg.xyz[ua] - superg.xyz[sa])

    # TODO: Don't use a mega-array
    dr = (superg.xyz.reshape(1, -1, 3) - unitg.xyz.reshape(-1, 1, 3)).dot(unitg.icell.T)
    dcell, dr = np.divmod(dr, 1)
    rounding = np.isclose(dr, 1).nonzero()
    dcell[rounding] += 1
    dr[rounding] = 0
    uca, sca = np.logical_and.reduce(np.isclose(dr, 0), axis=2).nonzero()

    if ret_cell_offsets:
        return uca, sca, dcell[uca, sca]
    return uca, sca
