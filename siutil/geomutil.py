import numpy as np
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
    eps = 1e-4
    dupes = np.where(
        np.linalg.norm(g.xyz.reshape(-1, 1, 3) - g.xyz.reshape(1, -1, 3), axis=2) < eps)
    todelete = list(set(j for i, j in zip(*dupes) if j > i))
    g = g.remove(todelete)

    return g


def geom_uc_match(geom0, geom1):
    """For each atom in the unit cell of geom0, a list of unit cell atom indices in geom1 that
    are identical (coordinate, specie) is returned. So if `geom0 is geom1`, you will get
    `[[0], [1], [2], ...]` (up to the number of unit cell atoms)."""
    # TODO: compute distances for same species only, allow taking pre-computed samespecie matrix
    isclose = np.linalg.norm(geom0.xyz[:, None, :]-geom1.xyz[None, :, :], axis=2) < 1e-3
    samespecie = atoms_match(geom0.atom, geom1.atom)
    match = np.logical_and(isclose, samespecie)
    ret = [[] for _ in range(len(geom0))]
    for i, j in np.argwhere(match):
        ret[i].append(j)
    return ret


def geom_sc_match(geom0, geom1):
    """For each atom in the supercell of geom0, a list of supercell atom indices in geom1 that
    are identical (coordinate, specie) is returned. So if `geom0 is geom1`, you will get
    `[[0], [1], [2], ...]` (up to the number of supercell atoms)."""
    # Note: Supercell atoms in geom0 may be unit cell atoms in geom1, so we cant 'simply' use uc2sc
    # Perhaps iterate supercells in each geom
    raise NotImplementedError()
