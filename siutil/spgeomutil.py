import numpy as np
from .geomutil import geom_tile_from_matrix


def spgeom_transfer_periodic(spfrom, spto, pair):
    """Copy all the matrix elements from spfrom to spto in places where spto correspond to periodic
    repetitions of spfrom. You must provide a `pair`, being a two-tuple consisting of an index from
    each of the two sparse geometries that match (eg. `(0, 0)` if the first atoms are the same)."""
    gfrom = spfrom.geom.move(spto.geom.xyz[pair[1]] - spfrom.geom.xyz[pair[0]])
    for iaold in range(len(gfrom)):
        gtotmp = spto.geom.move(-gfrom.xyz[iaold])
        gtof = np.dot(gtotmp.xyz, gfrom.icell.T)
        images_in_new_uc = np.flatnonzero(np.linalg.norm(gtof % 1, axis=1) < 1e-3)
        # images_in_new_sc = gtotmp.auc2sc(images_in_new_uc)
        for ianew in images_in_new_uc:
            # Now match the SUPERCELS of the new and translated old geoms
            # After that we should be able to copy elements
            pass
            # todo


def spgeom_tile_from_matrix(spgeom, tile):
    """Choose a new periodicity for a sparse geometry. `tile` is a matrix where each row represents
    the linear combination of old lattice vectors that form a new lattice vector.
    Must be integers."""
    gtiled = geom_tile_from_matrix(spgeom.geom, tile)
    spg = spgeom.__class__(gtiled, dim=spgeom.dim)
    spgeom_transfer_periodic(spgeom, spg, (0, 0))
    spg._orthogonal = spgeom.orthogonal
    spg._reset()
    return spg
