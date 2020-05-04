import sisl as si
from itertools import starmap
import numpy as np
from .geomutil import (
    geom_tile_from_matrix, geom_sc_geom, geom_uc_match, geom_sc_match, geom_uc_wrap
    )


def spgeom_wrap_uc(spgeom):
    """Wrap any atoms outside the unit cell back into the unit cell. Matrix elements 'make sense',
    ie. unit cell couplings reaching outside the uc become super cell couplings (and vice versa)."""
    new_geom = geom_uc_wrap(spgeom.geom)
    kwargs = dict(orthogonal=spgeom.orthogonal)
    if hasattr(spgeom, "spin"):
        kwargs["spin"] = spgeom.spin
    newspgeom = spgeom.__class__(new_geom, **kwargs)
    newspgeom._orthogonal = spgeom.orthogonal
    spgeom_transfer_periodic(spgeom, newspgeom, (0, 0))
    return newspgeom


def spgeom_transfer_to_sub(spfrom, spto, pair, only_dim=None, match_specie=True):
    """Copy all matrix elements from spfrom to spto in the place specified with pair; pair should
    be an atom in each of the spgeoms that match. The geometries are then rigidly matched from
    there, and on all matches (species and location) the matrix elements are transferred."""
    gfrom = spfrom.geom.move(spto.geom.xyz[pair[1]] - spfrom.geom.xyz[pair[0]])

    # uc0m, uc1m = geom_uc_match(gfrom, spto.geom).T
    sc0m, sc1m = geom_sc_match(gfrom, spto.geom, match_specie=match_specie).T
    uc0m = sc0m[sc0m < gfrom.na]
    uc1m = sc1m[sc1m < spto.na]

    spgeom_transfer_outeridx(spfrom, spto, uc0m, sc0m, uc1m, sc1m, atomic_indices=True, only_dim=only_dim)

    # for m0, m1 in zip(uc0m, uc1m):
    #     # These are the matched 'from' atoms (uc)
    #     m0os = gfrom.a2o(m0, all=True)
    #     m1os = spto.geom.a2o(m1, all=True)
    #     for sm0, sm1 in zip(sc0m, sc1m):
    #         # These are the matched 'to' atoms (sc)
    #         sm0os = gfrom.a2o(sm0, all=True)
    #         sm1os = spto.geom.a2o(sm1, all=True)
    #         for m0o, m1o in zip(m0os, m1os):
    #             spto[m1o, sm1os] = spfrom[m0o, sm0os]
    return  # inplace operation


def spgeom_transfer_outeridx(spfrom, spto, from_left, from_right, to_left, to_right, atomic_indices=False, only_dim=None):
    """Akin to `spto[to_left, to_right] = spfrom[from_left, from_right]`, but where 'outer indexing' is understood.
    Ie. if spto was a 2d numpy array, then spto[to_left, to_right] would be a 'sub' 2d array.
    If `atomic_indices=True` (default false), the indices are translated to orbitals first.
    """
    from_left, from_right, to_left, to_right = map(
        np.ravel, (from_left, from_right, to_left, to_right)
    )
    if atomic_indices:
        from_left = spfrom.a2o(from_left, all=True)
        to_left = spfrom.a2o(to_left, all=True)
        from_right = spfrom.a2o(from_right, all=True)
        to_right = spfrom.a2o(to_right, all=True)
    for from_row, to_row in zip(from_left, to_left):
        if only_dim is None:
            spto[to_row, to_right] = spfrom[from_row, from_right]
        else:
            spto[to_row, to_right, only_dim] = spfrom[from_row, from_right, only_dim]


def spgeom_transfer_periodic(spfrom, spto, pair):
    """Copy all the matrix elements from spfrom to spto in places where spto correspond to periodic
    repetitions of spfrom. You must provide a `pair`, being a two-tuple consisting of an index from
    each of the two sparse geometries that match (eg. `(0, 0)` if the first atoms are the same)."""
    gfrom = spfrom.geom.move(spto.geom.xyz[pair[1]] - spfrom.geom.xyz[pair[0]])

    gfromsc = geom_sc_geom(gfrom)
    gtosc = geom_sc_geom(spto.geom)

    for iaold in range(len(gfrom)):
        # For every atom in spfrom, find the periodic repetitions in spto
        # Todo use geomutil.geom_periodic_match_geom
        gtotmp = spto.geom.move(-gfrom.xyz[iaold])
        gtof = np.dot(gtotmp.xyz, gfrom.icell.T)
        # Note: small negative (eg 1e-17) becomes 1 when mod is taken
        # breakpoint()
        images_in_new_uc = np.flatnonzero(np.linalg.norm(np.abs(gtof) % (1-1e-15), axis=1) < 1e-3)
        # Orbitals on iaold
        io_old = spfrom.a2o(iaold, all=True)
        for ianew in images_in_new_uc:
            io_new = spto.a2o(ianew, all=True)
            # Now iaold and ianew are the same atom (save a unit cell translation).
            # Therefore we now need to match the supercells here and then transfer elements.
            gf = gfromsc.move(-gfromsc.xyz[iaold, :])
            gt = gtosc.move(-gtosc.xyz[ianew, :])
            gfm, gtm = geom_uc_match(gf, gt).T
            for match0, match1 in zip(gfm, gtm):
                # Need to only use a2o for one atom at a time to avoid reordering
                orb0 = spfrom.a2o(match0, all=True)
                orb1 = spto.a2o(match1, all=True)
                for o_old, o_new in zip(io_old, io_new):
                    spto[o_new, orb1] = spfrom[o_old, orb0]
    return  # inplace operation


def spgeom_tile_from_matrix(spgeom, tile):
    """Choose a new periodicity for a sparse geometry. `tile` is a matrix where each row represents
    the linear combination of old lattice vectors that form a new lattice vector.
    Must be integers."""
    gtiled = geom_tile_from_matrix(spgeom.geom, tile)
    kwargs = dict(orthogonal=spgeom.orthogonal)
    if hasattr(spgeom, "spin"):
        kwargs["spin"] = spgeom.spin
    spg = spgeom.__class__(gtiled, **kwargs)
    spgeom_transfer_periodic(spgeom, spg, (0, 0))
    spg._orthogonal = spgeom.orthogonal
    spg._reset()
    return spg


def spgeom_lrsub(spgeom, left, right, geom="left"):
    """'cross-sub' a spgeom. Result is a new spgeom. Does not necessarily make sense except for very particular cases."""
    if geom == "left":
        geom = spgeom.geom.sub(left)
    elif geom == "right":
        geom = spgeom.geom.sub(right)
    elif isinstance(geom, si.Geometry):
        pass
    else:
        raise ValueError()

    kwargs = dict(orthogonal=spgeom.orthogonal)
    if hasattr(spgeom, "spin"):
        kwargs["spin"] = spgeom.spin
    newsp = spgeom.__class__(geom, **kwargs)
    nsc = newsp.nsc

    left = spgeom.a2o(left, all=True)
    right = spgeom.a2o(right, all=True)
    to_l = np.arange(spgeom.no)
    to_r = np.arange(spgeom.no * nsc[0] * nsc[1] * nsc[2])
    spgeom_transfer_outeridx(spgeom, newsp, left, right, to_l, to_r)
    return newsp
