import sisl as si
from sisl.utils.ranges import array_arange
import scipy.sparse as ssp
from itertools import starmap
import numpy as np
from .geomutil import (
    geom_periodic_match_geom,
    geom_tile_from_matrix,
    geom_sc_geom,
    geom_uc_match,
    geom_sc_match,
    geom_uc_wrap,
)


def spgeom_wrap_uc(spgeom):
    """Wrap any atoms outside the unit cell back into the unit cell. Matrix elements 'make sense',
    ie. unit cell couplings reaching outside the uc become super cell couplings (and vice versa)."""
    new_geom = geom_uc_wrap(spgeom.geometry)
    kwargs = dict(orthogonal=spgeom.orthogonal)
    if hasattr(spgeom, "spin"):
        kwargs["spin"] = spgeom.spin
    newspgeom = spgeom.__class__(new_geom, **kwargs)
    newspgeom._orthogonal = spgeom.orthogonal
    spgeom_transfer_periodic(spgeom, newspgeom, (0, 0))
    return newspgeom


def spgeom_transfer_to_sub(
    spfrom, spto, pair, only_dim=None, match_specie=True, op="assign"
):
    """Copy all matrix elements from spfrom to spto in the place specified with pair; pair should
    be an atom in each of the spgeoms that match. The geometries are then rigidly matched from
    there, and on all matches (species and location) the matrix elements are transferred."""
    gfrom = spfrom.geometry.move(
        spto.geometry.xyz[pair[1]] - spfrom.geometry.xyz[pair[0]]
    )

    sc0m, sc1m = geom_sc_match(gfrom, spto.geometry, match_specie=match_specie).T
    uc0m = sc0m[sc0m < gfrom.na]
    uc1m = sc1m[sc1m < spto.na]

    # TODO: probably faster by using spto.spsame(spfrom) and then using csr-methods?
    spgeom_transfer_outeridx(
        spfrom,
        spto,
        uc0m,
        sc0m,
        uc1m,
        sc1m,
        atomic_indices=True,
        only_dim=only_dim,
        op=op,
    )

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


def spgeom_transfer_outeridx(
    spfrom,
    spto,
    from_left,
    from_right,
    to_left,
    to_right,
    atomic_indices=False,
    only_dim=None,
    op="assign",
):
    """Akin to `spto[to_left, to_right] = spfrom[from_left, from_right]`, but where 'outer indexing' is understood.
    Ie. if spto was a 2d numpy array, then spto[to_left, to_right] would be a 'sub' 2d array.
    If `atomic_indices=True` (default false), the indices are translated to orbitals first.
    """
    from_left, from_right, to_left, to_right = map(
        np.ravel, (from_left, from_right, to_left, to_right)
    )
    if atomic_indices:
        from_left = spfrom.a2o(from_left, all=True)
        to_left = spto.a2o(to_left, all=True)
        from_right = spfrom.a2o(from_right, all=True)
        to_right = spto.a2o(to_right, all=True)
    if op == "assign":
        for from_row, to_row in zip(from_left, to_left):
            if only_dim is None:
                spto[to_row, to_right] = spfrom[from_row, from_right]
            else:
                spto[to_row, to_right, only_dim] = spfrom[
                    from_row, from_right, only_dim
                ]
    elif op == "add":
        for from_row, to_row in zip(from_left, to_left):
            if only_dim is None:
                spto[to_row, to_right] += spfrom[from_row, from_right]
            else:
                spto[to_row, to_right, only_dim] += spfrom[
                    from_row, from_right, only_dim
                ]
    elif op == "subtract":
        for from_row, to_row in zip(from_left, to_left):
            if only_dim is None:
                spto[to_row, to_right] -= spfrom[from_row, from_right]
            else:
                spto[to_row, to_right, only_dim] -= spfrom[
                    from_row, from_right, only_dim
                ]
    else:
        raise ValueError(f"Invalid op {op}")


def _spgeom_transfer_vary_col_assign(spfrom, spto, from_left, from_right, to_left, to_right, atomic_indices=False, only_dim=None, op='assign'):
    """Similar to spgeom_transfer_outeridx but the `right` indices are list of arrays corresponding to different column assignments for each row.
    Can be used only to assign non-zero values. Useful if you want to preserve sparsity.
    """
    raise NotImplementedError("doesnt work, at least with transfer_periodic for unknown reason")
    if atomic_indices:
        from_left = spfrom.a2o(from_left, all=True)
        to_left = spto.a2o(to_left, all=True)
        from_right = [spfrom.a2o(frr, all=True) for frr in from_right]
        to_right = [spto.a2o(tor, all=True) for tor in to_right]
    iterator = zip(from_left, to_left, from_right, to_right)

    if op == 'assign':
        for from_row, to_row, from_cols, to_cols in iterator:
            if only_dim is None:
                spto[to_row, to_cols] = spfrom[from_row, from_cols]
            else:
                spto[to_row, to_cols, only_dim] = spfrom[from_row, from_cols, only_dim]
    elif op == 'add':
        for from_row, to_row, from_cols, to_cols in iterator:
            if only_dim is None:
                spto[to_row, to_cols] += spfrom[from_row, from_cols]
            else:
                spto[to_row, to_cols, only_dim] += spfrom[from_row, from_cols, only_dim]
    elif op == 'subtract':
        for from_row, to_row, from_cols, to_cols in iterator:
            if only_dim is None:
                spto[to_row, to_cols] -= spfrom[from_row, from_cols]
            else:
                spto[to_row, to_cols, only_dim] -= spfrom[from_row, from_cols, only_dim]
    else:
        raise ValueError(f"Invalid op {op}")


def spgeom_transfer_periodic(spfrom, spto, pair, op="assign"):
    """Copy all the matrix elements from spfrom to spto in places where spto correspond to periodic
    repetitions of spfrom. You must provide a `pair`, being a two-tuple consisting of an index from
    each of the two sparse geometries that match (eg. `(0, 0)` if the first atoms are the same)."""
    # The method increases sparsity somewhat as it does block-transfer (outer idx).
    # If spto is tiled/repeated of spfrom, then spto will end up "block dense".
    # This might be inefficient.
    # An efficient implementation could be possible with (scipy) csr matrices.
    gfrom = spfrom.geometry.move(
        spto.geometry.xyz[pair[1]] - spfrom.geometry.xyz[pair[0]]
    )
    gtosc = geom_sc_geom(spto.geometry)

    # Match from_uc to to_sc
    afrom, ato, offsets = geom_periodic_match_geom(
        gfrom, gtosc, pair, ret_cell_offsets=True
    )
    # Todo: Only do uc-uc and calculate the uc-sc directly (should give speedup)

    def to_cell_dict(a_frto, offsets):
        cellmatches = np.unique(offsets, axis=0)
        mdict = dict()
        for cm in cellmatches:
            key = cm.astype(int).tobytes()
            idces = np.flatnonzero(np.all(cm == offsets, axis=1))
            mdict[key] = a_frto[:, idces]
        return mdict

    # All the matches from uc to sc
    sca_match = np.vstack((afrom, ato))
    d_sc_match = to_cell_dict(sca_match, offsets)

    # All the matches from uc to uc
    uca_match_filter = np.flatnonzero(sca_match[1, :] < spto.na)
    uca_match = sca_match[:, uca_match_filter]
    offsets_uca = offsets[uca_match_filter, :]
    d_uc_match = to_cell_dict(uca_match, offsets_uca)

    # Prepare edges (to preserve some sparsity after transfer)
    fr_edges = [set(spfrom.edges(atom=i)) for i in range(spfrom.na)]

    # For each uc-uc cell match, the matches are LEFT side atomic indices
    # To obtain RIGHT side atomic indices, use sc_off for gfrom in combination with
    # uc-sc matches to obtain the neighboring places.
    for uc_off_k, afrto_l in d_uc_match.items():
        uc_off = np.frombuffer(uc_off_k, dtype=int)
        afrto_r = list()
        for i, sc_off in enumerate(gfrom.sc.sc_off.astype(int)):
            sc_uc_off = uc_off + sc_off
            afrto_here = d_sc_match[sc_uc_off.tobytes()].copy()
            afrto_here[0, :] += i * gfrom.na
            afrto_r.append(afrto_here)
        afrto_r = np.hstack(afrto_r)

        afr_r, ato_r = afrto_r
        afr_l, ato_l = afrto_l

        # # Create afr_r and ato_r for each row (remove zeros)
        # # TODO: Using the vary col assign is broken :(
        # # TODO: Is doing it by orbital faster?
        # # TODO: Vectorize (numpyize) the loops
        # afr_rf = [[] for _ in afr_l]
        # ato_rf = [[] for _ in ato_l]
        # for i, iafr_l in enumerate(afr_l):
        #     edges = fr_edges[iafr_l]
        #     for afrr, ator in zip(afr_r, ato_r):
        #         if afrr in edges:
        #             afr_rf[i].append(afrr)
        #             ato_rf[i].append(ator)

        # # these should be identical to using outeridx but dont work
        # afr_rf = [afr_r] * len(afr_l)  
        # ato_rf = [ato_r] * len(afr_l)

        # _spgeom_transfer_vary_col_assign(
        #     spfrom, spto, afr_l, afr_rf, ato_l, ato_rf, atomic_indices=True, op=op
        # )
        spgeom_transfer_outeridx(spfrom, spto, afr_l, afr_r, ato_l, ato_r, atomic_indices=True, op=op)
    return  # inplace operation



    # For each uc-uc cell match, the matches are LEFT side atomic indices
    # To obtain RIGHT side atomic indices, use sc_off for gfrom in combination with
    # uc-sc matches to obtain the neighboring places.
    prog = lambda x: x
    if progress and tqdm:
        prog = tqdm
    for uc_off_k, (afr_l, ato_l) in prog(d_uc_match.items()):
        uc_off = np.frombuffer(uc_off_k, dtype=int)
        afromto = list()
        for sc_off in gfrom.sc.sc_off.astype(int):
            sc_uc_off = uc_off + sc_off
            afromto.append(d_sc_match[sc_uc_off.tobytes()])
        afr_r, ato_r = np.hstack(afromto)

        


def spgeom_tile_from_matrix(spgeom, tile):
    """Choose a new periodicity for a sparse geometry. `tile` is a matrix where each row represents
    the linear combination of old lattice vectors that form a new lattice vector.
    Must be integers."""
    gtiled = geom_tile_from_matrix(spgeom.geometry, tile)
    kwargs = dict(orthogonal=spgeom.orthogonal)
    if hasattr(spgeom, "spin"):
        kwargs["spin"] = spgeom.spin
    spg = spgeom.__class__(gtiled, **kwargs)
    spgeom_transfer_periodic(spgeom, spg, (0, 0))
    spg._orthogonal = spgeom.orthogonal
    spg._reset()
    return spg


def tprint(*args, **kwargs):
    from datetime import datetime

    print(f"{datetime.now():%H:%M:%S}:", *args, **kwargs)


def spgeom_lrsub(spgeom, left, right, geom="left", can_finalize=True):
    """'cross-sub' a spgeom. Result is a new spgeom. Does not necessarily make sense except for very particular cases."""

    if geom == "left":
        geom = spgeom.geometry.sub(left)
    elif geom == "right":
        geom = spgeom.geometry.sub(right)
    elif isinstance(geom, si.Geometry):
        pass
    else:
        raise TypeError(
            "Invalid geometry, must refer to 'left' or 'right' or be a sisl.Geometry"
        )

    left = spgeom.a2o(left, all=True)
    right = spgeom.a2o(spgeom.auc2sc(right), all=True)

    # Perform on stack of csr-matrices
    if can_finalize:
        spgeom.finalize()
    csrs = []
    for nd in range(spgeom.dim):
        if can_finalize:
            csr = spgeom._csr
            # no-copy version of sisl.tocsr
            csr = ssp.csr_matrix((csr._D[:, nd], csr.col, csr.ptr), shape=csr.shape[:2])
            # Copy after sub instead (see scipy gh #11255 for indexing)
            csrs.append(csr[left, :][:, right].copy())
        else:
            csrs.append(spgeom.tocsr(dim=nd))

    kwargs = dict()
    if hasattr(spgeom, "spin"):
        kwargs["spin"] = spgeom.spin
    if spgeom.orthogonal:
        newsp = spgeom.fromsp(geom, P=csrs, **kwargs)
    else:
        newsp = spgeom.fromsp(geom, P=csrs[:-1], S=csrs[-1], **kwargs)
    return newsp
