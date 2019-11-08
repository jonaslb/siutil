import numpy as np


def atoms_unique_match(atoms0, atoms1):
    """Return a list of two-tuples of matching specie in atoms0 and atoms1"""
    ua0 = atoms0._atom
    ua1 = atoms1._atom
    match = []
    for ia0, a0 in enumerate(ua0):
        try:
            match.append((ia0, ua1.index(a0)))
        except ValueError:
            # Atom is not in ua1
            pass
    return match


def atoms_match(atoms0, atoms1):
    """Return a bool matrix_ij for whether atoms0[i]==atoms1[j] (all atoms in a geom)."""
    umatch = np.array(atoms_unique_match(atoms0, atoms1))
    spec0 = atoms0.specie.copy()
    spec1 = atoms1.specie.copy()
    all_isect0 = umatch[:, 0]
    all_isect1 = umatch[:, 1]
    spec0[~np.in1d(spec0, all_isect0)] = -1
    spec1[~np.in1d(spec1, all_isect1)] = -2
    spec1_o = spec1.copy()
    for ispec0, ispec1 in umatch:
        spec1[np.where(spec1_o == ispec1)] = ispec0
    issame_matrix = (spec0[:, None] == spec1[None, :])
    return issame_matrix
