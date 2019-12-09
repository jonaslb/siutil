import sisl as si
from ase.visualize import view as aview
import warnings


def sview(g, ghosts_z=None, **kwargs):
    if ghosts_z is not None:
        g = g.copy()
        cw = warnings.catch_warnings()
        cw.__enter__()
        warnings.simplefilter("ignore")
        replacement = si.Atom(ghosts_z)
        for a in g.atoms._atom:
            if a.Z < 1:
                g.atoms.replace(a, replacement)
        g.atoms.reduce(in_place=True)
        cw.__exit__()
    aview(g.toASE(), **kwargs)
