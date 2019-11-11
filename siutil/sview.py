from ase.visualize import view as aview


def sview(g, **kwargs):
    aview(g.toASE(), **kwargs)
