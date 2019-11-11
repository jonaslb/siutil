# Some Sisl utils (`siutil`)

These are some extra utility functions that are built around the [Sisl](github.com/zerothi/sisl) package.
They might not exactly fit into Sisl or need further work before being moved there, but since they are still useful, I thought they deserved their own repository.
Any sort-of-general-purpose utils that might not fit into sisl or could just be 'litter' there are welcome here.

The functions are still subject to significant change.

## Quick overview

### Commandline and etc

| What | Description |
| :--- | :---------- |
| Command `sgui` | Similar to `ase gui` but loads the given files with sisl (ie. better file support) |
| `from siutil.sview import sview; sview(geom)` | Opens ase gui for the sisl geometry `geom`. |

### Atoms utils

| Function             | Description                                   |
|:-------------------- |:--------------------------------------------- |
| `atoms_unique_match(atoms0, atoms1)` | Return a list of two-tuples of matching specie in atoms0 and atoms1 |
| `atoms_match(atoms0, atoms1)` | Return a bool matrix_ij for whether atoms0[i]==atoms1[j] (all atoms in a geom). |

### Geometry utils

| Function             | Description                                   |
|:-------------------- |:--------------------------------------------- |
| `geom_uc_wrap(geom)` | Wrap any atoms outside the unit cell into the unit cell. |
| `geom_tile_from_matrix(geom, tile)` | Choose a new periodicity for a geometry. `tile` is a matrix where each row represents the linear combination of old lattice vectors that form a new lattice vector. Must be integers. |
| `geom_uc_match(geom0, geom1)` | Returns an nx2 matrix where n in number of matches and col 1 is idx match in g0 and col 2 is idx match in g1. |
| `geom_sc_match(geom0, geom1)` | Returns an nx2 matrix where n in number of matches and col 1 is idx match in g0 sc and col 2 is idx match in g1 sc. |
| `geom_sc_geom(geom, uc_lowerleft=True, wrap=False)` | Return a geometry where the unit cell is the supercell of the given geometry (incl. ordering). Works for spgeom as well. |

### Spgeom utils (hamiltonian etc)

| Function             | Description                                   |
|:-------------------- |:--------------------------------------------- |
| `spgeom_wrap_uc(spgeom)` | Wrap any atoms outside the unit cell back into the unit cell. Matrix elements 'make sense', ie. unit cell couplings reaching outside the uc become super cell couplings (and vice versa). |
| `spgeom_tile_from_matrix(spgeom, tile)` | Choose a new periodicity for a sparse geometry. `tile` is a matrix where each row represents the linear combination of old lattice vectors that form a new lattice vector.  Must be integers. |
| `spgeom_transfer_periodic(spfrom, spto, pair)` | Copy all the matrix elements from spfrom to spto in places where spto correspond to periodic repetitions of spfrom. You must provide a `pair`, being a two-tuple consisting of an index from each of the two sparse geometries that match (eg. `(0, 0)` if the first atoms are the same). |
