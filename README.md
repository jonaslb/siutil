# Some Sisl utils (`siutil`)

These are some extra utility functions that are built around the (Sisl)[github.com/zerothi/sisl] package.
They might not exactly fit into Sisl or need further work before being moved there, but since they are still useful, I thought they deserved their own repository.
Any sort-of-general-purpose utils that might not fit into sisl or could just be 'litter' there are welcome here.

The functions are still subject to significant change.

## Quick overview

### Atoms utils

| Function             | Description                                   |
|:-------------------- |:--------------------------------------------- |
| `atoms_unique_match(atoms0, atoms1)` | Return a list of two-tuples of matching specie in atoms0 and atoms1 |
| `atoms_match(atoms0, atoms1)` | Return a bool matrix_ij for whether atoms0[i]==atoms1[j] (all atoms in a geom). |

### Geometry utils

| Function             | Description                                   |
|:-------------------- |:--------------------------------------------- |
| `geom_tile_from_matrix(geom, tile)` | Choose a new periodicity for a geometry. `tile` is a matrix where each row represents the linear combination of old lattice vectors that form a new lattice vector. Must be integers. |
| `geom_uc_match(geom0, geom1)` | For each atom in the unit cell of geom0, a list of unit cell atom indices in geom1 that are identical (coordinate, specie) is returned. So if `geom0 is geom1`, you will get `[[0], [1], [2], ...]` (up to the number of unit cell atoms). |
| `geom_sc_match(geom0, geom1)` | Not implemented |

### Spgeom utils (hamiltonian etc)

| Function             | Description                                   |
|:-------------------- |:--------------------------------------------- |
| `spgeom_tile_from_matrix(spgeom, tile)` | Not (fully) implemented |
| `spgeom_transfer_periodic(spfrom, spto, pair)` | Not implemented |
