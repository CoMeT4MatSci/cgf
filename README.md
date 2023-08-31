# Coarse Grained Frameworks (cgf)

cgf enables the coarse-graining of framework materials based on a Bond-Angle Force-Field (BAFF) or based on a Mikado Model.

## Description of the coarse-grained system (`cg_atoms`)
As basis, we use the atomic simulation environment (ASE). The coarse-grained system is described by the sites of the cores as pseudo-atoms in an ASE atoms-object (`cg_atoms`).
`cg_atoms` contains additionally as arrays the following information:
- `neighbor_ids`: The IDs of the neighbors of each core-site
- `neighbor_distances`:  The distances to the respective neighbors of each core-site
- `linker_sites`: Vectors which indicate position of attatched linkesr for each core-site
- `linker_neighbors`: Provides a mapping which `linker_sites` are connected to which neighbors