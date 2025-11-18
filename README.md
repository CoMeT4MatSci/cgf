[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15075281.svg)](https://doi.org/10.5281/zenodo.15075281)

# Coarse Grained Frameworks (cgf)

cgf enables the coarse-graining of framework materials based on a Bond-Angle Force-Field (BAFF) or on elastic beams (MikadoRR).

## Installation
```sh
pip install git+https://github.com/CoMeT4MatSci/cgf
```

## The core functionalities
- **Fitting a CG model** such as BAFF or MikadoRR (see [example 1](examples/example1/example1.ipynb))
- Performing **energy calculations** or **geometry/cell optimizations** in 2D with the CG model (see [example 2](examples/example2/example2.ipynb))
- **Redecorating** CG structure with atomistic building blocks to recreate a full atomistic description of a system (see [example 3](examples/example3/example3.ipynb))

## Description of the coarse-grained system (`cg_atoms`)
As basis, we use the atomic simulation environment (ASE). The coarse-grained system is described by the sites of the cores as pseudo-atoms in an ASE atoms-object (`cg_atoms`).
`cg_atoms` contains additionally as arrays the following information:
- `neighbor_ids`: The IDs of the neighbors of each core-site
- `neighbor_distances`:  The distances to the respective neighbors of each core-site
- `linker_sites`: Vectors which indicate position of attatched linkesr for each core-site
- `linker_neighbors`: Provides a mapping which `linker_sites` are connected to which neighbors


## References
- Toward Coarse-Grained Elasticity of Single-Layer Covalent Organic Frameworks. Alexander Croy, Antonios Raptakis, David Bodesheim, Arezoo Dianat, Gianaurelio Cuniberti, *The Journal of Physical Chemistry C*, 2022, 126, 44, 18943-18951: https://doi.org/10.1021/acs.jpcc.2c06268
- Elastic properties of defective 2D polymers from regression driven coarse-graining. David Bodesheim, Alexander Croy, Gianaurelio Cuniberti, *Journal of Chemical Theory and Computation*, 2025, 21, 21, 11210–11218: https://doi.org/10.1021/acs.jctc.5c01339
