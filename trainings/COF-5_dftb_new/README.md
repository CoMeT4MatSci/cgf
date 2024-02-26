Contains stressed and relaxed structures of a unit cell of a COF-5 unit cell in 'traj_strain.traj'.

The first structure is the cell-relaxed structure at 0 strain.
The next 10 structures are isotropic strains from -1% to 1% (excluding 0%).
The next 10 structures are shear strains (meaning positive(negative) strain along x and negative(positive) strain along y). From -1% to 1% (excluding 0%).
With the following dftb settings with the matsci slater-koster files:
dftb_SinglePoint = {
            'Hamiltonian_Dispersion_': 'DftD4',
            'Hamiltonian_Dispersion_s6': 1.0,
            'Hamiltonian_Dispersion_s8': 3.3157614,
            'Hamiltonian_Dispersion_s9': 1.0,
            'Hamiltonian_Dispersion_a1': 0.4826330,
            'Hamiltonian_Dispersion_a2': 5.3811976,
            'Hamiltonian_MaxAngularMomentum_O': 'p',
            'Hamiltonian_MaxAngularMomentum_B': 'p',
            'Hamiltonian_MaxAngularMomentum_C': 'p',
            'Hamiltonian_MaxAngularMomentum_N': 'p',
            'Hamiltonian_MaxAngularMomentum_H': 's',
            }

Relaxation was performed with the SciPyFminBFGS algorithm with fmax=0.01 (see ASE documentation for details).

Additionally, a Stone-Wales defect of COF-5 was relaxed (with the same settings) based on a 3x3 system of the relaxed unit-cell of COF-5 from the previous calc. This is used to compare the performance of the trained model.