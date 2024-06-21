All training structures were performed with the following settings (unless stated differently in the respective folder:)

Contains stressed and relaxed structures of a unit cell of a COF unit cell in 'traj_training.traj'.

The first structure is the cell-relaxed structure at 0 strain.
The next 20 structures are isotropic strains from -1% to 1% (excluding 0%).
The next 20 structures are shear strains (meaning positive(negative) strain along x and negative(positive) strain along y). From -1% to 1% (excluding 0%).

In case DFTB calculations were performed, the following settings were used:
With the following dftb settings with the matsci slater-koster files:
dftb_SinglePoint = {
            'Hamiltonian_MaxSCCIterations': 500,
            'Hamiltonian_SCC': 'Yes',
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
            'Hamiltonian_Filling_': 'Fermi',
            'Hamiltonian_Filling_Temperature [K]': '100',
            }
In case of dftb_noSCC the SCC calculations were turned off.

All relaxation was performed with the SciPyFminBFGS algorithm with fmax=0.01 (see ASE documentation for details).

The resulting fitted parameters are saved in 'training_model_MikadoRR.json' and 'training_model_BAFF.json'.