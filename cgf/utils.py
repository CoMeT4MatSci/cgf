from ase import Atoms

def remove_hatoms(s):
    del s[[atom.symbol == 'H' for atom in s]]
    return s
