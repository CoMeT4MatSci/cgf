import numpy as np
from grandiso import find_motifs

def find_unique_motifs(motif, G):
    # find all motifs in graph
    mfs = find_motifs(motif, G, isomorphisms_only=True)
    print(len(mfs))

    # eliminate duplicates (same nodes, different order)
    n = 0
    while n < len(mfs):
        unique_mfs = mfs[0:n+1]
        m0 = set(mfs[n].values())
        for i in np.arange(n+1, len(mfs)):
            if set(mfs[i].values()) != m0:
                unique_mfs.append(mfs[i])

        mfs = unique_mfs.copy()
        n = n + 1

    return mfs
