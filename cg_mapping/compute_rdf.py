import mdtraj
import itertools
import pdb
import numpy as np
import time
trajfile = 'cg-traj.xtc'
topfile = 'frame0.gro'

traj = mdtraj.load(trajfile, top=topfile)
beadtypes=['P3', 'Nda', 'Na', 'C1', 'Q0', 'Qa']
# Get a combination of bead types
for i,j in itertools.combinations_with_replacement(beadtypes, 2):
    # Find the corresponding indices
    i_indices = traj.topology.select('name {}'.format(i))
    j_indices = traj.topology.select('name {}'.format(j))
    # Create a matrix so each row has two elements
    # Each element being an atom index
    start = time.time()
    pairs = [(i,j) for i,j in itertools.product(i_indices, j_indices)]
    (first, second) = mdtraj.compute_rdf(traj, pairs, [0.1, 2])
    np.savetxt('{}-{}-state_A.txt'.format(i,j), np.column_stack([first,second]))
    end = time.time()
    print(end-start)
