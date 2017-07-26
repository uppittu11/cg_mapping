import mdtraj
import itertools
import pdb
from optparse import OptionParser
import numpy as np
import time

parser = OptionParser()
parser.add_option("-f", action="store", type="string", dest = "trajfile")
parser.add_option("-c", action="store", type="string", dest = "topfile")
parser.add_option("-o", action="store", type="string", dest = "output", default='state_A')
(options, args) = parser.parse_args()


#trajfile = 'cg-111_DSPC_C16OH_C16FFA_bulkfluid_7-25.xtc'
#topfile = 'cg-111_DSPC_C16OH_C16FFA_bulkfluid_7-25.gro'

traj = mdtraj.load(options.trajfile, top=options.topfile)
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
    np.savetxt('{}-{}-{}.txt'.format(i, j, options.output), np.column_stack([first,second]))
    end = time.time()
    print(end-start)
