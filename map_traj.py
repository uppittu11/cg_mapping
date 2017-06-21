import os
import numpy as np
import mdtraj
from collections import OrderedDict

#trajfile = "md_pureDSPC.xtc"
trajfile = "md_pureDSPC.pdb"
traj = mdtraj.load(trajfile)

# Read in the mapping file

# Go through the trajectory frame by frame

# Iterate through the atoms, 
# if a tuple matches the dictionary,
# perform the forward mapping by
# computing the center of mass 
# and adding that to an array of coordinates 
# with the respective atom type

