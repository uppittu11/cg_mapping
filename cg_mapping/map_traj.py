import os
import numpy as np
import mdtraj
from collections import OrderedDict

def _load_mapping(mapfile=None):
    """ Load a forward mapping
    
    Parameters
    ----------
    mapfile : str
        Path to mapping file

    Returns
    -------
    mapping_dict : OrderedDict()
        OrderedDict (CG bead: list of atom indices)

        """

    mapping_dict = OrderedDict()
    with open(mapfile,'r') as f:
        mapping_dict = {line.split(":")[0].rstrip(): line.split(":")[1].rstrip().split()
                        for line in f if line.rstrip()}
    return mapping_dict




#trajfile = "md_pureDSPC.xtc"
trajfile = "md_pureDSPC.pdb"
traj = mdtraj.load(trajfile)

# Read in the mapping file
mapfile = 'mappings/DSPC.map'
mapping_dict = _load_mapping(mapfile=mapfile)

# Go through the trajectory frame by frame

# Iterate through the atoms, 
# if a tuple matches the dictionary,
# perform the forward mapping by
# computing the center of mass 
# and adding that to an array of coordinates 
# with the respective atom type

