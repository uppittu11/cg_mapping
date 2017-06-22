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



def _compute_com(traj):
    """ Compute center of mass

    Parameters
    -----------
    traj : mdtraj trajectory
        Trajectory of atoms for which center of mass will be computed
    
    Returns
    -------
    com : tuple (n_frame, 3)
        Coordinates of center of mass
    """
    numerator = np.sum(traj.xyz[:,:,:], axis=1)
    totalmass = sum(atom.element.mass for atom in traj.top.atoms)
    com = numerator/totalmass 
    return com

def _create_CG_topology(topol=None, mapping=None):
    """ Create CG topology from given topology and mapping

    Parameters
    ---------
    topol : mdtraj Topology
    mapping : dict

    Returns
    -------
    CG_topology : mdtraj Topology
    """
    CG_topology = mdtraj.Topology()
    for atom in topol.atoms:
        # Read each atom and see if it matches a mapping
        print(atom.name)



    return CG_topology



trajfile = "md_pureDSPC.xtc"
pdbfile = "md_pureDSPC.pdb"
#traj = mdtraj.load(trajfile,top=pdbfile)
traj = mdtraj.load(pdbfile)
topol = traj.topology

# Read in the mapping file
mapfile = 'mappings/DSPC.map'
mapping = _load_mapping(mapfile=mapfile)

# Go through the trajectory frame by frame

CG_topol = _create_CG_topology(topol=topol, mapping=mapping)

# Generate a CG topology from the atomistic using the mapping
# Generate a CG trajecctory from the atomistci using the CG topology

# Iterate through the atoms, 
# if a tuple matches the dictionary,
# perform the forward mapping by
# computing the center of mass 
# and adding that to an array of coordinates 
# with the respective atom type

