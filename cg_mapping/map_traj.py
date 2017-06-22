import numpy as np
import mdtraj
from collections import OrderedDict

def _load_mapping(mapfile=None,reverse=False):
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
        #if reverse:
        #    # Reverse: keys are CG beads, values are indices
        #    mapping_dict = {line.split(":")[0].rstrip(): line.split(":")[1].rstrip().split()
        #                for line in f if line.rstrip()}
        #else:
        #    # Forward: keys are indices, values are CG beads
        #    mapping_dict = { line.split(":")[1].rstrip().split() : line.split(":")[0].rstrip()
        #                for line in f if line.rstrip()}

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

def _create_CG_topology(topol=None, all_CG_mappings=None):
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
    for residue in topol.residues:
        # Iterate through each AA residue while starting index at 0
        temp_CG_bead = []
        molecule_mapping = all_CG_mappings[residue.name]
        for index, atom in enumerate(residue.atoms):
            temp_CG_bead.append(index)
            for key in molecule_mapping.keys():
                if molecule_mapping[key] is temp_CG_bead:
                    print("hit")



    return CG_topology



trajfile = "md_pureDSPC.xtc"
pdbfile = "md_pureDSPC.pdb"
#traj = mdtraj.load(trajfile,top=pdbfile)
traj = mdtraj.load(pdbfile)
topol = traj.topology

# Read in the mapping file
mapfile = 'mappings/DSPC.map'
# Huge dictionary of dictionaries, keys are molecule names
# Values are the molecule's mapping dictionary
all_CG_mappings = OrderedDict()
molecule_mapping = _load_mapping(mapfile=mapfile)
all_CG_mappings.update({''.join(molecule_mapping['Name']): molecule_mapping})

# Go through the trajectory frame by frame

CG_topol = _create_CG_topology(topol=topol, all_CG_mappings=all_CG_mappings)

# Generate a CG topology from the atomistic using the mapping
# Generate a CG trajecctory from the atomistci using the CG topology

# Iterate through the atoms, 
# if a tuple matches the dictionary,
# perform the forward mapping by
# computing the center of mass 
# and adding that to an array of coordinates 
# with the respective atom type

