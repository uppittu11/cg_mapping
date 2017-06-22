import numpy as np
import mdtraj
import time
from CG_bead import CG_bead
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
        OrderedDict (CG bead index : [beadtype, list of atom indices])
    
    Notes
    -----
    mapping files are ":" delimited
    col0: bead index
    col1: bead type
    col2: atom indices

        """

    mapping_dict = OrderedDict()
    with open(mapfile,'r') as f:
        mapping_dict = {line.split(":")[0].rstrip(): 
                [line.split(":")[1].rstrip(),line.split(":")[2].rstrip().split()] 
                for line in f if line.rstrip()}

    #return mapping_dict, CG_index_to_type_dict
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
    all_CG_mappings : dict
        maps residue names to respective CG 
        mapping dictionaries(CG index, [beadtype, atom indices])

    Returns
    -------
    CG_topology_map : tuple of CG_bead()
    CG_topology : mdtraj topology
        Need to fill in more details for topology creation

    """
    CG_topology_map = []
    CG_topology = mdtraj.Topology()
    CG_beadindex = 0
    for residue in topol.residues:
        # Create a temporary coarse grained bead to add atom indices
        temp_CG_indices = []
        temp_CG_atoms = []
        # Obtain the correct molecule mapping based on the residue
        molecule_mapping = all_CG_mappings[residue.name]
        # iterate through the residue's atoms
        for index, atom in enumerate(residue.atoms):
            # Add the index to the temporary CG bead
            temp_CG_indices.append(str(index))
            temp_CG_atoms.append(atom)
            # Check if the temp CG bead exists in the molecule mapping
            for key in molecule_mapping.keys():
                # If the molecule mapping has this sequence, 
                # And reset the temporary CG bead
                if set(molecule_mapping[key][1]) == set(temp_CG_indices):
                    new_bead = CG_bead(beadindex=CG_beadindex, 
                                       beadtype=molecule_mapping[key][0],
                                       resname=residue.name,
                                       atom_indices=[atom.index for atom in temp_CG_atoms])
                    CG_beadindex +=1 
                    temp_CG_indices = []
                    temp_CG_atoms = []
                    CG_topology_map.append(new_bead)
                    CG_topology.add_atom(new_bead.beadtype,
                            new_bead.beadtype, CG_topology.add_residue(
                                new_bead.resname, CG_topology.add_chain()))
                else:
                    pass

    return CG_topology_map, CG_topology

def _convert_xyz(traj=None, CG_topology_map=None):
    """Take atomistic trajectory and convert to CG trajectory

    Parameters
    ---------
    traj : mdtraj Trajectory
        Atomistic trajectory
    CG_topology : list
        list of CGbead()

    Returns
    ------
    CG_xyz : np.ndarray(n_frame, n_CG_beads, 3)

    
    """
    # Iterate through the CG_topology
    # For each bead, get the atom indices
    # Then slice the trajectory and compute hte center of mass for that particular bead
    CG_xyz = np.ndarray(shape=(traj.n_frames, len(CG_topology_map),3))
    for bead in CG_topology_map:
        atom_indices = bead.atom_indices 
        # Two ways to compute center of mass, both are pretty fast
        bead_coordinates = _compute_com(traj.atom_slice(atom_indices))
        #bead_coordinates = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices))
        CG_xyz[:, bead.beadindex, :] = bead_coordinates


    return CG_xyz




trajfile = "md_pureDSPC.xtc"
pdbfile = "md_pureDSPC.pdb"
#traj = mdtraj.load(trajfile,top=pdbfile)
traj = mdtraj.load(pdbfile)
topol = traj.topology
start=time.time()
# Read in the mapping files, could be made more pythonic
DSPCmapfile = 'mappings/DSPC_index.map'
watermapfile = 'mappings/water_index.map'

# Huge dictionary of dictionaries, keys are molecule names
# Values are the molecule's mapping dictionary
# could be made more pythonic
all_CG_mappings = OrderedDict()

molecule_mapping = _load_mapping(mapfile=DSPCmapfile)
all_CG_mappings.update({'DSPC': molecule_mapping})

molecule_mapping = _load_mapping(mapfile=watermapfile)
all_CG_mappings.update({'HOH': molecule_mapping})

CG_topology_map, CG_topology = _create_CG_topology(topol=topol, all_CG_mappings=all_CG_mappings)
CG_xyz = _convert_xyz(traj=traj, CG_topology_map=CG_topology_map)

CG_traj = mdtraj.Trajectory(CG_xyz, CG_topology, time=traj.time, 
        unitcell_lengths=traj.unitcell_lengths, unitcell_angles = traj.unitcell_angles)
CG_traj.save('cg-traj.xtc')
CG_traj[0].save('frame0.gro')
end=time.time()
print(end-start)

# Iterate through the atoms, 
# if a tuple matches the dictionary,
# perform the forward mapping by
# computing the center of mass 
# and adding that to an array of coordinates 
# with the respective atom type

