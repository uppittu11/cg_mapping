import time
from collections import OrderedDict
import os
from optparse import OptionParser
import numpy as np
import mdtraj
import mbuild as mb
import cg_mapping
from cg_mapping.CG_bead import CG_bead

PATH_TO_MAPPINGS='/raid6/homes/ahy3nz/Programs/cg_mapping/cg_mapping/mappings/'
HOOMD_FF="/raid6/homes/ahy3nz/Programs/setup/FF/CG/myforcefield.xml"

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

    return mapping_dict

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
            # Obtain the correct molecule mapping based on the residue
        molecule_mapping = all_CG_mappings[residue.name]
        temp_residue = CG_topology.add_residue(residue.name, CG_topology.add_chain())
        temp_CG_indices = []
        temp_CG_atoms = []
        temp_CG_beads = [None]*len(molecule_mapping.keys())

        for index, atom in enumerate(residue.atoms):
            temp_CG_indices.append(str(index))
            temp_CG_atoms.append(atom)
            for key in molecule_mapping.keys():
                if set(molecule_mapping[key][1]) == set(temp_CG_indices):
                    new_bead = CG_bead(beadindex=0, 
                                       beadtype=molecule_mapping[key][0],
                                       resname=residue.name,
                                       atom_indices=[atom.index for atom in temp_CG_atoms])
                    CG_beadindex +=1 
                    temp_CG_indices = []
                    temp_CG_atoms = []
                    temp_CG_beads[int(key)] = new_bead

                else:
                    pass

        for index, bead in enumerate(temp_CG_beads):
            bead.beadindex = index
            CG_topology_map.append(bead)
            CG_topology.add_atom(bead.beadtype, bead.beadtype, temp_residue)

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
    for index, bead in enumerate(CG_topology_map):
        atom_indices = bead.atom_indices 
        # Two ways to compute center of mass, both are pretty fast
        #bead_coordinates = _compute_com(traj.atom_slice(atom_indices))
        bead_coordinates = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices))
        #CG_xyz[:, bead.beadindex, :] = bead_coordinates
        CG_xyz[:, index, :] = bead_coordinates


    return CG_xyz


parser = OptionParser()
parser.add_option("-f", action="store", type="string", dest = "trajfile")
parser.add_option("-c", action="store", type="string", dest = "topfile")
parser.add_option("-o", action="store", type="string", dest = "output", default='traj')
(options, args) = parser.parse_args()


#trajfile = "last20.xtc"

#pdbfile = "md_DSPC-34_alc16-33_acd16-33_1-27b.gro"
traj = mdtraj.load(options.trajfile, top=options.topfile)
topol = traj.topology
start=time.time()
# Read in the mapping files, could be made more pythonic
DSPCmapfile = os.path.join(PATH_TO_MAPPINGS,'DSPC.map')#'mappings/DSPC.map'
watermapfile = os.path.join(PATH_TO_MAPPINGS,'water.map')
alc16mapfile = os.path.join(PATH_TO_MAPPINGS,'C16OH.map')
acd16mapfile = os.path.join(PATH_TO_MAPPINGS,'C16FFA.map')
# Huge dictionary of dictionaries, keys are molecule names
# Values are the molecule's mapping dictionary
# could be made more pythonic
all_CG_mappings = OrderedDict()

molecule_mapping = _load_mapping(mapfile=DSPCmapfile)
all_CG_mappings.update({'DSPC': molecule_mapping})

molecule_mapping = _load_mapping(mapfile=watermapfile)
all_CG_mappings.update({'HOH': molecule_mapping})

molecule_mapping = _load_mapping(mapfile=alc16mapfile)
all_CG_mappings.update({'alc16': molecule_mapping})

molecule_mapping = _load_mapping(mapfile=acd16mapfile)
all_CG_mappings.update({'acd16': molecule_mapping})

CG_topology_map, CG_topology = _create_CG_topology(topol=topol, all_CG_mappings=all_CG_mappings)
CG_xyz = _convert_xyz(traj=traj, CG_topology_map=CG_topology_map)

CG_traj = mdtraj.Trajectory(CG_xyz, CG_topology, time=traj.time, 
        unitcell_lengths=traj.unitcell_lengths, unitcell_angles = traj.unitcell_angles)
CG_traj.save('cg-{}.xtc'.format(options.output))
CG_traj[0].save('cg-{}.gro'.format(options.output))
CG_traj[0].save('cg-{}.h5'.format(options.output))
CG_traj[0].save('cg-{}.xyz'.format(options.output))


mb_compound = mb.Compound()
mb_compound.from_trajectory(CG_traj, frame=-1, coords_only=False)
for particle in mb_compound.particles():
    particle.name = "_"+ particle.name.strip()
mb_compound.save('cg-{}.hoomdxml'.format(options.output), ref_energy = 0.239, ref_distance = 10, forcefield_files=HOOMD_FF, overwrite=True)


end=time.time()
print(end-start)

# Iterate through the atoms, 
# if a tuple matches the dictionary,
# perform the forward mapping by
# computing the center of mass 
# and adding that to an array of coordinates 
# with the respective atom type

