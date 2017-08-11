import time
import warnings
from collections import OrderedDict
import os
from optparse import OptionParser
import numpy as np
from sklearn import cluster
import mdtraj
import mbuild as mb
import cg_mapping
from cg_mapping.CG_bead import CG_bead
#import CG_bead

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

def _create_CG_topology(topol=None, all_CG_mappings=None, water_bead_mapping=4):
    """ Create CG topology from given topology and mapping

    Parameters
    ---------
    topol : mdtraj Topology
    all_CG_mappings : dict
        maps residue names to respective CG 
        mapping dictionaries(CG index, [beadtype, atom indices])
    water_bead_mapping : int
        specifies how many water molecules get mapped to a water CG bead

    Returns
    -------
    CG_topology_map : tuple of CG_bead()
    CG_topology : mdtraj topology
        Need to fill in more details for topology creation

    """
    CG_topology_map = []
    CG_topology = mdtraj.Topology()
    CG_beadindex = 0
    water_counter = 0
    # Loop over all residues
    for residue in topol.residues:
        if not residue.is_water:
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
        else:
            water_counter +=1
            if water_counter % water_bead_mapping == 0:
                temp_residue = CG_topology.add_residue("HOH", CG_topology.add_chain())
                new_bead = CG_bead(beadindex=0, beadtype="P4",
                        resname='HOH')
                CG_topology_map.append(new_bead)
                CG_topology.add_atom("P4", "P4", temp_residue)





    return CG_topology_map, CG_topology

def _convert_xyz(traj=None, CG_topology_map=None, water_bead_mapping=4):
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
    entire_start = time.time()
    CG_xyz = np.ndarray(shape=(traj.n_frames, len(CG_topology_map),3))
    water_indices = []
    print("Converting non-water atoms into beads over all frames")
    start = time.time()
    for index, bead in enumerate(CG_topology_map):
        if 'HOH' not in bead.resname:
            # Handle non-water residuse with center of mass calculation over all frames
            atom_indices = bead.atom_indices 
            # Two ways to compute center of mass, both are pretty fast
            bead_coordinates = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices))
            CG_xyz[:, index, :] = bead_coordinates
        else:
            # Handle waters by initially setting the bead coordinates to zero
            # Remember which coarse grain indices correspond to water
            water_indices.append(index)
            CG_xyz[:,index,:] = np.zeros((traj.n_frames,3))

    end = time.time()
    print("Converting took: {}".format(end-start))

    # Figure out at which coarse grain index the waters start
    water_start = min(water_indices)

    # Perform kmeans, frame-by-frame, over all water residues
    from sklearn import cluster
    for frame_index, frame in enumerate(traj):
        # Get atom indices of all water oxygens
        waters = traj.topology.select('water and name O')

        # Get coordinates and number of all water oxygens
        n_aa_water = len(waters)
        aa_water_xyz = traj.atom_slice(waters).xyz[frame_index,:,:]

        # Number of CG water molecules based on mapping scheme
        n_cg_water = int(n_aa_water /  water_bead_mapping)
        # Water clusters are a list (n_cg_water) of empty lists
        water_clusters = [[] for i in range(n_cg_water)]

        # Perform the k-means clustering based on the AA water xyz
        start = time.time()
        print("Clustering for frame {}".format(frame_index))
        k_means = cluster.KMeans(n_clusters=n_cg_water)
        k_means.fit(aa_water_xyz)
        end = time.time()
        print("Clustering one frame took: {}".format(end-start))

        # Each cluster index says which cluster an atom belongs to
        print("Assigning water atoms to water clusters for frame {}".format(frame_index))
        start = time.time()
        for atom_index, cluster_index in enumerate(k_means.labels_):
            # Sort each water atom into the corresponding cluster
            # The item being added should be an atom index
            water_clusters[cluster_index].append(waters[atom_index])
        end = time.time()
        print("Assigning took: {}".format(end-start))


        # For each cluster, compute enter of mass
        print("Computing cluster centers for frame {}".format(frame_index))
        start = time.time()
        for cg_index, water_cluster in enumerate(water_clusters):
            com = mdtraj.compute_center_of_mass(traj.atom_slice(water_cluster)[frame_index])
            CG_xyz[frame_index, cg_index + water_start,:] = com
        end = time.time()
        print("Computing took: {}".format(end-start))

    entire_end = time.time()
    print("XYZ conversion took: {}".format(entire_end - entire_start))


    return CG_xyz


parser = OptionParser()
parser.add_option("-f", action="store", type="string", dest = "trajfile", default='last20.xtc')
parser.add_option("-c", action="store", type="string", dest = "topfile", default='md_pureDSPC.pdb')
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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

