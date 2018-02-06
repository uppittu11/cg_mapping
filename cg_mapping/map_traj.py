import time
import pdb
import warnings
import itertools
from multiprocessing import Pool
from collections import OrderedDict
import os
from optparse import OptionParser
import numpy as np
import mdtraj
import mbuild as mb
import cg_mapping
from cg_mapping.CG_bead import CG_bead

PATH_TO_MAPPINGS='/raid6/homes/ahy3nz/Programs/cg_mapping/cg_mapping/mappings/'
HOOMD_FF="/raid6/homes/ahy3nz/Programs/setup/FF/CG/msibi_ff.xml"

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
    bonding_info = []
    with open(mapfile,'r') as f:
        for line in f:
            if line.rstrip():
                if 'bond' in line.split(":")[0]:
                    atom_i = line.split(":")[1].split()[0].rstrip()
                    atom_j = line.split(":")[1].split()[1].rstrip()
                    bonding_info.append((atom_i, atom_j))

                else:
                    mapping_dict.update({line.split(":")[0].rstrip():
                            [line.split(":")[1].rstrip(), line.split(":")[2].rstrip().split()]})

    return mapping_dict, bonding_info


   
def _create_CG_topology(topol=None, all_CG_mappings=None, water_bead_mapping=4,
        all_bonding_info=None):
    """ Create CG topology from given topology and mapping

    Parameters
    ---------
    topol : mdtraj Topology
    all_CG_mappings : dict
        maps residue names to respective CG 
        mapping dictionaries(CG index, [beadtype, atom indices])
    water_bead_mapping : int
        specifies how many water molecules get mapped to a water CG bead
    all_bonding_info : dict
        maps residue names to bonding info arrays 
        np.ndarray (n, 2)


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
            CG_atoms = []

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


            # Add beads to topology by adding atoms
            for index, bead in enumerate(temp_CG_beads):
                bead.beadindex = index
                CG_topology_map.append(bead)
                new_CG_atom = CG_topology.add_atom(bead.beadtype, None, temp_residue)
                CG_atoms.append(new_CG_atom)

            # Add bonds to topolgoy 
            bonding_info = all_bonding_info[residue.name]
            for (index_i, index_j) in bonding_info:
                CG_topology.add_bond(CG_atoms[int(index_i)], CG_atoms[int(index_j)])

        else:
            water_counter +=1
            if water_counter % water_bead_mapping == 0:
                temp_residue = CG_topology.add_residue("HOH", CG_topology.add_chain())
                new_bead = CG_bead(beadindex=0, beadtype="P4",
                        resname='HOH')
                CG_topology_map.append(new_bead)
                CG_topology.add_atom("P4", None, temp_residue)





    return CG_topology_map, CG_topology

def _map_waters(traj, water_start, frame_index):
    """ Worker function to parallelize mapping waters via kmeans

    Parameters
    ----------
    traj : mdtraj trajectory
        full atomstic trajectory
    frame index : int
        parallelizing calculation frame by frame
    water_start : int
        counter denoting which index in the CG coordiantes is water

    """
    from sklearn import cluster
    start = time.time()
    frame = traj[frame_index]
    # Get atom indices of all water oxygens
    waters = frame.topology.select('water and name O')

    # Get coordinates and number of all water oxygens
    n_aa_water = len(waters)
    aa_water_xyz = frame.atom_slice(waters).xyz[0,:,:]

    # Number of CG water molecules based on mapping scheme
    water_bead_mapping = 4
    n_cg_water = int(n_aa_water /  water_bead_mapping)
    # Water clusters are a list (n_cg_water) of empty lists
    water_clusters = [[] for i in range(n_cg_water)]

    # Perform the k-means clustering based on the AA water xyz
    k_means = cluster.KMeans(n_clusters=n_cg_water)
    k_means.fit(aa_water_xyz)

    # Each cluster index says which cluster an atom belongs to
    for atom_index, cluster_index in enumerate(k_means.labels_):
        # Sort each water atom into the corresponding cluster
        # The item being added should be an atom index
        water_clusters[cluster_index].append(waters[atom_index])


    single_frame_coms = []
    # For each cluster, compute enter of mass
    for cg_index, water_cluster in enumerate(water_clusters):
        com = mdtraj.compute_center_of_mass(frame.atom_slice(water_cluster))
        single_frame_coms.append((frame_index, cg_index+water_start, com))
        #CG_xyz[frame_index, cg_index + water_start,:] = com
    end = time.time()
    print("K-means for frame {}: {}".format(frame_index, end-start))

    return single_frame_coms
 
def _convert_xyz(traj=None, CG_topology_map=None, water_bead_mapping=4,parallel=True):
    """Take atomistic trajectory and convert to CG trajectory

    Parameters
    ---------
    traj : mdtraj Trajectory
        Atomistic trajectory
    CG_topology : list
        list of CGbead()
    parallel : boolean
        True if using parallelized, false if using serial

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
    print("Converting water beads via k-means")
    start = time.time()
    if len(water_indices)>0:
        water_start = min(water_indices)


        # Perform kmeans, frame-by-frame, over all water residues
        # Workers will return centers of masses of clusters, frame index, and cg index
        # Master will assign to CG_xyz
        if parallel:
            all_frame_coms = []
            with Pool() as p:
                all_frame_coms = p.starmap(_map_waters, zip(itertools.repeat(traj), 
                    itertools.repeat(water_start), range(traj.n_frames)))

            end = time.time()
            print("K-means and converting took: {}".format(end-start))

            print("Writing to CG-xyz")
            start = time.time()
            for snapshot in all_frame_coms:
                for element in snapshot:
                    CG_xyz[element[0],element[1],:] = element[2]
            end =  time.time()
            print("Writing took: {}".format(end-start))

            return CG_xyz
        
        else:
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

def compute_avg_box(traj):
    """ Compute average box lengths"""

    return np.mean(traj.unitcell_lengths, axis=0)


parser = OptionParser()
parser.add_option("-f", action="store", type="string", dest = "trajfile", default='last20.xtc')
parser.add_option("-c", action="store", type="string", dest = "topfile", default='md_pureDSPC.pdb')
parser.add_option("-o", action="store", type="string", dest = "output", default='cg-traj')
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
all_bonding_info = OrderedDict()

molecule_mapping, molecule_bonding = _load_mapping(mapfile=DSPCmapfile)
all_CG_mappings.update({'DSPC': molecule_mapping})
all_bonding_info.update({'DSPC': molecule_bonding})

molecule_mapping, molecule_bonding = _load_mapping(mapfile=watermapfile)
all_CG_mappings.update({'HOH': molecule_mapping})
all_bonding_info.update({'HOH': molecule_bonding})

molecule_mapping, molecule_bonding = _load_mapping(mapfile=alc16mapfile)
all_CG_mappings.update({'alc16': molecule_mapping})
all_bonding_info.update({'alc16': molecule_bonding})

molecule_mapping, molecule_bonding = _load_mapping(mapfile=acd16mapfile)
all_CG_mappings.update({'acd16': molecule_mapping})
all_bonding_info.update({'acd16': molecule_mapping})

CG_topology_map, CG_topology = _create_CG_topology(topol=topol, all_CG_mappings=all_CG_mappings, all_bonding_info=all_bonding_info)
CG_xyz = _convert_xyz(traj=traj, CG_topology_map=CG_topology_map)

CG_traj = mdtraj.Trajectory(CG_xyz, CG_topology, time=traj.time, 
        unitcell_lengths=traj.unitcell_lengths, unitcell_angles = traj.unitcell_angles)


avg_box_lengths = compute_avg_box(traj)
print("Avg box length: {}".format(avg_box_lengths))

CG_traj.save('{}.xtc'.format(options.output))
CG_traj[-1].save('{}.gro'.format(options.output))
CG_traj[-1].save('{}.h5'.format(options.output))
CG_traj[-1].save('{}.xyz'.format(options.output))
CG_traj[-1].save('{}.pdb'.format(options.output))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mb_compound = mb.Compound()
    mb_compound.from_trajectory(CG_traj, frame=-1, coords_only=False)
    original_box = mb.Box(lengths=[length for length in mb_compound.periodicity])

    # Because we are resizing the box based on the average box over the trajectory,
    # we need to scale the coordinates appropriately, since they are taken only
    # from the final frame.
    scaling_ratio = [new/old for old, new in zip(original_box.lengths, avg_box_lengths)]
    print("scaling_ratio: {}".format(scaling_ratio))
    for particle in mb_compound.particles():
        for i, frame in enumerate(particle.xyz):
            particle.xyz[i] = [factor*coord for factor, coord in zip(scaling_ratio, particle.xyz[i])]

    # Tile this using grid 3d pattern
    cube = mb.Grid3DPattern(2,2,2)
    # Define a new box based on average box lengths from AA traj, scaled by 8
    new_box = mb.Box(lengths=[2*length for length in avg_box_lengths])
    # Scale pattern lengths based on the new box lengths
    cube.scale([length for length in new_box.lengths])
    replicated = cube.apply(mb_compound)
    mirrored_image = mb.Compound()
    for item in replicated:
        mirrored_image.add(item)

    # Particle renaming due to mbuild coarsegrained format
    for particle in mb_compound.particles():
        particle.name = "_"+ particle.name.strip()
    for particle in mirrored_image.particles():
        particle.name = "_"+ particle.name.strip()

    mb_compound.save('{}.hoomdxml'.format(options.output), ref_energy = 0.239, ref_distance = 10, forcefield_files=HOOMD_FF, overwrite=True, box=original_box)
    mirrored_image.save('{}_2x2x2.hoomdxml'.format(options.output), ref_energy = 0.239, ref_distance = 10, forcefield_files=HOOMD_FF, overwrite=True, box=new_box)

    
end=time.time()
print(end-start)


