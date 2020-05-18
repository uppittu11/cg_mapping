from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def _map_solvent(frame, mapping, solvent_name):
    """ 
    Worker function to parallelize mapping waters via kmeans

    Parameters
    ----------
    traj : mdtraj.Trajectory
        A single frame of the atomistic trajectory
    frame_index : int
        The index of the frame.

    """

    # Get center of each solvent
    solvent_centers = []
    residues = [residue for residue in frame.top.residues 
                if residue.name == solvent_name]
    for residue in residues:
        atom_idxs = frame.top.select(f"resid {residue.index}")
        cog = np.mean(frame.xyz[-1, atom_idxs, :], axis=1).reshape(-1)
        solvent_centers.append(cog)

    # Number of CG water molecules based on mapping scheme
    n_cg_water = int(len(solvent_centers) / mapping)

    # Water clusters are a list (n_cg_water) of empty lists
    water_clusters = defaultdict(list)

    # Perform the k-means clustering based on the AA water xyz
    solvent_centers = np.vstack(solvent_centers)
    k_means = KMeans(n_clusters=n_cg_water)
    k_means.fit(solvent_centers)

    # Each cluster index says which cluster an atom belongs to
    for residue_index, cluster_index in enumerate(k_means.labels_):
        # Sort each water atom into the corresponding cluster
        # The item being added should be an atom index
        residue = residues[residue_index].index
        water_clusters[cluster_index] = frame.top.select(f"resid {residue}")

    coms = []
    # For each cluster, compute enter of mass
    masses = np.array([atom.element.mass for atom in frame.top.atoms])
    for atom_indices in water_clusters.values():
        atom_indices = np.hstack(atom_indices)
        res_masses = masses.take(atom_indices)
        com = np.mean((res_masses[None, :, None] * 
                       frame.xyz[-1,atom_indices,:]), axis=1)
        coms.append(com)

    return coms