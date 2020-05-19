from os import path, system
import glob
from multiprocessing import Pool, cpu_count

import numpy as np
from mdtraj import Trajectory, Topology

from cg_mapping.mapping import ResMapping, BeadMapping
from cg_mapping.cg_bead import CGBead
from cg_mapping.io import default_mapping_dir
from cg_mapping.exceptions import OutOfOrderError
from cg_mapping.workers import _map_solvent


class Mapper:
    """
    An object to convert an atomistic system to a CG system

    Attributes
    ----------
    mappings : dict
        A dictionary containing the {name : mapping} for each residue

    solvent_mapping : int, default=4
        Number of solvent molecules to map to a single bead via k-means
        clustering

    solvent_name : string, default='tip3p'
        Name of solvent residue in atomistic system

    aa_traj : mdtraj.Trajectory
        The atomistic trajectory to convert to CG

    aa_top: mdtraj.Topology
        The atomistic topology to convert to CG

    cg_traj : mdtraj.Trajectory
        The CG trajectory to converted from atomistic

    cg_top: mdtraj.Topology
        The CG topology to convert from atomistic

    """
    def __init__(self, solvent_name='tip3p', solvent_mapping=4):
        self._mappings = dict()
        self._solvent_mapping = solvent_mapping
        self._solvent_name = solvent_name
        self._cg_traj = None
        self._cg_top  = None


    def load_trajectory(self, trajectory):
        self._aa_traj = trajectory
        self._aa_top = trajectory.top


    def load_mapping_dir(self, mapping_dir=None):
        """
        Load all mapping files from a directory.

        Arguments:
        ----------
        mapping_dir : string, default=None
            Path to the directory containing mapping files. Loads from
            the internal `mappings` directory by default

        """
        if mapping_dir is None:
            mapping_dir = default_mapping_dir()

        assert path.exists(mapping_dir)

        for filename in glob.glob("{}/*map".format(mapping_dir)):
            self.load_mapping(filename)


    def load_mapping(self, filename):
        """
        Load a single mapping file from disk.

        Arguments:
        ----------
        filename : string
            Path to the mapping file

        """

        assert path.exists(filename)

        name = path.basename(filename).split(".")[0]
        mapping = ResMapping.load(name, filename)
        self._mappings.update({name : mapping})


    def cg_map(self):
        """
        Execute full CG mapping pipeline and return the CG trajectory

        """

        if self._cg_traj is None:
            self._map_topology()
            self._convert_xyz()
            self._construct_traj()

        return self._cg_traj


    def _map_topology(self):
        """
        Create CG topology from given topology and mapping

        """

        # Ensure that a trajectory has been loaded
        if self._aa_traj is None:
            raise OutOfOrderError("An atomistic trajectory has not "
                                  "been loaded into this Mapper yet.")


        self._atom_bead_mapping = dict()
        self._cg_top = Topology()
        self._solvent_counter = 0

        # Loop over all residues
        for residue in self._aa_top.residues:
            if residue.name == self._solvent_name:
                self._map_solvent_top(residue)
            else:
                self._map_nonsolvent_top(residue)

    def _map_solvent_top(self, residue):
        """
        Create CG solvent residue from given residue and add it to the
        CG topology.

        Arguments:
        ----------
        residue: mdtraj.topology.Residue
            The atomistic residue to be mapped to CG

        """
        self._solvent_counter += 1
        if self._solvent_counter % self._solvent_mapping == 0:
            cg_residue = self._cg_top.add_residue(self._solvent_name,
                                                  self._cg_top.add_chain())
            cg_bead = CGBead(bead_type=self._solvent_name)
            mdtraj_bead = self._cg_top.add_atom(self._solvent_name, None,
                                                cg_residue)
            self._atom_bead_mapping[mdtraj_bead] = cg_bead
            return cg_residue

    def _map_nonsolvent_top(self, residue):
        """
        Create CG non-solvent residue from given residue and add it to
        the CG topology.

        Arguments:
        ----------
        residue: mdtraj.topology.Residue
            The atomistic residue to be mapped to CG

        """

        # Obtain the correct molecule mapping based on the residue
        res_mapping = self._mappings[residue.name]

        # Add an empty residue to the CG topology
        cg_residue = self._cg_top.add_residue(
                            residue.name,
                            self._cg_top.add_chain())

        # Make a list of atoms in the residue
        atoms = np.array([atom.index for atom in residue.atoms])

        # Make an empty list to store beads
        cg_beads = []

        # Create CG beads for each bead in the mapping
        for bead in res_mapping.beads:
            bead_atoms = atoms.take(bead.mapping_indices)
            cg_bead = CGBead(bead_type=bead.name, atom_indices=bead_atoms)
            mdtraj_bead = self._cg_top.add_atom(cg_bead.bead_type, None,
                                                cg_residue)
            cg_beads.append(mdtraj_bead)
            self._atom_bead_mapping[mdtraj_bead] = cg_bead

        # Add bonds to topology
        for index_i, index_j in res_mapping.bonds:
            self._cg_top.add_bond(cg_beads[int(index_i)],
                                    cg_beads[int(index_j)])

        return cg_residue


    def _convert_xyz(self):
        """
        Take atomistic trajectory and convert to CG trajectory

        """

        cg_xyz = []
        for bead in self._cg_top.atoms:
            if bead.name == self._solvent_name:
                bead_xyz = np.zeros((self._aa_traj.n_frames,3))
            else:
                atom_indices = self._atom_bead_mapping[bead].atom_indices
                masses = np.array([self._aa_top.atom(i).element.mass
                                   for i in atom_indices])
                bead_xyz = (np.sum((self._aa_traj.xyz[:,atom_indices,:]
                                    * masses[None,:,None]), axis=1) /
                            np.sum(masses))

            cg_xyz.append(bead_xyz)

        cg_xyz = np.array(cg_xyz)
        cg_xyz = np.swapaxes(cg_xyz, 0, 1)

        # Figure out at which coarse grain index the waters start
        # Perform kmeans, frame-by-frame, over all water residues
        # Workers will return centers of masses of clusters, frame index, and cg index
        # Master will assign to CG_xyz
        if self._solvent_counter > 0:
            with Pool(cpu_count()) as pool:
                chunksize = int(self._aa_traj.n_frames / cpu_count()) + 1
                args = list(zip(self._aa_traj,
                                [self._solvent_mapping]*self._aa_traj.n_frames,
                                [self._solvent_name]*self._aa_traj.n_frames))
                coms = pool.starmap(_map_solvent, args, chunksize)

            pool.join()

            coms = np.squeeze(np.array(coms))
            cg_xyz[:,self._cg_top.select(f"name {self._solvent_name}"),:] = coms

        self._cg_xyz = cg_xyz


    def _construct_traj(self):
        """
        Create an mdtraj.Trajectory from the CG topology and xyz.

        """

        cg_traj = Trajectory(self._cg_xyz,
                             self._cg_top,
                             time=self._aa_traj.time,
                             unitcell_lengths=self._aa_traj.unitcell_lengths,
                             unitcell_angles=self._aa_traj.unitcell_angles)

        self._cg_traj = cg_traj
