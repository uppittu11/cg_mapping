import pytest
from cg_mapping.mapper import Mapper
import mdtraj as md
import numpy as np
from os import path

class ConfTest:
    @pytest.fixture
    def loaded_mapper(self):
        mapper = Mapper(solvent_mapping=4, solvent_name='tip3p')
        trajfile = path.join(path.dirname(__file__), 
                             "test_files/traj.xtc")
        topfile = path.join(path.dirname(__file__), 
                            "test_files/start.gro")
        traj = md.load(trajfile, top=topfile)
        mapper.load_trajectory(traj)
        mapper.load_mapping_dir()
        return mapper

    @pytest.fixture
    def minimal_mapper(self):
        mapper = Mapper(solvent_mapping=4, solvent_name='tip3p')
        trajfile = path.join(path.dirname(__file__), 
                             "test_files/traj.xtc")
        topfile = path.join(path.dirname(__file__), 
                            "test_files/start.gro")
        traj = md.load(trajfile, top=topfile)
        residues_to_keep = np.arange(0, traj.top.n_residues, 10) + 1
        selection  = f"residue {' '.join(residues_to_keep.astype(str))}"
        traj = traj.atom_slice(traj.top.select(selection))
        mapper.load_trajectory(traj)
        mapper.load_mapping_dir()
        return mapper
