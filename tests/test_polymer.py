from os import path

import pytest
import numpy as np
from conf_test import ConfTest

import mdtraj as md
from cg_mapping.mapper import Mapper
from cg_mapping.mapping import PolymerMapping

class TestTraj(ConfTest):
    def test_construct_traj(self, minimal_mapper):
        my_mapper = Mapper()
        my_mapper.load_mapping_dir(ff="martini")

        monomers = [my_mapper.mappings[name] for name in ["upcap", "downcap", "deaema"]]
        sequence = ["upcap"] + ["deaema"]*6 + ["downcap"]
        eb_mapping = PolymerMapping(name="EB", monomers=monomers, sequence=sequence)
        my_mapper.load_mapping(eb_mapping)

        atomistic_file = path.join(path.dirname(__file__),
                "test_files/atomistic_poly.gro")
        atomistic_trajectory = md.load(atomistic_file)
        my_mapper.load_trajectory(atomistic_trajectory)
        cg_trajectory = my_mapper.cg_map()
        cg_trajectory.xyz = cg_trajectory.xyz.round(decimals=3)

        cg_file = path.join(path.dirname(__file__),
                "test_files/cg_poly.gro")
        correct_cg_trajectory = md.load(cg_file)

        assert cg_trajectory.xyz.shape == correct_cg_trajectory.xyz.shape
        assert np.allclose(np.array(cg_trajectory.xyz), np.array(correct_cg_trajectory.xyz), rtol=1e-02)
