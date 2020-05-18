import pytest
import numpy as np
from conf_test import ConfTest

class TestTopology(ConfTest):
    def test_map_top(self, loaded_mapper):
        loaded_mapper._map_topology()
        correct_atom_names = (["ter2"] +
                              7*["tail"] + 
                              ["amide", "mhead2"] +
                              5*["tail"] + 
                              ["oh1", "oh2", "oh3"])
        correct_atom_names = np.array(correct_atom_names, dtype=str)
        names_to_test = np.array([atom.name for atom in 
                                  loaded_mapper._cg_top.atoms], dtype=str)
        assert loaded_mapper._cg_top.n_residues == 1200
        assert loaded_mapper._cg_top.n_atoms == 4600
        assert len(loaded_mapper._cg_top.select("name 'tip3p'")) == 1000
        assert np.all(correct_atom_names == names_to_test[:18])
        