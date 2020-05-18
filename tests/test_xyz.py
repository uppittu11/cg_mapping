import pytest
import numpy as np
from conf_test import ConfTest

class TestXYZ(ConfTest):
    def test_map_xyz(self, minimal_mapper):
        minimal_mapper._aa_traj = minimal_mapper._aa_traj[:2]
        minimal_mapper._map_topology()
        minimal_mapper._convert_xyz()
        