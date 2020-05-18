import pytest
import numpy as np
from conf_test import ConfTest

class TestTraj(ConfTest):
    def test_construct_traj(self, minimal_mapper):
        minimal_mapper._aa_traj = minimal_mapper._aa_traj[:2]
        minimal_mapper._map_topology()
        minimal_mapper._convert_xyz()
        minimal_mapper._construct_traj()
        assert minimal_mapper._cg_traj.n_atoms == 460
        assert np.allclose(minimal_mapper._cg_traj.xyz[0,0],
                           [1.0085664, 1.4182341, 3.6780086])
        assert np.allclose(minimal_mapper._cg_traj.unitcell_lengths[0],
                           [6.3977127, 6.3977127, 8.594183 ])
        