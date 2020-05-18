import pytest
from cg_mapping.mapping import ResMapping
from cg_mapping.mapper import Mapper
from conf_test import ConfTest

class TestLoadMapping(ConfTest):
    def test_single_mapfile(self):
        filename = "/Users/parashara/devel/git/cg_mapping/cg_mapping/mappings/chol.map"
        mapping = ResMapping.load("chol", filename)
        assert mapping.name == "chol"
        assert len(mapping.beads) == 9
        assert len(mapping.bonds) == 11

    def test_load_dir(self):
        mapper = Mapper()
        mapper.load_mapping_dir()
        assert len(mapper._mappings) > 0

