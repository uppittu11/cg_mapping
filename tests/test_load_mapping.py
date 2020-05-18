import pytest
from cg_mapping.mapping import ResMapping
from cg_mapping.mapper import Mapper
from cg_mapping.io import default_mapping_dir
from os import path
from conf_test import ConfTest


class TestLoadMapping(ConfTest):
    def test_single_mapfile(self):
        filename = path.join(default_mapping_dir(), "chol.map")
        mapping = ResMapping.load("chol", filename)
        assert mapping.name == "chol"
        assert len(mapping.beads) == 9
        assert len(mapping.bonds) == 11

    def test_load_dir(self):
        mapper = Mapper()
        mapper.load_mapping_dir()
        assert len(mapper._mappings) > 0

