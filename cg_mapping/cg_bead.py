class CGBead:
    def __init__(self,
            bead_index=0,
            bead_type=None,
            res_name=None,
            atom_indices=None):
        self._bead_index = bead_index
        self._bead_type = bead_type
        self._res_name = res_name
        self._atom_indices = atom_indices

    @property
    def bead_index(self):
        return self._bead_index

    @bead_index.setter
    def bead_index(self, bead_index):
        self._bead_index = bead_index

    @property
    def bead_type(self):
        return self._bead_type

    @bead_type.setter
    def bead_type(self, bead_type):
        self._bead_type = bead_type

    @property
    def atom_indices(self):
        return self._atom_indices

    @atom_indices.setter
    def atom_indices(self, atom_indices):
        self._atom_indices = atom_indices

    @property
    def res_name(self):
        return self._res_name

    @res_name.setter
    def res_name(self, res_name):
        self._res_name = res_name

