class CGBead:
    def __init__(self,
            bead_type=None,
            atom_indices=None):
        self._bead_type = bead_type
        self._atom_indices = atom_indices

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