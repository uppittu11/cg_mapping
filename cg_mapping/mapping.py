class ResMapping:
    """
    An object to store mapping information for a molecule

    Attributes
    ----------
    name : string
        Name of the residue/molecule
    
    atoms : tuple
        A tuple of BeadMapping objects

    bonds: tuple
        A tuple of pairs of indices for bonded CG beads

    """
    def __init__(self, name, beads, bonds):
        self._name = name
        self._beads = tuple(beads)
        self._bonds = tuple(bonds)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self):
        raise TypeError("'name' attribute does not support assignment")

    @property
    def beads(self):
        return self._beads
    
    @beads.setter
    def beads(self):
        raise TypeError("'beads' attribute does not support assignment")

    @property
    def bonds(self):
        return self._bonds
    
    @bonds.setter
    def bonds(self):
        raise TypeError("'bonds' attribute does not support assignment")



    @classmethod
    def load(cls, name, filename):
        """ Load a forward mapping

        Parameters
        ----------
        filename : str
            Path to mapping file

        Returns
        -------
        mapping_dict : OrderedDict()
            OrderedDict (CG bead index : [beadtype, list of atom indices])

        Notes
        -----
        mapping files are ":" delimited
        col0: bead index
        col1: bead type
        col2: atom indices

        """

        mappings = []
        bonds = []

        with open(filename,'r') as f:
            for line in f:
                if line.strip():
                    if 'bond' in line.split(":")[0]:
                        bonds.append(_parse_bond(line))
                    else:
                        mappings.append(_parse_bead(line))
        
        mappings = sorted(mappings)
        bead_mappings = [entry[1] for entry in mappings]

        return cls(name, bead_mappings, bonds)
    

class BeadMapping():
    """
    An object to store mapping information for a CG bead

    Attributes
    ----------
    name : string
        The name of the CG bead

    mapping_indices: tuple
        the list of atomistic indices encompassed by the bead

    """
    def __init__(self, name, mapping_indices):
        self._name = name
        self._mapping_indices = tuple(mapping_indices)

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self):
        raise TypeError("'name' attribute does not support assignment")

    @property
    def mapping_indices(self):
        return self._mapping_indices

    @mapping_indices.setter
    def mapping_indices(self):
        raise TypeError("'mapping_indices' attribute does not support" 
                        "assignment")

def _parse_bond(line):
    """ Gets bond indices from a line in the mapping file

    Parameters
    ----------
    line : str
        line to parse

    Returns
    -------
    tuple:
        tuple containing indices of a pair of bonded beads

    Notes
    -----
    The format for a bond line is as follows:
    
    `bond : {index1} {index2}`

    """

    atom_i = line.split(":")[1].split()[0].strip()
    atom_j = line.split(":")[1].split()[1].strip()
    return (atom_i, atom_j)

def _parse_bead(line):
    """ Gets bead mapping from a line in the mapping file

    Parameters
    ----------
    line : str
        line to parse

    Returns
    -------
    tuple:
        tuple of the bead's index and the corresponding BeadMapping
        object.

    Notes
    -----
    The format for a bead line is as follows:
    
    `{bead index} : {bead name} : [{list of indices}]

    """

    bead_index = int(line.split(":")[0].strip())
    bead_name = line.split(":")[1].strip()
    mapping_indices = eval(line.split(":")[2])
    bead_mapping = BeadMapping(bead_name, mapping_indices)
    return (bead_index, bead_mapping)