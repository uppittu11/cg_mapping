class ResMapping:
    """
    An object to store mapping information for a molecule

    Attributes
    ----------
    name : string
        Name of the residue/molecule

    beads : tuple
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

class PolymerMapping(ResMapping):
    """
    An object to store mapping information for a polymer

    Attributes
    ----------
    name : string
        Name of the residue/molecule

    beads : tuple
        A tuple of BeadMapping objects

    bonds: tuple
        A tuple of pairs of indices for bonded CG beads

    monomers : list
        A list of ResMappings for each monomer

    """
    def __init__(self, name, monomers, sequence, bonding_sites=None):
        """
        sequence : list
            A list of monomer names in order of connection.

        bonding_sites : list, optional
            The bead for each monomer that participates in inter-monomer
            bonds

        """

        self._monomers = {monomer.name : monomer for monomer in monomers}

        if bonding_sites is None:
            bonding_sites = [0 for _ in monomers]

        bonding_sites = {monomer.name : bonding_site for
                         monomer, bonding_site in
                         zip(monomers, bonding_sites)}

        beads, bonds = self._build_polymer(bonding_sites, sequence)

        super().__init__(name, beads, bonds)

    @property
    def monomers(self):
        return self._monomers

    @monomers.setter
    def monomers(self):
        raise TypeError("'monomers' attribute does not support assignment")

    def _build_polymer(self, bonding_sites, sequence):
        beads = []
        bonds = []
        polymer_bonding_sites = []
        curr_atom = 0
        for monomer_name in sequence:
            monomer = self._monomers[monomer_name]
            curr_bead = len(beads)

            polymer_bonding_sites.append(curr_bead + bonding_sites[monomer_name])

            for bead in monomer.beads:
                name = bead.name
                new_mapping_indices = [i + curr_atom for i in bead.mapping_indices]
                new_bead = BeadMapping(name, new_mapping_indices)
                beads.append(new_bead)

            for bond in monomer.bonds:
                new_bond = tuple([i + curr_bead for i in bond])
                bonds.append(new_bond)

            curr_atom = max([max(bead.mapping_indices) for bead in beads]) + 1

        for i, j in zip(polymer_bonding_sites[1:], polymer_bonding_sites[:-1]):
            bonds.append(tuple([i, j]))

        return beads, bonds

class BeadMapping:
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
    return (int(atom_i), int(atom_j))

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
