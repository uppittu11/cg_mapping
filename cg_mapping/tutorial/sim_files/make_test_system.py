import mbuild as mb
import foyer
import numpy
import mdtraj
import pdb

single_propane = mb.load('propane_ua.mol2')
single_propane.name="prop"
#system = mb.fill_box(single_propane, n_compounds = 100, density = 1.0)
cube = mb.Grid3DPattern(10,10,10)
cube.scale([5,5,5])
cube_list = cube.apply(single_propane)
system = mb.Compound()
for compound in cube_list:
    system.add(compound)

system.save('propane.gro', overwrite=True, residues=['prop'])
#system.save('propane.top', forcefield_files='small_ff.xml', overwrite=True)

