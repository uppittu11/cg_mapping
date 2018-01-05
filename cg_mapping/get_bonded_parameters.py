from cg_mapping import *
import numpy as np
import pandas as pd
import mdtraj
import itertools
import argparse

""" Simple script to iterate through all bead types
and compute bond/angle parameters by fitting to
Gaussian distributions"""
beadtypes=['P4', 'P3', 'Nda', 'Na', 'C2', 'C1', 'Qa', 'Q0']

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest="trajectory", help="Trajectory")
parser.add_argument("-c", dest="topology", help="File with structure/topology info")
parser.add_argument("-o", dest="output", help="Output for rdf filenames")
args = parser.parse_args()
#traj = mdtraj.load("bonded_cg-traj.xtc", top="bonded_cg-traj.pdb")
traj = mdtraj.load(args.trajectory, top=args.topology)
all_atoms = [a.name for a in traj.topology.atoms]

system_state = cg_utils.State(k_b=8.314e-3, T=900)
print("*"*20)
print(system_state)
print("*"*20)

print("*"*20)
print("Bonding parameters")
print("*"*20)
all_bonding_parameters = pd.DataFrame(columns=['#bond', 'force_constant','x0'])

for x,y in itertools.combinations_with_replacement(beadtypes, 2):
    print("---{}-{}---".format(x,y))
    bond_parameters = system_state.compute_bond_parameters(traj, x, y)
    print(bond_parameters)
    if bond_parameters:
        all_bonding_parameters.loc[len(all_bonding_parameters)] = \
            ['{}-{}'.format(x,y),
            bond_parameters['force_constant'], bond_parameters['x0']]
print(all_bonding_parameters)
all_bonding_parameters.to_csv('bond_parameters.dat', sep='\t', index=False)

print("*"*20)
print("Angle parameters")
print("*"*20)
all_angle_parameters = pd.DataFrame(columns=['#angle','force_constant', 'x0'])
for x,z in itertools.combinations_with_replacement(beadtypes, 2):
        for y in beadtypes: 
            print("{}-{}-{}: ".format(x,y,z))
            angle_parameters = system_state.compute_angle_parameters(traj, x, y, z)
            print(angle_parameters)
            if angle_parameters:
                all_angle_parameters.loc[len(all_angle_parameters)] = \
                  ['{}-{}-{}'.format(x,y,z),
                  angle_parameters['force_constant'], angle_parameters['x0']]

print(all_angle_parameters)
all_angle_parameters.to_csv('angle_parameters.dat', sep='\t', index=False)
print("*"*20)
print("Generating RDFs")
print("*"*20)
for x,y in itertools.combinations_with_replacement(beadtypes, 2):
    print("---{}-{}---".format(x,y))
    system_state.compute_rdf(traj, x,y,"{}-{}-{}".format(x,y, args.output))
