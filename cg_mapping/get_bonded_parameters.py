from cg_mapping import *
import mdtraj
import itertools
import argparse

""" Simple script to iterate through all bead types
and compute bond/angle parameters by fitting to
Gaussian distributions"""
beadtypes=['P4', 'P3', 'Nda', 'Na', 'C1', 'Qa', 'Q0']

parser = argparse.ArgumentParser()
parser.add_argument("-f", dest="trajectory", help="Trajectory")
parser.add_argument("-c", dest="topology", help="File with structure/topology info")
parser.add_argument("-o", dest="output", help="Output for rdf filenames")
args = parser.parse_args()
#traj = mdtraj.load("bonded_cg-traj.xtc", top="bonded_cg-traj.pdb")
traj = mdtraj.load(args.trajectory, top=args.topology)
all_atoms = [a.name for a in traj.topology.atoms]

bulk_DSPC_900K = cg_utils.State(k_b=8.314e-3, T=900)
print("*"*20)
print(bulk_DSPC_900K)
print("*"*20)

print("*"*20)
print("Bonding parameters")
print("*"*20)
for x,y in itertools.product(beadtypes,repeat=2):
    print("---{}-{}---".format(x,y))
    bond_parameters = bulk_DSPC_900K.compute_bond_parameters(traj, x, y)
    print(bond_parameters)

print("*"*20)
print("Angle parameters")
print("*"*20)
for x,z in itertools.product(beadtypes,repeat=2):
        for y in beadtypes: 
            print("{}-{}-{}: ".format(x,y,z))
            angle_parameters = bulk_DSPC_900K.compute_angle_parameters(traj, x, y, z)
            print(angle_parameters)

print("*"*20)
print("Generating RDFs")
print("*"*20)
for x,y in itertools.product(beadtypes,repeat=2):
    bulk_DSPC_900K.compute_rdf(x,y,"{}-{}-{}".format(x,y, args.output))
