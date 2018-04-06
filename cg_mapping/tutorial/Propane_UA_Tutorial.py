
# coding: utf-8

# # United-Atom Propane Tutorial
# We're going to recover propane's bonded parameters from a 
# short gromacs simulation (the relevant setup and simulation files
# can be found in sim_files). 
# 
# In order to run this simulation, we had to specify the bonded 
# parameters, so we're just going to see if we can verify these
# parameters from a simulation.
# 
# In reality with this set of tools, you would generate an atomistic
# trajectory, map the system to coarse-grained one, and then
# identify the (unknown) bonded parameters.

# First, we need to construct an MDTraj Trajectory object, making
# sure to pass a trajectory file and a topology file
# *that includes bonds*

# In[1]:

from cg_mapping import *
import numpy as np
import pandas as pd
import mdtraj
import itertools


traj = mdtraj.load("npt_b.xtc", top="npt_b.pdb")


# Next, we need to identify all the beadtypes in this system

# In[2]:

beadtypes = set([a.name for a in traj.topology.atoms])


# ## Thermodynamic Setup
# Construct a State object, which is really just a way to keep 
# track of our units and temperature. In this case, we will be using $k_b = 8.314e{-3} kJ mol^{-1} K^{-1}$ and 305 Kelvin.

# In[3]:

system_state = cg_utils.State(k_b=8.314e-3, T=305)


# ## Bond stretching parameters
# To store all this information, we will be using Pandas DataFrames.
# The bonds are computed behind-the-scenes, one bond type at a time
# 1. Construct a probability distribution of bonds $P(r)$
# 2. Perform a Boltzmann inversion to calculate the potential energy of these bonds, $V(x) = -k_b * T * ln(P(x)) = \frac{1}{2}K_b(x-x_0)^{2}$
# 3. Fit a gaussian function to the probability distribution centered around the energetic minimum, $P(x) = \frac{A}{(w\sqrt{\pi /2})} * e^{\frac{-2(x-x_0)^{2}}{w^{2}}}$
# 4. From the fitted parameters, the force constant $K_b = \frac{4*k_b * T}{w^{2}}$ and the reference distance is $x_0$
# 5. Bond distributions and energies are plotted for visualization and verification

# In[4]:

all_bonding_parameters = pd.DataFrame(columns=['#bond', 'force_constant','x0'])

for x,y in itertools.combinations_with_replacement(beadtypes, 2):
    print("---{}-{}---".format(x,y))
    bond_parameters = system_state.compute_bond_parameters(traj, x, y, plot=True)
    if bond_parameters:
        all_bonding_parameters.loc[len(all_bonding_parameters)] =             ['{}-{}'.format(x,y),
            bond_parameters['force_constant'], bond_parameters['x0']]
print(all_bonding_parameters)
all_bonding_parameters.to_csv('bond_parameters.dat', sep='\t', index=False)


# For reference, `prop.itp` says the reference distance is $0.154 nm$ and the force constant is $502416 kJ mol^{-1} nm^{-1}$

# ## Bond bending parameters
# To store all this information, we will be using Pandas DataFrames.
# The angles (in radians) are computed behind-the-scenes, one angle type at a time
# 1. Construct a probability distribution of angles $P(\theta)$
# 2. Perform a Boltzmann inversion to calculate the potential energy of these angles, noting the additional weighting factor, $V(\theta) = -k_b * T * ln(p(\theta)) = \frac{1}{2} K_\theta*(\theta-\theta_0)^{2}$, where $p(\theta) = \frac{P(\theta)}{\sin({\theta})}$
# 3. Fit a gaussian function to the probability distribution centered around the energetic minimum, $P(\theta) = \frac{A}{w\sqrt{\pi /2}} * e^{\frac{-2 * (\theta-\theta_0)^{2}}{w^{2}}}$
# Note: due to the weighting factor, the gaussian distribution is somewhat skewed, so the distribution may be mirrored to provide a better energy well for fitting (see code `compute_angle_parameters()` in `cg_utils.py` for more detail)
# 4. From the fitted parameters, the force constant $K_\theta = \frac{4*k_b * T}{w^{2}}$ and the reference angle is $\theta_0$
# 5. Angle distributions and energies are plotted for visualization and verification

# In[5]:

all_angle_parameters = pd.DataFrame(columns=['#angle','force_constant', 'x0'])
for x,z in itertools.combinations_with_replacement(beadtypes, 2):
        for y in beadtypes: 
            print("{}-{}-{}: ".format(x,y,z))
            angle_parameters = system_state.compute_angle_parameters(traj, x, y, z, plot=True)
            print(angle_parameters)
            if angle_parameters:
                all_angle_parameters.loc[len(all_angle_parameters)] =                   ['{}-{}-{}'.format(x,y,z),
                  angle_parameters['force_constant'], angle_parameters['x0']]

print(all_angle_parameters)
all_angle_parameters.to_csv('angle_parameters.dat', sep='\t', index=False)


# For reference the reference angle is $120deg (2.0944 rad)$ with a force constant of $519.654 kJ mol^{-1} deg^{-1}$

# ## Computing RDFs
# To compute nonbonded interactions, a Boltzmann inversion of the radial distribution function (RDF) is necessary. However, given the highly correlated nature of nonbonded interactions, an iterative approach is necessary (see multistate, iterative Boltzmann inversion developed by Timothy Moore)

# In[7]:

for x,y in itertools.combinations_with_replacement(beadtypes, 2):
    print("---{}-{}---".format(x,y))
    system_state.compute_rdf(traj, x,y,"{}-{}-{}".format(x,y, "state"))

