# cg_mapping
[![Build Status](https://dev.azure.com/pshama/cg_mapping/_apis/build/status/uppittu11.cg_mapping?branchName=master)](https://dev.azure.com/pshama/cg_mapping/_build/latest?definitionId=2&branchName=master)
[![codecov](https://codecov.io/gh/uppittu11/cg_mapping/branch/master/graph/badge.svg)](https://codecov.io/gh/uppittu11/cg_mapping)

### Mapping atomistic systems to coarse-grained systems

Convert an atomistic trajectory to a coarse-grained trajectory using MDTraj and user-defined mapping files.

## Usage
The CG mapping module can be scripted and run from a python interpreter.
Here is an example: 

```python3
import mdtraj as md
from cg_mapping.mapper import Mapper

# Load in the mdtraj.Trajectory from disk
atomistic_trajectory = md.load("traj.xtc", top="start.gro")

# Initialize mapper and load in default mappings
my_mapper = Mapper()
my_mapper.load_mapping_dir()

# Load atomistic trajectory into the mapper
my_mapper.load_trajectory(atomistic_trajectory)

# Map the atomistic trajectory to the CG level
cg_trajectory = my_mapper.cg_map()

# Save the CG trajectory out to disk
cg_trajectory[0].save("cg_traj.gro")
cg_trajectory.save("cg_traj.xtc")
```