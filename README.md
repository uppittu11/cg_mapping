# cg_mapping
Mapping atomistic systems to coarse-grained systems
Using MDTraj libraries and user-defined mapping files, convert an atomistic trajectory to a coarse-grained trajectory.
Currently only support gromacs file formats.

# Notes
Mapping files can be custom-named, but adding to the dict  "all_cg_mappings" requires the keys (residue names) to correspond 
to residue names found in the atomistic trajectory. Values correspond to the mapping file.
