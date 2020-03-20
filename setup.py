from setuptools import setup, find_packages
import sys

try:
    import mdtraj
except ImportError:
    print('Building and running analysis requires mdtraj. See '
          'http://mdtraj.org/latest/installation.html for help!')
    sys.exit(1)

setup(name='analysis',
      version='0.1',
      description=('Using MDTraj libraries and user-defined mapping files,'
          ' convert an atomistic trajectory to a coarse-grained trajectory')
      url='https://github.com/ahy3nz/cg_mapping',
      author='Alexander H. Yang',
      author_email='alexander.h.yang@vanderbilt.edu',
      license='MIT',
      packages=find_packages(),
      package_dir={'cg_mapping': 'cg_mapping'},
      include_package_data=True,
      install_requires=["mdtraj", "mbuild", "foyer", "scipy", ],
      entry_points={
          "console_scripts" : ["map_traj=bin.map_traj:main"],
          },
)
