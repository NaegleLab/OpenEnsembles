from distutils.core import setup
import os
import shutil

DESC = 'An open source Python package for performing, finishing, and analyzing ensemble clustering.'

PACKAGE_FILES = [
    '__init__.py',
    'openensembles.py',
    'clustering_algorithms.py',
    'cooccurrence.py',
    'finishing.py',
    'transforms.py',
    'validation.py',
    'mutualinformation.py'
]

# Create a temporary package directory and copy all src files to it
os.mkdir('openensembles')
for f in PACKAGE_FILES:
    shutil.copy(f, 'openensembles/' + f)

# Run setup
setup(
    name='OpenEnsembles',
    version='1.0.0',
    author='Naegle Lab',
    author_email='see github',
    url='https://github.com/NaegleLab/OpenEnsembles',
    packages=['openensembles'],
    install_requires=['pandas', 'numpy', 'sklearn'],
    license='GNU General Public License v3',
    summary=DESC,
    long_description=DESC,
)

# Remove package directory
shutil.rmtree('openensembles')
