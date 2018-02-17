from distutils.core import setup

DESC = 'An open source Python package for performing, finishing, and analyzing ensemble clustering.'

# Run setup
setup(
    name='OpenEnsembles',
    version='v1.0.1',
    author='Naegle Lab',
    url='https://github.com/NaegleLab/OpenEnsembles',
    packages=['openensembles'],
    install_requires=['pandas', 'numpy', 'sklearn', 'scipy', 'networkx', 'matplotlib'],
    license='GNU General Public License v3',
    summary=DESC,
    long_description=DESC,
)

