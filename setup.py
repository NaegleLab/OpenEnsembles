from distutils.core import setup

DESC = 'An open source Python package for performing, finishing, and analyzing ensemble clustering.'

# Run setup
setup(
    name='OpenEnsembles',
    version='v2.0.0',
    author='Naegle Lab',
    url='https://github.com/NaegleLab/OpenEnsembles',
    packages=['openensembles'],
    install_requires=['pandas==1.2.*', 'numpy==1.19.*', 'scikit-learn==0.24.*', 'scipy==1.6.*', 'networkx==2.5', 'matplotlib==3.3.*', 'hdbscan==0.8.*'],
    license='GNU General Public License v3',
    summary=DESC,
    long_description=DESC,
)

