import setuptools
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'milliontrees'))
from version import __version__

print(f'Version {__version__}')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Remove 'sphinx', 'yapf', and any dev packages
required = [pkg for pkg in required if pkg not in ['sphinx', 'yapf','furo','myst_parser','docformatter','pytest','bumpversion','sphinx_markdown_tables'] and not pkg.startswith('dev-')]

setuptools.setup(
    name="milliontrees",
    version=__version__,
    author="Ben Weinstein",
    author_email="ben.weinstein@weecology.org",
    url="https://milliontrees.idtrees.org/",
    description="MillionTrees Benchmark for Airborne Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    license='MIT',
    packages=setuptools.find_packages(exclude=['data_prep', 'examples']),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
)
