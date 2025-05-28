# Configuration file for the Sphinx documentation builder.
import os
import sys
# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('../src'))

from milliontrees.version import __version__

project = 'MillionTrees'
copyright = '2024, Ben Weinstein'
author = 'Ben Weinstein'
release = __version__

master_doc = 'index'
extensions = ['myst_parser', 'sphinx.ext.autodoc', "sphinx.ext.napoleon", "nbsphinx"]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'furo'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
