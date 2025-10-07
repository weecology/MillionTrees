# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add the src directory and project root to the Python path
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, ".."))
sys.path.insert(0, os.path.abspath("../src"))

from milliontrees.version import __version__

project = 'MillionTrees'
copyright = '2024, Ben Weinstein'
author = 'Ben Weinstein'
release = __version__

master_doc = 'index'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', 
    'sphinx.ext.viewcode',     # Adds source code links
    'nbsphinx',               # For Jupyter notebooks
    'myst_parser'             # For Markdown files
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'furo'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
