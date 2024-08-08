# Configuration file for the Sphinx documentation builder.
import os
import sys

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '..'))
sys.path.insert(0, os.path.abspath('..'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MillionTrees'
copyright = '2024, Ben Weinstein'
author = 'Ben Weinstein'
release = '0.y.z'

master_doc = 'landing'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'sphinx.ext.autodoc']

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
