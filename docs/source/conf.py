# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'micmac'
copyright = '2024, Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
author = 'Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    # 'myst_nb',
    'myst_parser',
    # 'sphinx_copybutton',
    # 'nbsphinx_link',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.mathjax',
    # 'sphinx.ext.githubpages',
    # 'sphinx_rtd_theme',
    # 'nbsphinx',
    # 'sphinx_tabs.tabs',
    # 'sphinx_git',
    # 'sphinx.ext.autosectionlabel',
    # 'sphinxemoji.sphinxemoji',
]

myst_enable_extensions = ['dollarmath', 'colon_fence']
# source_suffix = '.rst'
master_doc = 'index'

nbsphinx_execute = 'off'
nb_execution_timeout = -1

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_ignore_module_all = False
autosummary_imported_members = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'
html_extra_path = ['latex']
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ''
html_title = project
html_static_path = ['_static']

napoleon_include_init_with_doc = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
