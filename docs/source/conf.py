# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../micmac'))

# import micmac

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MICMAC'
copyright = '2024, Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
author = 'Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
language = 'en'
# version = micmac.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Create neat summary tables
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    # 'autoapi.extension',
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
autosummary_imported_members = True

# Modules to mock for the purposes of doc build.
autodoc_mock_imports = []
for missing in [
    'scipy',
    'matplotlib',
    'healpy',
    'jax',
    'numpy',
    'scipy',
    'pysm3',
    'cmbdb',
    'jaxlib',
    'jax-tqdm',
    'jax-healpy',
    'numpyro',
    'chex',
    'anytree',
    'camb',
    'lineax',
    'jaxopt',
    'toml',
    'pandas',
    'astropy',
]:
    autodoc_mock_imports.append(missing)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_book_theme'
html_extra_path = ['latex']
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ''
html_title = project
html_static_path = ['_static']
html_logo = '../../MICMAC-2.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

napoleon_include_init_with_doc = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
