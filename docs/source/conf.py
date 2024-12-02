# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MICMAC'
copyright = '2024, Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
author = 'Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_copybutton',
    'nbsphinx_link',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinx_tabs.tabs',
    'sphinx_git',
    'sphinxcontrib.texfigure',
    'sphinx.ext.autosectionlabel',
    'sphinxemoji.sphinxemoji',
]

nbsphinx_execute = 'never'
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_numpy_docstring = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
