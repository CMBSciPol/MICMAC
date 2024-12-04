# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import os
# import sys

# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.append(os.path.abspath('../../micmac'))

# # import micmac

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MICMAC'
copyright = '2024, Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
author = 'Magdy Morshed, Arianna Rizzieri, Clément Leloup, Josquin Errard, Radek Stompor'
language = 'en'
# version = micmac.__version__

release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Core Sphinx library for auto html doc generation from docstrings
    # 'sphinx.ext.autosummary',  # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    'sphinx.ext.viewcode',  # Add a link to the Python source code for classes, functions etc.
    'sphinx_autodoc_typehints',  # Automatically document param types (less noise in class signature)
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_immaterial',
    'sphinx_design',
    # 'IPython.sphinxext.ipython_console_highlighting'
    # 'autoapi.extension',
    # 'myst_nb',
    'myst_parser',
    # 'sphinx_copybutton',
    'nbsphinx_link',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.mathjax',
    # 'sphinx.ext.githubpages',
    # 'sphinx_rtd_theme',
    'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    # 'sphinx_tabs.tabs',
    # 'sphinx_git',
    # 'sphinx.ext.autosectionlabel',
    # 'sphinxemoji.sphinxemoji',
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
}


myst_enable_extensions = ['dollarmath', 'colon_fence']
# source_suffix = '.rst'
master_doc = 'index'

nbsphinx_execute = 'never'
nb_execution_timeout = -1

# autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autosummary_ignore_module_all = False
# autosummary_imported_members = False
autoclass_content = 'both'  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
# autodoc_typehints = "description" # Sphinx-native method. Not as good as sphinx_autodoc_typehints
add_module_names = False  # Remove namespaces from class/method signatures

# Modules to mock for the purposes of doc build.
# autodoc_mock_imports = []
# for missing in [
#     'matplotlib',
#     'healpy',
#     'jax',
#     'pysm3',
#     'cmbdb',
#     'jaxlib',
#     'jax-tqdm',
#     'jax-healpy',
#     'numpyro',
#     'chex',
#     'anytree',
#     'camb',
#     'lineax',
#     'jaxopt',
#     'toml',
#     'pandas',
#     'astropy',
# ]:
#     autodoc_mock_imports.append(missing)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_immaterial'  #'sphinx_book_theme'
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
    'site_url': 'https://minimally-informed-cmb-map-constructor-micmac.readthedocs.io/en/latest/index.html',
    'repo_url': 'https://github.com/CMBSciPol/MICMAC',
    'repo_name': 'MICMAC',
    'icon': {'repo': 'fontawesome/brands/git-alt'},
    'globaltoc_collapse': False,
    'features': [
        # "navigation.expand",
        'navigation.tabs',
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        'navigation.top',
        'navigation.tracking',
        'toc.follow',
        'toc.sticky',
        'content.tabs.link',
        'announce.dismiss',
    ],
}
object_description_options = [
    ('py:.*', dict(include_fields_in_toc=False)),
]

napoleon_include_init_with_doc = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
