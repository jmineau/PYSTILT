# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

PYDANTIC_AUTODOC_EXCLUDES = {
    "model_config",
    "model_fields",
    "model_computed_fields",
    "model_extra",
    "model_fields_set",
    "DEFAULT_TARGET",
}

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PYSTILT"
copyright = "2026, James Mineau"
author = "James Mineau"
release = "0.1.0a1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/_autosummary/kubernetes*.rst",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "PYSTILT"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/jmineau/PYSTILT",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navigation_depth": 3,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": ",".join(sorted(PYDANTIC_AUTODOC_EXCLUDES | {"__weakref__"})),
}

autodoc_type_aliases = {
    "ModelConfig": "stilt.config.ModelConfig",
    "Receptor": "stilt.receptor.Receptor",
    "Trajectories": "stilt.trajectory.Trajectories",
}

# Autosummary settings
# The checked-in pages under docs/api/_autosummary are the source of truth.
autosummary_generate = False
autosummary_generate_overwrite = False

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "pyyaml": ("https://pyyaml.org/wiki/PyYAMLDocumentation", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}


def skip_pydantic_members(
    app: object,
    what: str,
    name: str,
    obj: object,
    skip: bool,
    options: object,
) -> bool | None:
    """Hide Pydantic implementation attributes from generated API docs."""
    if name in PYDANTIC_AUTODOC_EXCLUDES or name.startswith("__pydantic_"):
        return True
    return None


def setup(app: object) -> None:
    """Register Sphinx hooks for API doc cleanup."""
    app.connect("autodoc-skip-member", skip_pydantic_members)
