"""Sphinx configuration for the live PYSTILT docs."""

from __future__ import annotations

import sys
from importlib.metadata import version as _version
from pathlib import Path

DOCS = Path(__file__).resolve().parent

sys.path.insert(0, str(DOCS / "_ext"))

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
release = _version("pystilt")  # from git tags, via setuptools-scm
version = release
# Builds from main (and local builds) are "dev"; release builds are their version.
version_match = "dev" if (".dev" in release or "+" in release) else release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "config_docs",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "api_pages",  # _ext/api_pages.py: which members a class page lists; after autodoc
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "reference/_api/*.__init__.rst",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"PYSTILT {version_match}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "announcement": (
        "PYSTILT is in alpha: names and options may change between releases."
    ),
    "github_url": "https://github.com/jmineau/PYSTILT",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navigation_depth": 3,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "header_links_before_dropdown": 6,
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version dropdown. The Documentation workflow publishes dev/ (main),
    # one folder per release and stable/, and writes switcher.json listing them.
    "switcher": {
        "json_url": "https://jmineau.github.io/PYSTILT/switcher.json",
        "version_match": version_match,
    },
    "check_switcher": False,  # switcher.json exists only on the deployed site
    "show_version_warning_banner": True,  # point old versions at the latest
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_context = {
    "github_user": "jmineau",
    "github_repo": "PYSTILT",
    "github_version": "main",
    "doc_path": "docs",
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
napoleon_use_ivar = True  # what a class page's tables leave in "Attributes"
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    "member-order": "bysource",
    "exclude-members": ",".join(sorted(PYDANTIC_AUTODOC_EXCLUDES | {"__weakref__"})),
}
autoclass_content = "class"
autodoc_class_signature = "mixed"
autodoc_typehints = "description"

autodoc_mock_imports = ["cartopy"]

autodoc_type_aliases = {
    "ProjectConfig": "stilt.config.ProjectConfig",
    "Receptor": "stilt.receptor.Receptor",
}

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True
# The function stilt.particles.background and the class Background would get
# pages whose names differ only by case, which overwrite each other on macOS
# and Windows.
autosummary_filename_map = {
    "stilt.particles.Background": "stilt.particles.Background-class"
}
set_type_checking_flag = True

# Intersphinx settings
intersphinx_mapping = {
    "arlmet": ("https://jmineau.github.io/arl-met/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pydantic": ("https://pydantic.dev/docs/validation/latest/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

typehints_fully_qualified = False


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


def trim_class_docstrings(
    app: object,
    what: str,
    name: str,
    obj: object,
    options: object,
    lines: list[str],
) -> None:
    """
    Drop a config model's Attributes and Methods sections, keeping its parameters.

    The ``config-model`` directive renders a model's fields. For other classes,
    ``api_pages`` drops only what the class page's tables list, so the
    descriptions of the rest (a ``NamedTuple``'s fields, say) stay.
    """
    if what != "class" or not hasattr(obj, "model_fields"):
        return

    drop_sections = {"Attributes", "Methods"}
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if title in drop_sections and underline and set(underline) == {"-"}:
            del lines[i:]
            break

    while lines and not lines[-1].strip():
        lines.pop()


def setup(app: object) -> None:
    """Register Sphinx hooks for API doc cleanup."""
    app.connect("autodoc-skip-member", skip_pydantic_members)
    app.connect("autodoc-process-docstring", trim_class_docstrings)
