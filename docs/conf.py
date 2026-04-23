"""Sphinx configuration for the live PYSTILT docs."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 docs builds
    import tomli as tomllib

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
SRC = ROOT / "src"

sys.path.insert(0, str(DOCS / "_ext"))
sys.path.insert(0, str(SRC))

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
release = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
version = release

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
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "api/_autosummary/kubernetes*.rst",
    "reference/api",
    "reference/api/*",
    "reference/api/**",
    "reference/_api/*.__init__.rst",
    "reference/generated",
    "reference/generated/*",
    "reference/generated/**",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"PYSTILT {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "announcement": (
        "PYSTILT is in alpha. Expect API and workflow changes while the "
        "execution and observation layers settle."
    ),
    "github_url": "https://github.com/jmineau/PYSTILT",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navigation_depth": 3,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "header_links_before_dropdown": 6,
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
napoleon_use_ivar = False
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
    "ModelConfig": "stilt.config.ModelConfig",
    "Receptor": "stilt.receptor.Receptor",
    "Trajectories": "stilt.trajectory.Trajectories",
}

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True
set_type_checking_flag = True

# Intersphinx settings
intersphinx_mapping = {
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
    """Keep generated class pages focused while preserving parameter docs."""
    if what != "class":
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
