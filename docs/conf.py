import os
from pathlib import Path

os.environ.setdefault("PLUM_SIMPLE_DOC", "1")

CURRENT_FILE_PATH = Path(__file__).absolute()
DOC_SRC_DIR = CURRENT_FILE_PATH.parent
assert DOC_SRC_DIR.is_dir()
BUILD_DIR = DOC_SRC_DIR / "_build"
STATIC_DIR = DOC_SRC_DIR / "_static"


# -- Project information -----------------------------------------------------

copyright = "2026, Anand Balakrishnan"
author = "Anand Balakrishnan"

subprojects = (
    "automatix",
    "morphata",
    "algebraic",
)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    # "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping: dict[str, tuple[str, None | str | tuple[str, tuple[str | None, ...]]]]
intersphinx_mapping = {
    # "python": ("https://docs.python.org/3", None),
    # "numpy": ("https://numpy.org/doc/stable/", None),
    # "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox", None),
}

intersphinx_mapping.update(
    {subproject: (f"../{subproject}", str(BUILD_DIR / "html" / subproject / "objects.inv")) for subproject in subprojects}
)

# -- Napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------

# html_theme = "pydata_sphinx_theme"
html_theme = "alabaster"
html_static_path = [str(STATIC_DIR)]

html_theme_options = {
    "globaltoc_maxdepth": 4,
    "globaltoc_collapse": False,
    "navigation_with_keys": True,
    "sidebar_collapse": True,
    "show_relbars": True,
    "extra_nav_links": {
        "Repository": "https://git.anandb.dev/automatix.git/about/",
    },
}

html_sidebars = {
    "**": [
        "about.html",
        "searchfield.html",
        "navigation.html",
        "relations.html",
        "donate.html",
    ]
}
