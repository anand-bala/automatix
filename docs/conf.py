import os
from pathlib import Path

os.environ.setdefault("PLUM_SIMPLE_DOC", "1")

CURRENT_FILE_PATH = Path(__file__).absolute()
DOC_SRC_DIR = CURRENT_FILE_PATH.parent
assert DOC_SRC_DIR.is_dir()
BUILD_DIR = DOC_SRC_DIR / "_build"


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


# -- autodoc signature fix for BetterABCMeta ---------------------------------
# BetterABCMeta overrides __call__ with (*args, **kwargs), which Sphinx picks
# up instead of the dataclass __init__.  When that happens, read __init__
# directly so the rendered signature shows the actual field names.


def _fix_metaclass_signature(
    app: object,
    what: str,
    name: str,
    obj: object,
    options: object,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    import inspect

    if what == "class" and signature in ("(*args, **kwargs)", "(*args: Any, **kwargs: Any)"):
        try:
            init = getattr(obj, "__init__", None)
            if init is None:
                return None
            sig = inspect.signature(init)
            params = [p for name_, p in sig.parameters.items() if name_ != "self"]
            new_sig = str(inspect.Signature(params))
            return new_sig, return_annotation
        except (ValueError, TypeError):
            pass
    return None


def setup(app: object) -> None:
    app.connect("autodoc-process-signature", _fix_metaclass_signature)  # type: ignore[union-attr]


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "globaltoc_maxdepth": 2,
    "globaltoc_collapse": False,
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "show_toc_level": 4,
    "secondary_sidebar_items": ["page-toc"],
    "primary_sidebar_end": ["indices"],
    "navbar_center": [],
    # "navbar_persistent": [],
    # "navbar_end": ["search-button", "navbar-icon-links"],
}

html_sidebars = {
    "**": ["globaltoc"],
}
