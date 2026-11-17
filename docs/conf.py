"""Sphinx configuration of utrcalling's documentation (Read the Docs builds it from here)."""

from utrcalling._metadata import __author__, __version__

project = "utrcalling"
author = __author__
copyright = f"2020-2026, {__author__}"
version = release = __version__

extensions = [
    "myst_parser",  # pages in Markdown
    "sphinx.ext.autodoc",  # the Python reference, from docstrings
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",  # a copy button on code blocks
    "cyclopts.ext.sphinx",  # the command reference, from the commands' help
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}
copybutton_prompt_text = "$ "

html_theme = "furo"
html_title = f"utrcalling {release}"
