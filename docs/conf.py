from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DOCS / "_ext"))

from ocean_docs_stubs import install as install_doc_stubs

install_doc_stubs()

project = "OCEAN"
author = "OCEAN contributors"

try:
    release = version("oceanpy")
except PackageNotFoundError:
    release = "local"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {
    "show-inheritance": True,
}
add_module_names = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_title = "OCEAN Documentation"
