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
copyright = "2026, Awa Khouna and the OCEAN contributors"

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
nitpick_ignore_regex = [
    ("py:class", r"0\.0|1\.0"),
    ("py:class", r"Args|GarbageObject|Getter|Objective|TreeVar"),
    ("py:class", r"S|U|V"),
    ("py:class", r"FieldInfo|Ge|Gt|Lt|NoneType|optional"),
    ("py:class", r"Parsed"),
    ("py:class", r"abc\.ABC|collections\.abc\.Mapping|enum\.Enum"),
    ("py:class", r"pd\.DataFrame"),
    (
        "py:class",
        (
            r"anytree\.node\.nodemixin\.NodeMixin"
            r"|cp\..*"
            r"|gp\..*"
            r"|gurobipy\..*"
            r"|ortools\.sat\.python\.cp_model\..*"
            r"|pysat\.formula\.WCNF"
        ),
    ),
    (
        "py:class",
        (
            r"numpy\.int64"
            r"|pandas\.core\.indexes\.base\.Index"
            r"|pandas\.core\.indexes\.multi\.MultiIndex"
            r"|sklearn\.ensemble\._iforest\.IsolationForest"
            r"|xgboost\.core\.Booster"
        ),
    ),
    (
        "py:class",
        (
            r"ocean\.cp\._base\.Var"
            r"|ocean\.cp\._builder\.model\.ModelBuilder"
            r"|ocean\.feature\._keeper\.FeatureKeeper"
            r"|ocean\.maxsat\._base\.Var"
            r"|ocean\.mip\._base\.Var"
            r"|ocean\.tree\._keeper\.TreeKeeper"
            r"|ocean\.typing\.BaseExplainableEnsemble"
            r"|ocean\.typing\.BaseExplainer"
            r"|ocean\.typing\.BaseExplanation"
        ),
    ),
    ("py:obj", r"typing\.V"),
]

html_theme = "sphinx_rtd_theme"
html_title = "OCEAN Documentation"
