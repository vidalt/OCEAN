Installation
============

Base package
------------

Install the published package with pip:

.. code-block:: bash

   pip install oceanpy

Runtime requirements by backend
-------------------------------

All three backends share the same preprocessing and tree parsing layers, but
the solver requirements differ.

``ocean.mip``
   Requires Gurobi at runtime. This is the backend to use if you want the MIP
   formulation, quadratic ``L2`` objectives, or isolation-forest constraints.

``ocean.cp``
   Uses OR-Tools CP-SAT. This is the simplest exact backend to run without a
   commercial license.

``ocean.maxsat``
   Uses PySAT. This backend exposes the same top-level workflow with a weighted
   MaxSAT encoding.

Development install
-------------------

If you are working from the repository, install the package in editable mode so
that examples and documentation always reflect the current source tree.

.. code-block:: bash

   pip install -e .

Build the documentation locally
-------------------------------

The repository includes a Sphinx configuration, a docs extra, and a tox target
for the reference site.

.. code-block:: bash

   pip install .[docs]
   sphinx-build -W -b html docs docs/_build/html

Or through tox:

.. code-block:: bash

   tox -e docs

Read the Docs configuration
---------------------------

The repository root contains a ``.readthedocs.yaml`` file that points Read the
Docs at ``docs/conf.py`` and installs the lightweight documentation
requirements from ``docs/requirements.txt``.

That configuration is intentionally separate from the full runtime dependency
set so the API reference can build even when every solver is not available in
the documentation environment.
