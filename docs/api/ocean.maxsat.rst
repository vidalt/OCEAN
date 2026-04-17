Weighted MaxSAT Backend
=======================

The :mod:`ocean.maxsat` module exposes the weighted MaxSAT formulation backed
by PySAT. The main public entry point is :class:`ocean.maxsat.Explainer`; the
remaining classes document the underlying Boolean model, variable bundles, and
manager helpers used to build that formulation. For random forests, the public
explainer also supports ``hard_voting=True`` to encode majority voting at the
tree level instead of comparing averaged class proportions.

.. automodule:: ocean.maxsat
   :members:
   :undoc-members:
   :imported-members:
   :member-order: bysource
