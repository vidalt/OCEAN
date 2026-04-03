Top-level Package
=================

The root :mod:`ocean` package re-exports the three main explainer classes used
by most applications:

- ``ocean.MixedIntegerProgramExplainer`` wraps :class:`ocean.mip.Explainer`
- ``ocean.ConstraintProgrammingExplainer`` wraps :class:`ocean.cp.Explainer`
- ``ocean.MaxSATExplainer`` wraps :class:`ocean.maxsat.Explainer`

It also exposes the public subpackages documented in the rest of this API
reference.

Typical imports
---------------

.. code-block:: python

   from ocean import ConstraintProgrammingExplainer
   from ocean.datasets import load_adult
   from ocean.feature import parse_features
