Shared Typing Helpers
=====================

The :mod:`ocean.typing` module collects the shared protocols and type aliases
used across the public explainers, tree parsing helpers, and internal backend
implementations. It is the reference page for supported ensemble types,
processed-array aliases, and the common explanation and explainer protocols.

Actionability aliases
---------------------

``ActionabilityRule`` is ``Literal["fixed", "increase", "decrease"]``.
``Actionability`` is ``Mapping[Key, ActionabilityRule]``, where ``Key`` is
``int | str``. Both aliases are importable from ``ocean.typing``. The mapping
is accepted by CP and MIP ``explain`` and ``add_actionability`` methods;
MaxSAT and the common ``BaseExplainer`` protocol retain their core API.
See :doc:`../explainer-guide` for rule semantics and examples.

.. automodule:: ocean.typing
   :members:
   :undoc-members:
   :member-order: bysource
