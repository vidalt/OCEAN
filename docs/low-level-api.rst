Low-level Internals
===================

This page documents the low-level helper classes that connect processed tabular
columns to backend variables and tree encodings.

The notation is the same as in :doc:`modelisation`:

- :math:`x` is the counterfactual explanation,
- :math:`\hat{x}` is the query instance,
- :math:`f` is the ensemble decision function,
- :math:`p_{t,\ell}` denotes the active leaf or path variable,
- :math:`\varepsilon_c` is the tie-breaking margin.

Mapper utilities
----------------

.. autoclass:: ocean.abc._mapper.Indexer
   :members: __init__, get
   :member-order: bysource

.. autoclass:: ocean.abc._mapper.Mapper
   :members: __init__, reduce, apply, __len__, __iter__, __getitem__, _get_args, _validate_args, _add_indexer, _add_getter, _get_with_name, _get_with_code, _repr
   :member-order: bysource
   :no-index:

MIP managers
------------

.. autoclass:: ocean.mip._managers._feature.FeatureManager
   :members: build_features, vget, _set_mapper
   :member-order: bysource

.. autoclass:: ocean.mip._managers._tree.TreeManager
   :members: build_trees, weighted_function, xgb_margin_function, _set_trees, _set_weights, _get_length, _get_function
   :member-order: bysource

.. autoclass:: ocean.mip._managers._garbage.GarbageManager
   :members: add_garbage, remove_garbage
   :member-order: bysource

CP managers
-----------

.. autoclass:: ocean.cp._managers._feature.FeatureManager
   :members: build_features, vget, _set_mapper
   :member-order: bysource

.. autoclass:: ocean.cp._managers._tree.TreeManager
   :members: build_trees, weighted_function, isolators, length, max_samples, min_average_length, min_length, min_length_scaled, _set_trees, _set_weights, _get_function
   :member-order: bysource

.. autoclass:: ocean.cp._managers._garbage.GarbageManager
   :members: add_garbage, remove_garbage
   :member-order: bysource

MaxSAT managers
---------------

.. autoclass:: ocean.maxsat._managers._feature.FeatureManager
   :members: build_features, vget, _set_mapper
   :member-order: bysource

.. autoclass:: ocean.maxsat._managers._tree.TreeManager
   :members: build_trees, weighted_function, _hard_voting_function, _set_trees, _set_weights, _get_function
   :member-order: bysource

.. autoclass:: ocean.maxsat._managers._garbage.GarbageManager
   :members: add_garbage, remove_garbage, garbage_list
   :member-order: bysource
