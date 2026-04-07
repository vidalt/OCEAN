Custom Dataset Example
======================

This example mirrors the dataset-generation style used in the test suite: mix
continuous features, ordered discrete features, binary flags, and unordered
categorical features, parse them with OCEAN, train a tree ensemble, and
explain one instance.

Why this example matters
------------------------

The packaged dataset loaders are convenient, but most real integrations start
from a custom pandas dataframe. This example shows the full path from raw data
to a readable counterfactual without depending on any external dataset.

Running the example
-------------------

.. code-block:: bash

   python examples/custom_dataset.py

What it does
------------

1. Generates a synthetic credit-style dataset with multiple feature types.
2. Uses :func:`ocean.feature.parse_features` to build the processed matrix and
   mapper.
3. Trains a random forest on that processed matrix.
4. Selects a query that the model predicts as class ``0``.
5. Uses ``ocean.ConstraintProgrammingExplainer`` to search for the closest
   class-``1`` counterfactual.
6. Prints both the original raw instance and the decoded explanation.

In this example, ``credit_lines`` is treated as an ordered discrete feature,
while ``job_type`` and ``region`` are treated as unordered categories and are
therefore one-hot encoded.

Source
------

.. literalinclude:: ../examples/custom_dataset.py
   :language: python
   :linenos:
   :caption: examples/custom_dataset.py
