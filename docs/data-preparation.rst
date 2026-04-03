Data Preparation
================

OCEAN expects a numerical design matrix, but it also needs enough metadata to
recover the meaning of each transformed column when it builds or prints an
explanation. The :func:`ocean.feature.parse_features` function handles both at
once by returning ``(processed_data, mapper)``.

Automatic feature typing
------------------------

For each input column, :func:`ocean.feature.parse_features` applies these rules.

.. list-table:: Parsing rules
   :header-rows: 1

   * - Input pattern
     - Resulting feature type
     - Notes
   * - Column listed in ``discretes``
     - Discrete
     - Kept numeric as an ordered or ordinal feature, with explicit levels and
       thresholds.
   * - Column listed in ``encoded``
     - One-hot encoded
     - Treated as an unordered nominal feature and expanded into indicator
       columns.
   * - Non-numeric or non-binary column
     - One-hot encoded
     - OCEAN assumes no natural order and encodes the column as unordered
       categories automatically.
   * - Column with exactly two unique values
     - Binary
     - Encoded as a single 0/1 column.
   * - Remaining numeric column
     - Continuous
     - Optionally scaled to the centered interval ``[-0.5, 0.5]``.

Cleaning behavior
-----------------

By default, preprocessing removes columns that would not be useful for the
explainers.

- ``drop_na=True`` removes columns containing missing values.
- ``drop_constant=True`` removes constant columns.
- ``scale=True`` centers continuous features in ``[-0.5, 0.5]``.

If you need to preserve a column, disable the relevant option explicitly.

Discrete versus one-hot encoded
-------------------------------

In OCEAN, these two categories are intentionally different.

``Discrete``
   Ordered or ordinal values that still carry a notion of rank or distance,
   such as integer counts, age buckets, or credit levels. These should usually
   stay numeric and be passed through ``discretes=...`` when they are not
   continuous.

``One-hot encoded``
   Unordered nominal categories, such as job titles, regions, or product
   labels, where no ordering should be assumed. These are expanded into binary
   indicator columns.

This distinction matters because the explainers may move along ordered discrete
levels differently than they switch between unordered categories.

Example
-------

.. code-block:: python

   import pandas as pd

   from ocean.feature import parse_features

   raw = pd.DataFrame({
       "age_bucket": [18, 25, 35, 45],
       "owns_home": [0, 1, 1, 0],
       "income_ratio": [0.1, 0.4, 0.7, 0.3],
       "job_type": ["office", "manual", "service", "office"],
   })

   data, mapper = parse_features(
       raw,
       discretes=("age_bucket",),
   )

   print(data.columns)
   print(mapper["age_bucket"].ftype)
   print(mapper["job_type"].codes)

Why the mapper matters
----------------------

The processed matrix seen by the ensemble may contain more columns than the raw
input because one-hot encoding expands categorical variables. The mapper stores
that relation and is required by every explainer constructor.

Without the mapper, OCEAN cannot correctly:

- associate split decisions with the right original feature,
- decode a one-hot explanation back into a category label,
- rebuild readable explanations from solver variables.

Recommended practice
--------------------

- Keep the mapper next to the trained model.
- Train and explain on the exact same processed columns.
- If you write your own preprocessing pipeline around OCEAN, make sure the
  final column order stays stable between training and explanation time.
