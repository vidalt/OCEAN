Custom Dataset Example
======================

This page turns the synthetic custom-dataset example into a notebook-style
walkthrough. It starts from a raw pandas dataframe, parses the feature types
with OCEAN, trains a random forest, and explains one class-``0`` prediction
with the constraint-programming backend.

Why this example matters
------------------------

The packaged dataset loaders are useful when you want a quick start, but most
real integrations begin with a dataframe that you already own. This example
shows the full workflow on mixed feature types:

- ordered discrete values through ``credit_lines``,
- binary flags through ``owns_home`` and ``has_guarantor``,
- continuous ratios through ``income_ratio``, ``debt_ratio``, and
  ``savings_ratio``,
- unordered nominal features through ``job_type`` and ``region``.

Cell 1: Build a mixed-type dataframe
------------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(42)
   raw = pd.DataFrame({
       "credit_lines": rng.choice([0, 1, 2, 4], size=300),
       "owns_home": rng.integers(0, 2, size=300),
       "has_guarantor": rng.integers(0, 2, size=300),
       "income_ratio": rng.uniform(-0.4, 0.8, size=300),
       "debt_ratio": rng.uniform(0.0, 1.0, size=300),
       "savings_ratio": rng.uniform(-0.5, 0.6, size=300),
       "job_type": rng.choice(
           ["office", "manual", "service", "student"],
           size=300,
       ),
       "region": rng.choice(
           ["north", "south", "east", "west"],
           size=300,
       ),
   })

   score = (
       (raw["credit_lines"] >= 2).astype(int)
       + raw["owns_home"].astype(int)
       + raw["has_guarantor"].astype(int)
       + (raw["income_ratio"] > 0.1).astype(int)
       + (raw["savings_ratio"] > 0.0).astype(int)
       + raw["job_type"].isin(["office", "service"]).astype(int)
       + raw["region"].isin(["north", "east"]).astype(int)
       - (raw["debt_ratio"] > 0.55).astype(int)
   )
   target = (score >= 4).astype(int).rename("approved")

Cell 2: Parse the features with OCEAN
-------------------------------------

.. code-block:: python

   from ocean.feature import parse_features

   data, mapper = parse_features(raw, discretes=("credit_lines",))
   print(data.columns)

.. code-block:: text

   MultiIndex([(  'credit_lines',        ''),
               (     'owns_home',        ''),
               ( 'has_guarantor',        ''),
               (  'income_ratio',        ''),
               (    'debt_ratio',        ''),
               ( 'savings_ratio',        ''),
               (      'job_type',  'manual'),
               (      'job_type',  'office'),
               (      'job_type', 'service'),
               (      'job_type', 'student'),
               (        'region',    'east'),
               (        'region',   'north'),
               (        'region',   'south'),
               (        'region',    'west')],
              )

The important part is that ``credit_lines`` stays ordered and numeric, while
``job_type`` and ``region`` expand into one-hot blocks.

Cell 3: Fit a classifier and choose a query
-------------------------------------------

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier

   model = RandomForestClassifier(
       n_estimators=40,
       max_depth=4,
       random_state=42,
   )
   model.fit(data, target)

   predictions = pd.Series(model.predict(data), index=data.index)
   query_index = predictions[predictions == 0].index[0]
   query = data.loc[query_index].to_numpy(dtype=float).flatten()
   query_frame = data.loc[[query_index]]
   raw_query = raw.loc[query_index]

   print(raw_query)
   print()
   print("Model prediction:", int(model.predict(query_frame).item()))

.. code-block:: text

   credit_lines            0
   owns_home               0
   has_guarantor           1
   income_ratio    -0.349179
   debt_ratio       0.260349
   savings_ratio    0.234634
   job_type          student
   region               west
   Name: 0, dtype: object

   Model prediction: 0

Cell 4: Explain the query
-------------------------

.. code-block:: python

   from ocean import ConstraintProgrammingExplainer

   explainer = ConstraintProgrammingExplainer(model, mapper=mapper)
   explanation = explainer.explain(
       query,
       y=1,
       norm=1,
       max_time=10,
       num_workers=1,
       random_seed=42,
   )
   if explanation is None:
       raise RuntimeError("No counterfactual was found for the synthetic example.")

   counterfactual_frame = pd.DataFrame(
       [explanation.to_numpy()],
       columns=data.columns,
   )

   print("Target class:", 1)
   print("Counterfactual prediction:", int(model.predict(counterfactual_frame).item()))

.. code-block:: text

   Target class: 1
   Counterfactual prediction: 1

Cell 5: Inspect the decoded explanation
---------------------------------------

.. code-block:: python

   print(explanation)

.. code-block:: text

   Explanation:
   credit_lines   : 0.0
   owns_home      : 0
   has_guarantor  : 1
   income_ratio   : -0.2833683341741562
   debt_ratio     : -0.1887158378958702
   savings_ratio  : 0.29842646420001984
   job_type       : student
   region         : north

This decoded view is usually the most readable one: categorical one-hot blocks
are mapped back to labels, and the keys match the original dataframe columns.

Cell 6: Inspect the processed vector and the final distance
-----------------------------------------------------------

.. code-block:: python

   print(explanation.to_series())
   print()
   print("Distance:", explainer.get_distance())

.. code-block:: text

   credit_lines              0.000000
   owns_home                 0.000000
   has_guarantor             1.000000
   income_ratio             -0.283368
   debt_ratio               -0.188716
   savings_ratio             0.298426
   job_type       manual     0.000000
                  office     0.000000
                  service    0.000000
                  student    1.000000
   region         east       0.000000
                  north      1.000000
                  south      0.000000
                  west       0.000000
   dtype: float64

   Distance: 1.3566914800296987

``get_distance()`` is the user-facing metric to report here: it reconstructs
the post-processed :math:`L_1` distance between the original query and the
decoded counterfactual, including the half-weight treatment for one-hot blocks.

Full script
-----------

If you want the exact runnable version behind this page:

.. literalinclude:: ../examples/custom_dataset.py
   :language: python
   :linenos:
   :caption: examples/custom_dataset.py
