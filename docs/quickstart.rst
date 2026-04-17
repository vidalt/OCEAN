Quickstart
==========

The example below trains a random forest on the Adult dataset and solves the
same counterfactual query with the three public explainers.

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier

   from ocean import (
       ConstraintProgrammingExplainer,
       MaxSATExplainer,
       MixedIntegerProgramExplainer,
   )
   from ocean.datasets import load_adult

   (data, target), mapper = load_adult()
   query_frame = data.iloc[[0]]

   model = RandomForestClassifier(
       n_estimators=10,
       max_depth=3,
       random_state=42,
   )
   model.fit(data, target)

   predicted_class = int(model.predict(query_frame).item())
   query = query_frame.to_numpy().flatten()
   target_class = 1 - predicted_class

   mip = MixedIntegerProgramExplainer(model, mapper=mapper)
   cp = ConstraintProgrammingExplainer(model, mapper=mapper)
   maxsat = MaxSATExplainer(model, mapper=mapper)

   mip_cf = mip.explain(query, y=target_class, norm=1)
   cp_cf = cp.explain(query, y=target_class, norm=1)
   maxsat_cf = maxsat.explain(query, y=target_class, norm=1)

   print("MIP objective:", mip.get_objective_value())
   print("MIP distance:", mip.get_distance())
   print(mip_cf)
   print("CP objective:", cp.get_objective_value())
   print("CP distance:", cp.get_distance())
   print(cp_cf)
   print("MaxSAT objective:", maxsat.get_objective_value())
   print("MaxSAT distance:", maxsat.get_distance())
   print(maxsat_cf)

Step-by-step
------------

1. ``load_adult()`` returns both the processed training matrix and the mapper
   needed to interpret explanations.
2. The ensemble is trained on the processed matrix, not on the raw dataset.
3. The query must be passed to ``explain`` as a one-dimensional numpy array.
4. ``y`` is the desired target class, not the current prediction.
5. The returned explanation is readable directly because its string
   representation uses the decoded ``.value`` mapping.

Common variations
-----------------

- Use ``norm=2`` with the MIP backend or the CP backend.
- Pass ``isolation=some_isolation_forest`` to the MIP or CP explainer when
  you want counterfactuals to avoid highly isolated regions.
- Use ``MaxSATExplainer(model, mapper=mapper, hard_voting=True)`` when you
  want MaxSAT to match a hard-voting random forest.
- Set ``verbose=True`` to expose solver logs while debugging.
- Set ``max_time`` and ``num_workers`` when you want more control over runtime.
- Reuse the same fitted model and mapper across multiple explainers to compare
  formulations on the exact same query.

When to use the next guides
---------------------------

- Read :doc:`visual-examples` if you want a geometric picture of what a
  counterfactual looks like on small 2D problems.
- Read :doc:`data-preparation` if your data is not already in OCEAN's expected
  processed form.
- Read :doc:`explainer-guide` for backend-specific behavior and repeated-solve
  patterns.
- Read :doc:`custom-dataset` for a fully synthetic example built from scratch.
