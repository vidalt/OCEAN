Overview
========

OCEAN turns fitted tree ensembles into optimization models that search for the
closest counterfactual satisfying a target prediction. The library is centered
around a simple idea: parse the ensemble into a solver-friendly tree structure,
link those tree decisions to feature variables, and optimize the smallest
change that flips the model output.

For the mathematical formulation behind that workflow, see
:doc:`modelisation`.

What OCEAN expects
------------------

OCEAN works best when you separate the workflow into two stages:

1. Convert raw tabular data into a numerical matrix with
   :func:`ocean.feature.parse_features`.
2. Train a supported tree ensemble on that processed matrix and keep the
   returned mapper alongside the fitted model.

The mapper is the bridge between the transformed columns seen by the ensemble
and the original feature names used by the explanation objects.

Supported models
----------------

At the public explainer level, OCEAN supports these fitted classifiers:

- ``sklearn.ensemble.RandomForestClassifier``
- ``sklearn.ensemble.AdaBoostClassifier``
- ``xgboost.XGBClassifier``

At the lower-level tree parsing layer, ``xgboost.Booster`` and
``sklearn.ensemble.IsolationForest`` are also supported where the backend uses
those structures.

Backend summary
---------------

.. list-table:: Backend comparison
   :header-rows: 1

   * - Backend
     - Public class
     - Supported norms
     - Notes
   * - MIP
     - ``ocean.MixedIntegerProgramExplainer``
     - ``1`` and ``2``
     - Requires Gurobi. Also supports adding isolation-forest constraints.
   * - CP
     - ``ocean.ConstraintProgrammingExplainer``
     - ``1``
     - Uses OR-Tools CP-SAT and is the easiest exact backend to run locally.
   * - MaxSAT
     - ``ocean.MaxSATExplainer``
     - ``1``
     - Uses a weighted MaxSAT encoding backed by PySAT.

Common workflow
---------------

1. Prepare data with :func:`ocean.feature.parse_features` or a packaged dataset
   loader such as :func:`ocean.datasets.load_adult`.
2. Fit a supported ensemble on the processed matrix.
3. Instantiate one of the public explainers from :mod:`ocean` with the model
   and mapper.
4. Select a query ``x`` as a one-dimensional numpy array in the processed
   feature space.
5. Call ``explainer.explain(x, y=target_class, norm=...)``.
6. Inspect the result through ``explanation.x``, ``explanation.to_series()``,
   or the more human-readable ``explanation.value`` mapping.

What the explanation object gives back
--------------------------------------

Every backend returns a backend-specific explanation object, but the user-level
surface is intentionally similar.

- ``.x`` returns the counterfactual as a flat numpy array aligned with the
  processed training columns.
- ``.to_series()`` returns the same information as a pandas series.
- ``.value`` returns a mapping keyed by the original feature names, decoding
  one-hot encoded groups back into a categorical value when possible.
- ``repr(explanation)`` prints that value-oriented mapping, which is usually
  the most readable form for reports and notebooks.

If you want to solve multiple queries with the same MIP explainer instance,
call ``cleanup()`` between solves. The CP and MaxSAT explainers already clear
query-specific constraints after each solve.
