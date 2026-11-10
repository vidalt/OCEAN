Explainer Guide
===============

This page focuses on the common public API shared by the three explainers and
highlights the few places where backend behavior differs.

Constructor pattern
-------------------

All public explainers follow the same constructor shape.

.. code-block:: python

   explainer = SomeExplainer(model, mapper=mapper)

The two required inputs are:

- a fitted supported tree ensemble,
- the mapper returned by :func:`ocean.feature.parse_features`.

Two optional constructor arguments are especially useful when you compare
backends:

- ``isolation=some_isolation_forest`` is supported by the MIP and CP
  explainers and adds an isolation-forest plausibility constraint.
- ``isolation_threshold=...`` can be passed together with ``isolation=...`` on
  MIP and CP. ``0.5`` reproduces the historical cutoff, values closer to ``1``
  are weaker, and smaller values are stricter.
- ``hard_voting=True`` is supported by the MaxSAT explainer for
  ``RandomForestClassifier`` and switches the score comparison from soft class
  proportions to per-tree votes.

Calling ``explain``
-------------------

Every explainer exposes an ``explain`` method with the same core arguments.

``x``
   Query instance as a one-dimensional numpy array in the processed feature
   space.

``y``
   Target class to enforce in the counterfactual.

``norm``
   Distance norm. The MIP backend supports ``1`` and ``2``. The CP backend
   supports integer norms with ``1`` as the default. The MaxSAT backend
   supports ``1``.

``max_time``
   Solver time limit in seconds.

``num_workers``
   Parallel worker count when the backend exposes it.

``random_seed``
   Solver seed for more repeatable runs. The MaxSAT backend currently accepts
   this argument for API compatibility but does not use it.

``verbose``
   Whether to print solver logs.

Actionability constraints (CP and MIP)
--------------------------------------

Pass ``actionability`` to restrict how individual features may change. The
public aliases in :mod:`ocean.typing` describe the mapping and its values:

.. code-block:: python

   from ocean.typing import Actionability, ActionabilityRule

   income_rule: ActionabilityRule = "increase"
   rules: Actionability = {
       "age": "increase",
       "income": income_rule,
       "employment_type": "fixed",
       "debt": "decrease",
   }
   explanation = explainer.explain(x, y=1, norm=1, actionability=rules)

Use feature names from ``mapper`` (strings or integers). For a one-hot encoded
feature, use the original group name, not an individual encoded column.

.. list-table:: Supported rules
   :header-rows: 1

   * - Rule
     - Meaning for the returned counterfactual
   * - ``"fixed"``
     - Keep the input value or category unchanged.
   * - ``"increase"``
     - Value must be greater than or equal to the input value.
   * - ``"decrease"``
     - Value must be less than or equal to the input value.

Unlisted features are unrestricted; ``None`` (the default) and ``{}`` add no
restrictions. Numeric and binary features support all three rules. Unordered
one-hot groups support only ``"fixed"``. Unknown names, unsupported rules,
invalid one-hot or binary query values, non-integer discrete query values,
and constrained numeric values outside the feature bounds raise ``ValueError``
before query constraints are added. When a mapping is provided, the query must
also be a finite one-dimensional array with the mapper's column count.

Rules apply in the processed feature space supplied to the explainer and to
its decoded counterfactual values. CP continuous variables represent intervals;
fixing the input's interval preserves its original value through decoding.
MIP likewise restricts the represented regions where necessary, including
when a discrete input is not one of its stored levels. The distance objective
and target-class constraint continue to apply. These rules do not encode causal
relationships between features or arbitrary category transitions.

An infeasible set of restrictions returns ``None`` with the usual solver
warning. With the default ``clean_up=True``, actionability constraints are
removed after both successful and unsuccessful solves. With
``clean_up=False``, call ``cleanup()`` before starting an independent query.
The MaxSAT backend does not support this argument.

Backend-specific behavior
-------------------------

.. list-table:: Runtime differences
   :header-rows: 1

   * - Topic
     - MIP
     - CP
     - MaxSAT
   * - Norm support
     - ``L1`` and ``L2``
     - Integer ``Lp`` norms, default ``L1``
     - ``L1`` only
   * - Anytime callback
     - Yes
     - Yes
     - No public callback list
   * - Automatic cleanup after solve
     - Yes by default
     - Yes
     - Yes
   * - Isolation forest support
     - Yes
     - Yes
     - No
   * - Hard-voting random forest mode
     - No
     - No
     - Yes

Repeated solves
---------------

All three explainers default to ``clean_up=True`` inside ``explain``. That
means query-specific objectives, target-class constraints, and CP/MIP
actionability constraints are removed
automatically after each solve unless you opt out.

Call ``cleanup()`` manually only when you deliberately run ``explain(...,
clean_up=False)`` and want to clear the previous query state yourself before
reusing the same explainer instance.

One backend detail matters when you keep old explanations around. The CP and
MaxSAT explanation objects read values back from the current shared backend
solver state, so they behave like live views rather than frozen snapshots. If
you want to preserve one result before another CP or MaxSAT solve, materialize
it immediately with ``explanation.to_numpy().copy()`` or
``explanation.to_series().copy()``.

Inspecting the result
---------------------

Once a counterfactual is found, these access patterns are typically the most
useful.

- ``explanation.x`` gives the processed numerical vector.
- ``explanation.to_series()`` keeps the processed column names.
- ``explanation.value`` decodes one-hot groups into original category labels.
- ``explainer.get_objective_value()`` returns the backend objective value for
  the last solve.
- ``explainer.get_distance()`` returns the post-processed distance between the
  query and the decoded counterfactual using the norm from the last
  ``explain(...)`` call.
- ``repr(explanation)`` is usually the best display form for notebooks and
  logs.

Handling infeasibility and time limits
--------------------------------------

All explainers return ``None`` when no counterfactual is found under the given
constraints. Depending on the backend, the solver may also warn when:

- the target class is infeasible,
- a feasible solution was found but optimality could not be certified in time,
- the solver terminated for a backend-specific reason.

When that happens, increase ``max_time``, simplify the ensemble, or choose a
query whose target class is realistically reachable under the learned model.
For MIP and CP with ``isolation=...``, also check whether
``isolation_threshold`` is too strict for the target region.
