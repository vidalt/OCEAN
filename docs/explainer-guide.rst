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
     - No
     - No

Repeated solves
---------------

All three explainers default to ``clean_up=True`` inside ``explain``. That
means query-specific objectives and target-class constraints are removed
automatically after each solve unless you opt out.

Call ``cleanup()`` manually only when you deliberately run ``explain(...,
clean_up=False)`` and want to clear the previous query state yourself before
reusing the same explainer instance.

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
