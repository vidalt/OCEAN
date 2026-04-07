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
   Solver seed for more repeatable runs.

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
     - No
     - Yes
     - Yes
   * - Isolation forest support
     - Yes
     - No
     - No

Repeated solves
---------------

If you solve multiple queries with the same MIP explainer instance, call
``cleanup()`` after each solve to remove the temporary objective and class
constraints created for the previous query.

The CP and MaxSAT explainers already clear those query-specific constraints
inside ``explain``.

Inspecting the result
---------------------

Once a counterfactual is found, these access patterns are typically the most
useful.

- ``explanation.x`` gives the processed numerical vector.
- ``explanation.to_series()`` keeps the processed column names.
- ``explanation.value`` decodes one-hot groups into original category labels.
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
