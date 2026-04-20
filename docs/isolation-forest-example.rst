Isolation Forest Example
========================

Suppose you are explaining a rejected application. The classifier has learned
two different ways to become approved:

- a tiny pocket close to the query, created by a single unusual training point,
- and a larger dense approved region farther to the right.

If we ask only for the shortest valid counterfactual, the optimizer goes to
the tiny pocket. Adding an isolation forest changes the feasible set: the
counterfactual must still flip the prediction, but it must also stay in a
region that looks typical with respect to the training data.

The figure below comes from ``examples/isolation_forest_example.py``. It shows
the MIP geometry for four thresholds. The script also solves the same query
with CP and prints both backends side by side.

.. image:: _static/figures/isolation-forest-example-2d.svg
   :alt: Four-panel two-dimensional random-forest example with threshold-specific isolation contours, a query, and MIP counterfactuals with and without isolation.
   :width: 98%
   :align: center

How to read the dashed contour
------------------------------

For a threshold :math:`\tau`, the plotted contour is not always the zero level
of the isolation forest. It is the threshold-specific boundary

.. math::

   \texttt{decision\_function}(x) = -\tau - \texttt{offset\_}.

Points on the dense side of that curve satisfy the isolation cutoff; points on
the other side are too isolated for that threshold.

This is why the four panels differ:

- ``0.9`` is weak, so the tiny pocket is still allowed.
- ``0.51`` is already strong enough to move the counterfactual inward.
- ``0.5`` is the historical cutoff and now excludes the tiny pocket.
- ``0.1`` is so strict that no target-class point survives.

Backend comparison
------------------

.. list-table:: MIP and CP on the same query
   :header-rows: 1

   * - Case
     - MIP status
     - MIP counterfactual
     - MIP :math:`L_1`
     - MIP isolation score
     - CP status
     - CP counterfactual
     - CP :math:`L_1`
     - CP isolation score
   * - Plain
     - ``OPTIMAL``
     - ``[0.331773, 0.864821]``
     - ``0.546594``
     - ``-0.050441``
     - ``OPTIMAL``
     - ``[0.331773, 0.864821]``
     - ``0.546594``
     - ``-0.050441``
   * - Isolation ``= 0.9``
     - ``OPTIMAL``
     - ``[0.331773, 0.864821]``
     - ``0.546594``
     - ``-0.050441``
     - ``OPTIMAL``
     - ``[0.331773, 0.864821]``
     - ``0.546594``
     - ``-0.050441``
   * - Isolation ``= 0.51``
     - ``OPTIMAL``
     - ``[0.640802, 1.169832]``
     - ``1.160633``
     - ``-0.009963``
     - ``OPTIMAL``
     - ``[0.640802, 1.169832]``
     - ``1.160633``
     - ``-0.009963``
   * - Isolation ``= 0.5``
     - ``OPTIMAL``
     - ``[1.021789, 1.188354]``
     - ``1.560143``
     - ``0.000561``
     - ``OPTIMAL``
     - ``[1.021789, 1.188354]``
     - ``1.560143``
     - ``0.000561``
   * - Isolation ``= 0.1``
     - ``INFEASIBLE``
     - ``-``
     - ``-``
     - ``-``
     - ``INFEASIBLE``
     - ``-``
     - ``-``
     - ``-``

Why ``0.1`` is infeasible
-------------------------

This is not just a solver status. We can prove it on this example.

Let :math:`L_t(x)` be the path length of the leaf reached by :math:`x` in
isolation tree :math:`t`. Let

.. math::

   L(x) = \sum_{t \in \mathcal{T}_{iso}} L_t(x)

be the total isolation-path length over the parsed isolation trees.

Keep the target-class constraints, remove the threshold by setting
``isolation_threshold=1.0``, and maximize the same aggregate isolation-path
length expression used by the MIP and CP backends. On this dataset, the best
target-class point reaches

.. math::

   \max L(x) = 514.377459.

At ``isolation_threshold=0.1``, the model requires

.. math::

   L(x) \ge 1559.821878.

Therefore

.. math::

   514.377459 < 1559.821878,

so no target-class point can satisfy the isolation inequality. The feasible set
is empty for both MIP and CP because they enforce the same path-length cutoff
over the same parsed isolation trees.

Run the example
---------------

.. code-block:: bash

   python examples/isolation_forest_example.py

The script saves the figure directly to
``docs/_static/figures/isolation-forest-example-2d.svg`` and prints the full
comparison table together with the infeasibility proof.
