Mathematical Model
==================

This page summarizes the optimization problem solved by OCEAN at the modeling
level. The goal is not to reproduce every implementation detail of every
backend, but to make the shared mathematical structure explicit.

Problem statement
-----------------

Let:

- :math:`\hat{x} \in \mathcal{X}` be the query instance,
- :math:`y` be the desired target class,
- :math:`x` be the counterfactual decision vector.

The ensemble prediction rule is:

.. math::

   \hat{y}(x) \in \arg\max_{c \in \mathcal{Y}} f_c(x).

At a high level, OCEAN solves:

.. math::

   x^\star \in \arg\min_{x \in \mathcal{X}} d(x, \hat{x})
   \quad \text{s.t.} \quad
   f_y(x) \ge f_c(x) + \varepsilon_c,
   \qquad \forall c \neq y.

Here :math:`f_c(x)` is the class-:math:`c` component of the ensemble decision
function, evaluated at the candidate counterfactual :math:`x`. The tie-breaking
constant is defined as:

.. math::

   \varepsilon_c =
   \begin{cases}
   \varepsilon & \text{if } c < y, \\
   0 & \text{otherwise.}
   \end{cases}

This enforces that the target class is strictly preferred, including in the
presence of ties. The default value of :math:`\varepsilon` depends on the
backend:

- MIP uses :math:`\varepsilon = 2^{-16} = 1/65536 \approx 1.5259 \times 10^{-5}`.
- CP uses :math:`\varepsilon = 1` because the score expressions are integer
  scaled.
- MaxSAT uses :math:`\varepsilon = 1` after integerizing the score
  differences used in the pseudo-Boolean encoding.

The MIP backend also uses a numerical strictness constant
:math:`\eta = \texttt{num\_epsilon} = 2^{-6} = 1/64 = 0.015625` for strict
right-branch comparisons in numerical splits.

Feature domains
---------------

The feasible set :math:`\mathcal{X}` depends on the feature type produced by
:func:`ocean.feature.parse_features`.

Continuous feature
   .. math:: x_j \in [a_j, b_j]

Ordinal discrete feature
   .. math:: x_j \in D_j = \{d_{j,1}, \dots, d_{j,m_j}\}

Binary feature
   .. math:: x_j \in \{0, 1\}

Unordered nominal feature
   Encoded as a one-hot block :math:`u_{j,k}` with:

   .. math::

      u_{j,k} \in \{0, 1\},
      \qquad
      \sum_{k \in K_j} u_{j,k} = 1.

This is why ordinal features and one-hot encoded features are treated
differently in OCEAN: ordinal values belong to an ordered finite set, while
unordered categories are modeled as mutually exclusive indicator variables.

Tree path variables
-------------------

For each tree :math:`t` and each leaf :math:`\ell` of that tree, OCEAN creates
path or leaf-selection variables. Abstractly, they satisfy:

.. math::

   p_{t,\ell} \in \{0, 1\},
   \qquad
   \sum_{\ell \in \mathcal{L}_t} p_{t,\ell} = 1.

The equality means exactly one leaf is active in each tree. Path consistency
constraints ensure that if :math:`p_{t,\ell} = 1`, then the feature vector
:math:`x` satisfies every split on the root-to-leaf path leading to
:math:`\ell`.

For a numerical split of the form :math:`x_j \le \tau`, the abstract logic is:

.. math::

   p_{t,\ell} = 1 \implies x_j \le \tau
   \qquad \text{for every left branch on the path,}

.. math::

   p_{t,\ell} = 1 \implies x_j \ge \tau + \eta
   \qquad \text{for every right branch on the path,}

where :math:`\eta` is the numerical strictness constant described above. In
practice, MIP uses explicit linear implications, CP works with threshold
intervals, and MaxSAT reasons over Boolean encodings of the same branch
conditions.

For one-hot encoded nominal features, branch tests are expressed on the
indicator variables rather than on an ordered scalar value. If a branch accepts
exactly the category subset :math:`S \subseteq K_j`, the corresponding path
constraint can be written as:

.. math::

   p_{t,\ell} = 1 \implies \sum_{k \in S} u_{j,k} = 1.

Because the one-hot block already satisfies :math:`\sum_{k \in K_j} u_{j,k}=1`,
this enforces membership in the allowed subset without inventing an artificial
ordering between categories.

Ensemble score aggregation
--------------------------

Let :math:`v_{t,\ell,c}` denote the class contribution of leaf :math:`\ell` in
tree :math:`t` for class :math:`c`, and let :math:`w_t` be the tree weight.
Then the ensemble decision function used by the optimization model can be
written as:

.. math::

   f_c(x) = b_c + \sum_{t \in \mathcal{T}} \sum_{\ell \in \mathcal{L}_t}
   w_t \, v_{t,\ell,c} \, p_{t,\ell}.

The optional base term :math:`b_c` appears for models such as XGBoost where a
base score is part of the ensemble definition.

Distance objectives
-------------------

The optimization objective measures how far the counterfactual is from the
query. OCEAN uses different realizations depending on the backend, but the
common modeling idea is the same.

For an :math:`L_1` objective, a compact description is:

.. math::

   d_1(x, \hat{x}) =
   \sum_{j \in \mathcal{C} \cup \mathcal{O} \cup \mathcal{B}}
   |x_j - \hat{x}_j|
   + \frac{1}{2}
   \sum_{j \in \mathcal{E}} \sum_{k \in K_j}
   |u_{j,k} - \hat{x}_{j,k}|.

The factor :math:`\tfrac{1}{2}` on one-hot blocks avoids double counting a
single category switch, because moving from one active code to another changes
two binary indicators.

The MIP backend also supports an :math:`L_2` variant:

.. math::

   d_2(x, \hat{x}) =
   \sum_{j \in \mathcal{C} \cup \mathcal{O} \cup \mathcal{B}}
   (x_j - \hat{x}_j)^2
   + \frac{1}{2}
   \sum_{j \in \mathcal{E}} \sum_{k \in K_j}
   (u_{j,k} - \hat{x}_{j,k})^2.

More generally, for any integer :math:`p \ge 1`, minimizing
:math:`d_p(x, \hat{x})` is equivalent to minimizing :math:`d_p(x, \hat{x})^p`
because the map :math:`t \mapsto t^p` is strictly increasing on
:math:`\mathbb{R}_{\ge 0}`.

Backend realizations
--------------------

The three backends solve the same conceptual problem with different modeling
primitives.

MIP
   Uses continuous and binary decision variables directly and optimizes either
   :math:`L_1` or :math:`L_2` with linear or quadratic expressions.

CP
   Uses finite-domain integer variables and a scaled integer
   :math:`L_p^p` objective for integer ``norm`` values. Continuous features
   are modeled through interval indices derived from tree thresholds.

MaxSAT
   Encodes the :math:`L_1` objective as weighted soft clauses and the target
   prediction constraints as hard clauses or pseudo-Boolean constraints.

Reading the implementation
--------------------------

If you want to map these equations back to code, the main entry points are:

- ``ocean.mip._model``
- ``ocean.cp._model``
- ``ocean.maxsat._model``

Those modules differ in how they encode the variables and constraints, but they
all implement the same optimization pattern described above.

Backend method reference
------------------------

The following internal backend model classes use the same notation as this
page:

- :math:`x` is the counterfactual decision vector encoded by feature variables,
- :math:`\hat{x}` is the processed query instance; in the implementation, the
  objective-building methods receive it through a parameter named ``x``,
- :math:`y` is the target class,
- :math:`f_c(x)` is the class score for class :math:`c`,
- :math:`p_{t,\ell}` denotes the active leaf or path variable.

Mixed-integer programming model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ocean.mip._model.Model
   :members: build, add_objective, set_majority_class, clear_majority_class, cleanup, _set_majority_class, _set_isolation, _add_objective, L1, L2
   :member-order: bysource

Constraint-programming model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ocean.cp._model.Model
   :members: build, add_objective, set_majority_class, cleanup, _set_majority_class, _add_objective, get_intervals_cost, L1
   :member-order: bysource

Weighted MaxSAT model
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ocean.maxsat._model.Model
   :members: build, add_objective, _add_soft_l1_binary, _add_soft_l1_ohe, _add_soft_l1_continuous, _add_soft_l1_discrete, _get_intervals_cost, set_majority_class, _set_majority_class, _encode_weighted_sum_constraint, cleanup
   :member-order: bysource
