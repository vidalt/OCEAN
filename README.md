# Optimal Counterfactual Explanations in Tree Ensembles

![Logo](logo.svg)

This repository provides methods to generate optimal counterfactual explanations in tree ensembles.
It is based on the paper *Optimal Counterfactual Explanations in Tree Ensemble* by Axel Parmentier and Thibaut Vidal in the *Proceedings of the thirty-eighth International Conference on Machine Learning*, 2021, in press. The article is [available here](http://proceedings.mlr.press/v139/parmentier21a/parmentier21a.pdf).

## Installation

This project requires the gurobi solver. You can request for a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Once you have installed gurobi, you can install the package with the following command:

```bash
pip install git+hhttps://github.com/eminyous/ocean.git
```

## Usage

The package provides multiple classes and functions to wrap the tree ensemble models from the `scikit-learn` library. A minimal example is provided below:

```python
import gurobipy as gp
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from ocean.feature import parse_features
from ocean.tree import parse_tree
from ocean.model import Model

data, target = load_iris(return_X_y=True, as_frame=True)
mapper, data = parse_features(data)

X, y = data.to_numpy(), target.to_numpy()

rf = RandomForestClassifier(
    n_estimators=10,
    max_depth=3,
    random_state=42,
)
rf.fit(X, y)
trees = [parse_tree(estimator.tree_, mapper=mapper) for estimator in rf.estimators_]

model = Model(trees, mapper)

objective = gp.LinExpr()

for feature in model.features.values():
    if feature.is_one_hot_encoded:
        for code in feature.codes:
            objective += feature[code]
    else:
        objective += feature.x

model.set_objective(objective)

model.optimize()

print(model.solution)

```
