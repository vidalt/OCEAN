# Optimal Counterfactual Explanations in Tree Ensembles

![Logo](logo.svg)

This repository provides methods to generate optimal counterfactual explanations in tree ensembles.
It is based on the paper *Optimal Counterfactual Explanations in Tree Ensemble* by Axel Parmentier and Thibaut Vidal in the *Proceedings of the thirty-eighth International Conference on Machine Learning*, 2021, in press. The article is [available here](http://proceedings.mlr.press/v139/parmentier21a/parmentier21a.pdf).

## Installation

This project requires the gurobi solver. You can request for a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Once you have installed gurobi, you can install the package with the following command:

```bash
pip install git+https://github.com/eminyous/ocean.git
```

## Usage

The package provides multiple classes and functions to wrap the tree ensemble models from the `scikit-learn` library. A minimal example is provided below:

```python
import gurobipy as gp
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ocean.datasets import load_adult
from ocean.explainer import MIPExplainer

mapper, (data, target) = load_adult()

rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

model = MIPExplainer(rf, mapper=mapper, env=env)

generator = np.random.default_rng(42)
i = generator.integers(0, data.shape[0])
x = data.iloc[[i]]
y = int(rf.predict(x)[0])

model.set_majority_class(m_class=1 - y)
model.optimize()

print(model.solution)
```
