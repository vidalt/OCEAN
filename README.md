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
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ocean.datasets import load_adult
from ocean.explainer import MIPExplainer

# Load the adult dataset
(data, target), mapper = load_adult()

# Generate a random instance from the dataset.
generator = np.random.default_rng(42)
x = generator.choice(data, size=1)

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

# Predict the class of the random instance
y = int(rf.predict(x).item())

# Explain the prediction using MIPEXplainer
model = MIPExplainer(rf, mapper=mapper)
x = x.flatten()
explanation = model.explain(x, y=1 - y, norm=1)

# Show the explanation
print(explanation)
```
