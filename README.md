# Optimal Counterfactual Explanations in Tree Ensembles

![Logo](https://github.com/eminyous/ocean/blob/main/logo.svg?raw=True)

This repository provides methods to generate optimal counterfactual explanations in tree ensembles.
It is based on the paper *Optimal Counterfactual Explanations in Tree Ensemble* by Axel Parmentier and Thibaut Vidal in the *Proceedings of the thirty-eighth International Conference on Machine Learning*, 2021, in press. The article is [available here](http://proceedings.mlr.press/v139/parmentier21a/parmentier21a.pdf).

## Installation

This project requires the gurobi solver. You can request for a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Once you have installed gurobi, you can install the package with the following command:

```bash
pip install oceanpy
```

## Usage

The package provides multiple classes and functions to wrap the tree ensemble models from the `scikit-learn` library. A minimal example is provided below:

```python
from sklearn.ensemble import RandomForestClassifier

from ocean import MixedIntegerProgramExplainer, ConstraintProgrammingExplainer
from ocean.datasets import load_adult

# Load the adult dataset
(data, target), mapper = load_adult()

# Select an instance to explain from the dataset
x = data.iloc[0].to_frame().T

# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(data, target)

# Predict the class of the random instance
y = int(rf.predict(x).item())

# Explain the prediction using MIPEXplainer
mip_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
x = x.to_numpy().flatten()
mip_explanation = mip_model.explain(x, y=1 - y, norm=1)

# Explain the prediction using CPEExplainer
cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
x = x.to_numpy().flatten()
cp_explanation = cp_model.explain(x, y=1 - y, norm=1)

# Show the explanation
print("MIP: ",mip_explanation)
print("CP : ",cp_explanation)

```

Expected output:

```plaintext
MIP Explanation:
Age              : 39.0
CapitalGain      : 2174.0
CapitalLoss      : 0
EducationNumber  : 13.0
HoursPerWeek     : 41.0
MaritalStatus    : 3
NativeCountry    : 0
Occupation       : 1
Relationship     : 0
Sex              : 0
WorkClass        : 6
CP Explanation:
Age              : 39.0
CapitalGain      : 2174.0
CapitalLoss      : 0.0
EducationNumber  : 13.0
HoursPerWeek     : 41.0
MaritalStatus    : 3
NativeCountry    : 0
Occupation       : 1
Relationship     : 0
Sex              : 0
WorkClass        : 6
```
