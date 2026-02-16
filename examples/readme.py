from sklearn.ensemble import RandomForestClassifier

from ocean import (
    ConstraintProgrammingExplainer,
    MaxSATExplainer,
    MixedIntegerProgramExplainer,
)
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
x = x.to_numpy().flatten()

# Explain the prediction using MIPEXplainer
mip_model = MixedIntegerProgramExplainer(rf, mapper=mapper)
mip_explanation = mip_model.explain(x, y=1 - y, norm=1)

# Explain the prediction using CPEExplainer
cp_model = ConstraintProgrammingExplainer(rf, mapper=mapper)
cp_explanation = cp_model.explain(x, y=1 - y, norm=1)

maxsat_model = MaxSATExplainer(rf, mapper=mapper)
maxsat_explanation = maxsat_model.explain(x, y=1 - y, norm=1)

# Show the explanations and their objective values
print("MIP objective value:", mip_model.get_objective_value())
print("MIP", mip_explanation, "\n")

print("CP objective value:", cp_model.get_objective_value())
print("CP", cp_explanation, "\n")

print("MaxSAT objective value:", maxsat_model.get_objective_value())
print("MaxSAT", maxsat_explanation, "\n")
